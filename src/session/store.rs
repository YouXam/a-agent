use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use uuid::Uuid;

use crate::model::{ContentBlock, ConversationItem, Role, Usage};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    cwd             TEXT NOT NULL,
    client_session_key TEXT,
    title           TEXT,
    head_item_id    TEXT,
    provider_type   TEXT NOT NULL,
    model           TEXT NOT NULL,
    model_profile   TEXT,
    effort          TEXT,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_cwd_updated ON sessions(cwd, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_items (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    parent_id       TEXT,
    role            TEXT NOT NULL,
    kind            TEXT NOT NULL,
    content_json    TEXT NOT NULL,
    usage_json      TEXT,
    created_at      INTEGER NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_items_session ON conversation_items(session_id);
CREATE INDEX IF NOT EXISTS idx_items_parent ON conversation_items(parent_id);

CREATE TABLE IF NOT EXISTS provider_state (
    session_id      TEXT NOT NULL,
    key             TEXT NOT NULL,
    value_json      TEXT NOT NULL,
    PRIMARY KEY(session_id, key),
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS shell_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cwd             TEXT NOT NULL,
    client_session_key TEXT,
    command         TEXT NOT NULL,
    exit_code       INTEGER,
    pipe_status     TEXT,
    started_at      INTEGER NOT NULL,
    duration_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_shell_cwd_time ON shell_history(cwd, started_at DESC);

CREATE TABLE IF NOT EXISTS input_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT NOT NULL,
    created_at      INTEGER NOT NULL
);
"#;

pub const TURN_INTERRUPTED_NOTICE: &str = "[The user interrupted the previous turn. Do not continue or retry its unfinished task unless the user explicitly asks you to.]";
pub const CONVERSATION_SUMMARY_PREFIX: &str = "[Compacted conversation summary]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub id: String,
    pub cwd: String,
    pub title: Option<String>,
    pub head_item_id: Option<String>,
    pub provider_type: String,
    pub model: String,
    pub model_profile: Option<String>,
    pub effort: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct NewSession {
    pub cwd: String,
    pub provider_type: String,
    pub model: String,
    pub client_session_key: Option<String>,
    pub model_profile: Option<String>,
    pub effort: Option<String>,
}

impl NewSession {
    pub fn new(
        cwd: impl Into<String>,
        provider_type: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            cwd: cwd.into(),
            provider_type: provider_type.into(),
            model: model.into(),
            client_session_key: None,
            model_profile: None,
            effort: None,
        }
    }

    pub fn with_client_session_key(mut self, key: impl Into<String>) -> Self {
        self.client_session_key = Some(key.into());
        self
    }

    pub fn with_model_selection(
        mut self,
        profile: impl Into<String>,
        effort: Option<&str>,
    ) -> Self {
        self.model_profile = Some(profile.into());
        self.effort = effort.map(str::to_owned);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellHistoryItem {
    pub id: i64,
    pub cwd: String,
    pub command: String,
    pub exit_code: Option<i32>,
    pub pipe_status: Option<String>,
    pub started_at: i64,
    pub duration_ms: Option<i64>,
}

pub struct SessionStore {
    connection: Connection,
}

impl SessionStore {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create state directory {}", parent.display()))?;
        }
        let connection = Connection::open(path)
            .with_context(|| format!("open session database {}", path.display()))?;
        Self::initialize(connection)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::initialize(Connection::open_in_memory()?)
    }

    fn initialize(connection: Connection) -> Result<Self> {
        connection.busy_timeout(std::time::Duration::from_millis(1000))?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.execute_batch(SCHEMA)?;
        migrate_schema(&connection)?;
        Ok(Self { connection })
    }

    pub fn create_session(&mut self, new: NewSession) -> Result<Session> {
        let id = format!("a_{}", Uuid::new_v4().simple());
        let now = now_millis();
        self.connection.execute(
            "INSERT INTO sessions (id, cwd, client_session_key, provider_type, model, model_profile, effort, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)",
            params![id, new.cwd, new.client_session_key, new.provider_type, new.model, new.model_profile, new.effort, now],
        )?;
        self.get_session(&id)?.context("new session disappeared")
    }

    pub fn get_session(&self, id: &str) -> Result<Option<Session>> {
        self.connection
            .query_row(
                "SELECT id, cwd, title, head_item_id, provider_type, model, model_profile, effort, created_at, updated_at FROM sessions WHERE id = ?1",
                [id],
                row_to_session,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn find_latest_session(&self, cwd: &str) -> Result<Option<Session>> {
        self.connection
            .query_row(
                "SELECT id, cwd, title, head_item_id, provider_type, model, model_profile, effort, created_at, updated_at FROM sessions WHERE cwd = ?1 ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                [cwd],
                row_to_session,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn find_client_session(&self, cwd: &str, key: &str) -> Result<Option<Session>> {
        self.connection
            .query_row(
                "SELECT id, cwd, title, head_item_id, provider_type, model, model_profile, effort, created_at, updated_at FROM sessions WHERE cwd = ?1 AND client_session_key = ?2 LIMIT 1",
                params![cwd, key],
                row_to_session,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn recent_sessions(&self, cwd: &str, limit: usize) -> Result<Vec<Session>> {
        let mut statement = self.connection.prepare(
            "SELECT id, cwd, title, head_item_id, provider_type, model, model_profile, effort, created_at, updated_at FROM sessions WHERE cwd = ?1 AND EXISTS (SELECT 1 FROM conversation_items WHERE conversation_items.session_id = sessions.id AND conversation_items.kind = 'user_message') ORDER BY updated_at DESC, rowid DESC LIMIT ?2",
        )?;
        let rows = statement.query_map(params![cwd, limit], row_to_session)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn first_user_prompt(&self, session_id: &str) -> Result<Option<String>> {
        let content = self
            .connection
            .query_row(
                "SELECT content_json FROM conversation_items WHERE session_id = ?1 AND kind = 'user_message' ORDER BY created_at ASC, rowid ASC LIMIT 1",
                [session_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        content
            .map(|content| {
                let blocks: Vec<ContentBlock> = serde_json::from_str(&content)?;
                Ok(blocks.into_iter().find_map(|block| match block {
                    ContentBlock::Text(text) => Some(text),
                    _ => None,
                }))
            })
            .transpose()
            .map(Option::flatten)
    }

    pub fn rebind_client_session_key(
        &mut self,
        cwd: &str,
        key: &str,
        target_session_id: &str,
    ) -> Result<()> {
        let transaction = self.connection.transaction()?;
        let (target_cwd, target_key): (String, Option<String>) = transaction
            .query_row(
                "SELECT cwd, client_session_key FROM sessions WHERE id = ?1",
                [target_session_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .with_context(|| format!("session not found: {target_session_id}"))?;
        if target_cwd != cwd {
            anyhow::bail!("cannot resume a session from a different cwd");
        }
        if target_key.as_deref().is_some_and(|target| target != key) {
            anyhow::bail!("session is active in another Fish process");
        }
        transaction.execute(
            "UPDATE sessions SET client_session_key = NULL WHERE cwd = ?1 AND client_session_key = ?2",
            params![cwd, key],
        )?;
        transaction.execute(
            "UPDATE sessions SET client_session_key = ?1, updated_at = ?2 WHERE id = ?3",
            params![key, now_millis(), target_session_id],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn append_item(
        &mut self,
        session_id: &str,
        role: Role,
        blocks: Vec<ContentBlock>,
    ) -> Result<ConversationItem> {
        let kind = item_kind(role, &blocks);
        self.append_item_with_kind(session_id, role, blocks, kind, None)
    }

    pub fn append_assistant_item(
        &mut self,
        session_id: &str,
        blocks: Vec<ContentBlock>,
        usage: Option<Usage>,
    ) -> Result<ConversationItem> {
        let kind = item_kind(Role::Assistant, &blocks);
        self.append_item_with_kind(session_id, Role::Assistant, blocks, kind, usage)
    }

    pub fn append_turn_interrupted(&mut self, session_id: &str) -> Result<ConversationItem> {
        self.append_item_with_kind(
            session_id,
            Role::User,
            vec![ContentBlock::Text(TURN_INTERRUPTED_NOTICE.into())],
            "turn_interrupted",
            None,
        )
    }

    pub fn replace_branch_with_summary(
        &mut self,
        session_id: &str,
        summary: &str,
    ) -> Result<ConversationItem> {
        self.get_session(session_id)?
            .with_context(|| format!("session not found: {session_id}"))?;
        let id = format!("i_{}", Uuid::new_v4().simple());
        let created_at = now_millis();
        let blocks = vec![ContentBlock::Text(format!(
            "{CONVERSATION_SUMMARY_PREFIX}\n{summary}"
        ))];
        let content_json = serde_json::to_string(&blocks)?;
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO conversation_items (id, session_id, parent_id, role, kind, content_json, usage_json, created_at) VALUES (?1, ?2, NULL, 'user', 'conversation_summary', ?3, NULL, ?4)",
            params![id, session_id, content_json, created_at],
        )?;
        transaction.execute(
            "UPDATE sessions SET head_item_id = ?1, updated_at = ?2 WHERE id = ?3",
            params![id, created_at, session_id],
        )?;
        transaction.commit()?;
        Ok(ConversationItem {
            id,
            session_id: session_id.into(),
            parent_id: None,
            role: Role::User,
            blocks,
            usage: None,
            created_at,
        })
    }

    fn append_item_with_kind(
        &mut self,
        session_id: &str,
        role: Role,
        blocks: Vec<ContentBlock>,
        kind: &str,
        usage: Option<Usage>,
    ) -> Result<ConversationItem> {
        let parent_id = self
            .get_session(session_id)?
            .with_context(|| format!("session not found: {session_id}"))?
            .head_item_id;
        let id = format!("i_{}", Uuid::new_v4().simple());
        let created_at = now_millis();
        let content_json = serde_json::to_string(&blocks)?;
        let usage_json = usage
            .map(|usage| serde_json::to_string(&usage))
            .transpose()?;
        let role_text = role_to_str(role);
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO conversation_items (id, session_id, parent_id, role, kind, content_json, usage_json, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![id, session_id, parent_id, role_text, kind, content_json, usage_json, created_at],
        )?;
        transaction.execute(
            "UPDATE sessions SET head_item_id = ?1, updated_at = ?2 WHERE id = ?3",
            params![id, created_at, session_id],
        )?;
        transaction.commit()?;
        Ok(ConversationItem {
            id,
            session_id: session_id.into(),
            parent_id,
            role,
            blocks,
            usage,
            created_at,
        })
    }

    pub fn get_item(&self, id: &str) -> Result<Option<ConversationItem>> {
        self.connection
            .query_row(
                "SELECT id, session_id, parent_id, role, content_json, usage_json, created_at FROM conversation_items WHERE id = ?1",
                [id],
                row_to_item,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn active_branch(&self, session_id: &str) -> Result<Vec<ConversationItem>> {
        let mut current = self
            .get_session(session_id)?
            .with_context(|| format!("session not found: {session_id}"))?
            .head_item_id;
        let mut branch = Vec::new();
        while let Some(id) = current {
            let item = self
                .get_item(&id)?
                .with_context(|| format!("conversation item not found: {id}"))?;
            current = item.parent_id.clone();
            branch.push(item);
        }
        branch.reverse();
        Ok(branch)
    }

    pub fn rewind(&mut self, session_id: &str, item_id: &str) -> Result<()> {
        let item = self
            .get_item(item_id)?
            .with_context(|| format!("rewind item not found: {item_id}"))?;
        if item.session_id != session_id {
            anyhow::bail!("rewind item does not belong to session {session_id}");
        }
        self.connection.execute(
            "UPDATE sessions SET head_item_id = ?1, updated_at = ?2 WHERE id = ?3",
            params![item_id, now_millis(), session_id],
        )?;
        Ok(())
    }

    pub fn user_checkpoints(&self, session_id: &str) -> Result<Vec<ConversationItem>> {
        let mut statement = self.connection.prepare(
            "SELECT id, session_id, parent_id, role, content_json, usage_json, created_at FROM conversation_items WHERE session_id = ?1 AND kind = 'user_message' ORDER BY created_at ASC, rowid ASC",
        )?;
        let rows = statement.query_map([session_id], row_to_item)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn count_items(&self, session_id: &str) -> Result<usize> {
        Ok(self.connection.query_row(
            "SELECT COUNT(*) FROM conversation_items WHERE session_id = ?1",
            [session_id],
            |row| row.get(0),
        )?)
    }

    pub fn update_model_selection(
        &self,
        session_id: &str,
        provider_type: &str,
        model: &str,
        model_profile: &str,
        effort: Option<&str>,
    ) -> Result<()> {
        self.connection.execute(
            "UPDATE sessions SET provider_type = ?1, model = ?2, model_profile = ?3, effort = ?4, updated_at = ?5 WHERE id = ?6",
            params![provider_type, model, model_profile, effort, now_millis(), session_id],
        )?;
        Ok(())
    }

    pub fn clear_session(&self, session_id: &str) -> Result<()> {
        self.connection.execute(
            "UPDATE sessions SET head_item_id = NULL, updated_at = ?1 WHERE id = ?2",
            params![now_millis(), session_id],
        )?;
        Ok(())
    }

    pub fn record_input_history(&self, text: &str) -> Result<()> {
        self.connection.execute(
            "INSERT INTO input_history (text, created_at) VALUES (?1, ?2)",
            params![text, now_millis()],
        )?;
        Ok(())
    }

    pub fn recent_input_history(&self, limit: usize) -> Result<Vec<String>> {
        let mut statement = self
            .connection
            .prepare("SELECT text FROM input_history ORDER BY id DESC LIMIT ?1")?;
        let rows = statement.query_map([limit], |row| row.get::<_, String>(0))?;
        let mut entries = rows.collect::<rusqlite::Result<Vec<_>>>()?;
        entries.reverse();
        Ok(entries)
    }

    pub fn prune_input_history(&self, maximum: usize) -> Result<()> {
        self.connection.execute(
            "DELETE FROM input_history WHERE id NOT IN (SELECT id FROM input_history ORDER BY id DESC LIMIT ?1)",
            [maximum],
        )?;
        Ok(())
    }

    pub fn set_provider_state(
        &self,
        session_id: &str,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<()> {
        self.connection.execute(
            "INSERT INTO provider_state (session_id, key, value_json) VALUES (?1, ?2, ?3) ON CONFLICT(session_id, key) DO UPDATE SET value_json = excluded.value_json",
            params![session_id, key, serde_json::to_string(value)?],
        )?;
        Ok(())
    }

    pub fn provider_state(&self, session_id: &str, key: &str) -> Result<Option<serde_json::Value>> {
        let value = self
            .connection
            .query_row(
                "SELECT value_json FROM provider_state WHERE session_id = ?1 AND key = ?2",
                params![session_id, key],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        value
            .map(|value| serde_json::from_str(&value).map_err(Into::into))
            .transpose()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_shell_history(
        &self,
        cwd: &str,
        client_session_key: Option<&str>,
        command: &str,
        exit_code: Option<i32>,
        started_at: i64,
        duration_ms: Option<i64>,
        pipe_status: Option<&str>,
    ) -> Result<()> {
        self.connection.execute(
            "INSERT INTO shell_history (cwd, client_session_key, command, exit_code, pipe_status, started_at, duration_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![cwd, client_session_key, command, exit_code, pipe_status, started_at, duration_ms],
        )?;
        Ok(())
    }

    pub fn recent_shell_history(
        &self,
        cwd: &str,
        client_session_key: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ShellHistoryItem>> {
        let query = if client_session_key.is_some() {
            "SELECT id, cwd, command, exit_code, pipe_status, started_at, duration_ms FROM shell_history WHERE cwd = ?1 AND client_session_key = ?2 ORDER BY started_at DESC, id DESC LIMIT ?3"
        } else {
            "SELECT id, cwd, command, exit_code, pipe_status, started_at, duration_ms FROM shell_history WHERE cwd = ?1 ORDER BY started_at DESC, id DESC LIMIT ?3"
        };
        let mut statement = self.connection.prepare(query)?;
        let rows = statement.query_map(params![cwd, client_session_key, limit], |row| {
            Ok(ShellHistoryItem {
                id: row.get(0)?,
                cwd: row.get(1)?,
                command: row.get(2)?,
                exit_code: row.get(3)?,
                pipe_status: row.get(4)?,
                started_at: row.get(5)?,
                duration_ms: row.get(6)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn prune_shell_history(&self, maximum: usize) -> Result<()> {
        self.connection.execute(
            "DELETE FROM shell_history WHERE id NOT IN (SELECT id FROM shell_history ORDER BY started_at DESC, id DESC LIMIT ?1)",
            [maximum],
        )?;
        Ok(())
    }

    pub fn shell_history_count(&self) -> Result<usize> {
        Ok(self
            .connection
            .query_row("SELECT COUNT(*) FROM shell_history", [], |row| row.get(0))?)
    }
}

pub fn default_database_path(home: &Path) -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".local/state"))
        .join("a/sessions.db")
}

fn migrate_schema(connection: &Connection) -> Result<()> {
    ensure_column(connection, "sessions", "client_session_key", "TEXT")?;
    ensure_column(connection, "sessions", "model_profile", "TEXT")?;
    ensure_column(connection, "sessions", "effort", "TEXT")?;
    ensure_column(connection, "shell_history", "client_session_key", "TEXT")?;
    ensure_column(connection, "conversation_items", "usage_json", "TEXT")?;
    connection.execute_batch(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_client_key ON sessions(cwd, client_session_key) WHERE client_session_key IS NOT NULL;
         CREATE INDEX IF NOT EXISTS idx_shell_client_time ON shell_history(cwd, client_session_key, started_at DESC);
         PRAGMA user_version = 4;",
    )?;
    Ok(())
}

fn ensure_column(connection: &Connection, table: &str, column: &str, kind: &str) -> Result<()> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if !columns.iter().any(|existing| existing == column) {
        connection.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {kind}"),
            [],
        )?;
    }
    Ok(())
}

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    Ok(Session {
        id: row.get(0)?,
        cwd: row.get(1)?,
        title: row.get(2)?,
        head_item_id: row.get(3)?,
        provider_type: row.get(4)?,
        model: row.get(5)?,
        model_profile: row.get(6)?,
        effort: row.get(7)?,
        created_at: row.get(8)?,
        updated_at: row.get(9)?,
    })
}

fn row_to_item(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConversationItem> {
    let role: String = row.get(3)?;
    let content: String = row.get(4)?;
    let usage: Option<String> = row.get(5)?;
    Ok(ConversationItem {
        id: row.get(0)?,
        session_id: row.get(1)?,
        parent_id: row.get(2)?,
        role: str_to_role(&role).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, error.into())
        })?,
        blocks: serde_json::from_str(&content).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, error.into())
        })?,
        usage: usage
            .map(|usage| serde_json::from_str(&usage))
            .transpose()
            .map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    5,
                    rusqlite::types::Type::Text,
                    error.into(),
                )
            })?,
        created_at: row.get(6)?,
    })
}

fn role_to_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn str_to_role(role: &str) -> Result<Role, String> {
    match role {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        _ => Err(format!("unknown conversation role: {role}")),
    }
}

fn item_kind(role: Role, blocks: &[ContentBlock]) -> &'static str {
    match role {
        Role::User => "user_message",
        Role::Tool => "tool_result",
        Role::System => "system_checkpoint",
        Role::Assistant
            if blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_))) =>
        {
            "tool_call"
        }
        Role::Assistant
            if blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::Reasoning(_))) =>
        {
            "assistant_reasoning"
        }
        Role::Assistant => "assistant_text",
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
