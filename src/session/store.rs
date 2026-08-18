use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use uuid::Uuid;

use crate::model::{ContentBlock, ConversationItem, Role};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    cwd             TEXT NOT NULL,
    title           TEXT,
    head_item_id    TEXT,
    provider_type   TEXT NOT NULL,
    model           TEXT NOT NULL,
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
    command         TEXT NOT NULL,
    exit_code       INTEGER,
    pipe_status     TEXT,
    started_at      INTEGER NOT NULL,
    duration_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_shell_cwd_time ON shell_history(cwd, started_at DESC);
PRAGMA user_version = 1;
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub id: String,
    pub cwd: String,
    pub title: Option<String>,
    pub head_item_id: Option<String>,
    pub provider_type: String,
    pub model: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct NewSession {
    pub cwd: String,
    pub provider_type: String,
    pub model: String,
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
        }
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
        Ok(Self { connection })
    }

    pub fn create_session(&mut self, new: NewSession) -> Result<Session> {
        let id = format!("a_{}", Uuid::new_v4().simple());
        let now = now_millis();
        self.connection.execute(
            "INSERT INTO sessions (id, cwd, provider_type, model, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            params![id, new.cwd, new.provider_type, new.model, now],
        )?;
        self.get_session(&id)?.context("new session disappeared")
    }

    pub fn get_session(&self, id: &str) -> Result<Option<Session>> {
        self.connection
            .query_row(
                "SELECT id, cwd, title, head_item_id, provider_type, model, created_at, updated_at FROM sessions WHERE id = ?1",
                [id],
                row_to_session,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn find_latest_session(&self, cwd: &str) -> Result<Option<Session>> {
        self.connection
            .query_row(
                "SELECT id, cwd, title, head_item_id, provider_type, model, created_at, updated_at FROM sessions WHERE cwd = ?1 ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                [cwd],
                row_to_session,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn append_item(
        &mut self,
        session_id: &str,
        role: Role,
        blocks: Vec<ContentBlock>,
    ) -> Result<ConversationItem> {
        let parent_id = self
            .get_session(session_id)?
            .with_context(|| format!("session not found: {session_id}"))?
            .head_item_id;
        let id = format!("i_{}", Uuid::new_v4().simple());
        let created_at = now_millis();
        let content_json = serde_json::to_string(&blocks)?;
        let role_text = role_to_str(role);
        let kind = item_kind(role, &blocks);
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO conversation_items (id, session_id, parent_id, role, kind, content_json, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, session_id, parent_id, role_text, kind, content_json, created_at],
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
            created_at,
        })
    }

    pub fn get_item(&self, id: &str) -> Result<Option<ConversationItem>> {
        self.connection
            .query_row(
                "SELECT id, session_id, parent_id, role, content_json, created_at FROM conversation_items WHERE id = ?1",
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
        Ok(self
            .active_branch(session_id)?
            .into_iter()
            .filter(|item| item.role == Role::User)
            .collect())
    }

    pub fn count_items(&self, session_id: &str) -> Result<usize> {
        Ok(self.connection.query_row(
            "SELECT COUNT(*) FROM conversation_items WHERE session_id = ?1",
            [session_id],
            |row| row.get(0),
        )?)
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
        command: &str,
        exit_code: Option<i32>,
        started_at: i64,
        duration_ms: Option<i64>,
        pipe_status: Option<&str>,
    ) -> Result<()> {
        self.connection.execute(
            "INSERT INTO shell_history (cwd, command, exit_code, pipe_status, started_at, duration_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![cwd, command, exit_code, pipe_status, started_at, duration_ms],
        )?;
        Ok(())
    }

    pub fn recent_shell_history(&self, cwd: &str, limit: usize) -> Result<Vec<ShellHistoryItem>> {
        let mut statement = self.connection.prepare(
            "SELECT id, cwd, command, exit_code, pipe_status, started_at, duration_ms FROM shell_history WHERE cwd = ?1 ORDER BY started_at DESC, id DESC LIMIT ?2",
        )?;
        let rows = statement.query_map(params![cwd, limit], |row| {
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

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    Ok(Session {
        id: row.get(0)?,
        cwd: row.get(1)?,
        title: row.get(2)?,
        head_item_id: row.get(3)?,
        provider_type: row.get(4)?,
        model: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

fn row_to_item(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConversationItem> {
    let role: String = row.get(3)?;
    let content: String = row.get(4)?;
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
        created_at: row.get(5)?,
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
