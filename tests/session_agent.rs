use std::sync::{Arc, Mutex};

use a_agent::agent::Agent;
use a_agent::model::{ContentBlock, ModelRequest, ModelTurn, Role, ToolCall, ToolResult, Usage};
use a_agent::provider::{EventSink, Provider};
use a_agent::session::{
    CONVERSATION_SUMMARY_PREFIX, NewSession, SessionStore, TURN_INTERRUPTED_NOTICE,
};
use a_agent::tools::runner::{ToolExecutor, ToolRunner};
use async_trait::async_trait;
use rusqlite::Connection;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

#[test]
fn sessions_resume_by_cwd_and_reconstruct_active_branch() {
    let mut store = SessionStore::open_in_memory().unwrap();
    let session = store
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    let user = store
        .append_item(
            &session.id,
            Role::User,
            vec![ContentBlock::Text("hello".into())],
        )
        .unwrap();
    let assistant = store
        .append_item(
            &session.id,
            Role::Assistant,
            vec![ContentBlock::Text("hi".into())],
        )
        .unwrap();
    assert_eq!(
        store.find_latest_session("/repo").unwrap().unwrap().id,
        session.id
    );
    assert_eq!(
        store
            .active_branch(&session.id)
            .unwrap()
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        [user.id, assistant.id]
    );
    assert_eq!(
        store.first_user_prompt(&session.id).unwrap().as_deref(),
        Some("hello")
    );
}

#[test]
fn assistant_usage_survives_database_reopen() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("sessions.db");
    let session_id = {
        let mut store = SessionStore::open(&path).unwrap();
        let session = store
            .create_session(NewSession::new("/repo", "responses", "model"))
            .unwrap();
        store
            .append_assistant_item(
                &session.id,
                vec![ContentBlock::Text("answer".into())],
                Some(Usage {
                    input_tokens: Some(11),
                    output_tokens: Some(7),
                    cached_tokens: Some(5),
                    cache_write_tokens: Some(3),
                    total_tokens: Some(26),
                }),
            )
            .unwrap();
        session.id
    };

    let store = SessionStore::open(&path).unwrap();
    let branch = store.active_branch(&session_id).unwrap();
    assert_eq!(branch[0].usage.unwrap().context_tokens(), Some(26));
}

#[test]
fn session_persists_model_profile_and_effort_changes() {
    let mut store = SessionStore::open_in_memory().unwrap();
    let session = store
        .create_session(
            NewSession::new("/repo", "responses", "gpt-fast")
                .with_model_selection("fast", Some("low")),
        )
        .unwrap();
    assert_eq!(session.model_profile.as_deref(), Some("fast"));
    assert_eq!(session.effort.as_deref(), Some("low"));

    store
        .update_model_selection(
            &session.id,
            "anthropic",
            "claude-deep",
            "claude",
            Some("high"),
        )
        .unwrap();
    let resumed = store.get_session(&session.id).unwrap().unwrap();
    assert_eq!(resumed.provider_type, "anthropic");
    assert_eq!(resumed.model, "claude-deep");
    assert_eq!(resumed.model_profile.as_deref(), Some("claude"));
    assert_eq!(resumed.effort.as_deref(), Some("high"));
}

#[test]
fn client_sessions_are_isolated_by_fish_key_and_cwd() {
    let mut store = SessionStore::open_in_memory().unwrap();
    let first = store
        .create_session(
            NewSession::new("/repo", "responses", "model").with_client_session_key("fish-one"),
        )
        .unwrap();
    let second = store
        .create_session(
            NewSession::new("/repo", "responses", "model").with_client_session_key("fish-two"),
        )
        .unwrap();
    let other_cwd = store
        .create_session(
            NewSession::new("/other", "responses", "model").with_client_session_key("fish-one"),
        )
        .unwrap();

    assert_eq!(
        store
            .find_client_session("/repo", "fish-one")
            .unwrap()
            .unwrap()
            .id,
        first.id
    );
    assert_eq!(
        store
            .find_client_session("/repo", "fish-two")
            .unwrap()
            .unwrap()
            .id,
        second.id
    );
    assert_eq!(
        store
            .find_client_session("/other", "fish-one")
            .unwrap()
            .unwrap()
            .id,
        other_cwd.id
    );
}

#[test]
fn recent_sessions_can_rebind_the_current_fish_key() {
    let mut store = SessionStore::open_in_memory().unwrap();
    let original = store
        .create_session(
            NewSession::new("/repo", "responses", "first").with_client_session_key("fish-one"),
        )
        .unwrap();
    let target = store
        .create_session(NewSession::new("/repo", "anthropic", "second"))
        .unwrap();
    store
        .create_session(NewSession::new("/other", "responses", "other"))
        .unwrap();
    store
        .append_item(
            &original.id,
            Role::User,
            vec![ContentBlock::Text("first session".into())],
        )
        .unwrap();
    store
        .append_item(
            &target.id,
            Role::User,
            vec![ContentBlock::Text("target session".into())],
        )
        .unwrap();

    let recent = store.recent_sessions("/repo", 10).unwrap();
    assert_eq!(
        recent
            .iter()
            .map(|session| session.id.as_str())
            .collect::<Vec<_>>(),
        [target.id.as_str(), original.id.as_str()]
    );
    store
        .rebind_client_session_key("/repo", "fish-one", &target.id)
        .unwrap();
    assert_eq!(
        store
            .find_client_session("/repo", "fish-one")
            .unwrap()
            .unwrap()
            .id,
        target.id
    );
}

#[test]
fn turn_interruption_is_model_context_but_not_a_rewind_checkpoint() {
    let mut store = SessionStore::open_in_memory().unwrap();
    let session = store
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    store
        .append_item(
            &session.id,
            Role::User,
            vec![ContentBlock::Text("long task".into())],
        )
        .unwrap();
    store.append_turn_interrupted(&session.id).unwrap();

    let branch = store.active_branch(&session.id).unwrap();
    assert!(
        branch.iter().any(|item| {
            item.blocks == vec![ContentBlock::Text(TURN_INTERRUPTED_NOTICE.into())]
        })
    );
    assert_eq!(store.user_checkpoints(&session.id).unwrap().len(), 1);
}

#[test]
fn rewind_moves_head_without_deleting_old_branch() {
    let mut store = SessionStore::open_in_memory().unwrap();
    let session = store
        .create_session(NewSession::new("/repo", "anthropic", "model"))
        .unwrap();
    let first = store
        .append_item(
            &session.id,
            Role::User,
            vec![ContentBlock::Text("first".into())],
        )
        .unwrap();
    let old = store
        .append_item(
            &session.id,
            Role::Assistant,
            vec![ContentBlock::Text("old".into())],
        )
        .unwrap();
    store.rewind(&session.id, &first.id).unwrap();
    let next = store
        .append_item(
            &session.id,
            Role::User,
            vec![ContentBlock::Text("new".into())],
        )
        .unwrap();
    assert_eq!(
        store
            .active_branch(&session.id)
            .unwrap()
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        [first.id.as_str(), next.id.as_str()]
    );
    assert_eq!(
        store
            .get_item(&old.id)
            .unwrap()
            .unwrap()
            .parent_id
            .as_deref(),
        Some(first.id.as_str())
    );
    assert_eq!(store.count_items(&session.id).unwrap(), 3);
}

#[test]
fn shell_history_is_bounded_and_cwd_scoped() {
    let store = SessionStore::open_in_memory().unwrap();
    for index in 0..5 {
        store
            .record_shell_history(
                "/repo",
                None,
                &format!("cmd {index}"),
                Some(index),
                index as i64,
                Some(10),
                None,
            )
            .unwrap();
    }
    let recent = store.recent_shell_history("/repo", None, 2).unwrap();
    assert_eq!(
        recent
            .iter()
            .map(|item| item.command.as_str())
            .collect::<Vec<_>>(),
        ["cmd 4", "cmd 3"]
    );
    store.prune_shell_history(3).unwrap();
    assert_eq!(store.shell_history_count().unwrap(), 3);
}

#[test]
fn shell_history_can_be_scoped_to_one_fish_session() {
    let store = SessionStore::open_in_memory().unwrap();
    for (key, command, timestamp) in [
        ("fish-one", "cargo test", 1),
        ("fish-two", "git status", 2),
        ("fish-one", "cargo clippy", 3),
    ] {
        store
            .record_shell_history(
                "/repo",
                Some(key),
                command,
                Some(0),
                timestamp,
                Some(10),
                None,
            )
            .unwrap();
    }
    let first = store
        .recent_shell_history("/repo", Some("fish-one"), 5)
        .unwrap();
    assert_eq!(
        first
            .iter()
            .map(|item| item.command.as_str())
            .collect::<Vec<_>>(),
        ["cargo clippy", "cargo test"]
    );
}

#[test]
fn input_history_is_global_chronological_and_bounded() {
    let store = SessionStore::open_in_memory().unwrap();
    for entry in ["first prompt", "second prompt", "third prompt"] {
        store.record_input_history(entry).unwrap();
    }
    assert_eq!(
        store.recent_input_history(2).unwrap(),
        ["second prompt", "third prompt"]
    );
    store.prune_input_history(2).unwrap();
    assert_eq!(
        store.recent_input_history(10).unwrap(),
        ["second prompt", "third prompt"]
    );
}

#[test]
fn opening_a_v1_database_adds_client_session_columns() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("sessions.db");
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE sessions (
                id TEXT PRIMARY KEY, cwd TEXT NOT NULL, title TEXT, head_item_id TEXT,
                provider_type TEXT NOT NULL, model TEXT NOT NULL,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
             );
             CREATE TABLE shell_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, cwd TEXT NOT NULL, command TEXT NOT NULL,
                exit_code INTEGER, pipe_status TEXT, started_at INTEGER NOT NULL, duration_ms INTEGER
             );
             PRAGMA user_version = 1;",
        )
        .unwrap();
    drop(connection);

    let mut store = SessionStore::open(&path).unwrap();
    let session = store
        .create_session(
            NewSession::new("/repo", "responses", "model").with_client_session_key("fish-one"),
        )
        .unwrap();
    assert_eq!(
        store
            .find_client_session("/repo", "fish-one")
            .unwrap()
            .unwrap()
            .id,
        session.id
    );
}

struct SequenceProvider {
    turns: Mutex<Vec<ModelTurn>>,
    request_count: AtomicCounter,
}

#[derive(Default)]
struct AtomicCounter(std::sync::atomic::AtomicUsize);

#[async_trait]
impl Provider for SequenceProvider {
    async fn stream_turn(
        &self,
        _request: ModelRequest,
        _events: EventSink,
        _cancel: CancellationToken,
    ) -> anyhow::Result<ModelTurn> {
        self.request_count
            .0
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(self.turns.lock().unwrap().remove(0))
    }
}

struct FakeTools;

#[async_trait]
impl ToolExecutor for FakeTools {
    async fn execute(&self, call: ToolCall) -> ToolResult {
        ToolResult::success(call.id, "1: code")
    }
}

#[tokio::test]
async fn one_logical_turn_runs_tools_until_final_response() {
    let provider = Arc::new(SequenceProvider {
        turns: Mutex::new(vec![
            ModelTurn::with_tools(vec![ToolCall::new("call_1", "read", "{\"path\":\"a.rs\"}")]),
            ModelTurn::text("fixed"),
        ]),
        request_count: AtomicCounter::default(),
    });
    let tools = Arc::new(ToolRunner::new(Arc::new(FakeTools), 8));
    let store = Arc::new(Mutex::new(SessionStore::open_in_memory().unwrap()));
    let session = store
        .lock()
        .unwrap()
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    let agent = Agent::new(
        provider.clone(),
        tools,
        store.clone(),
        session.id.clone(),
        "system".into(),
        10,
    );
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = events.clone();
    let sink = EventSink::new(move |event| captured_events.lock().unwrap().push(event));

    let result = agent
        .submit("fix it", sink, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.final_text.as_deref(), Some("fixed"));
    assert_eq!(
        provider
            .request_count
            .0
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, a_agent::model::StreamEvent::GenerationStart))
            .count(),
        2
    );
    assert_eq!(
        store
            .lock()
            .unwrap()
            .active_branch(&session.id)
            .unwrap()
            .iter()
            .map(|item| item.role)
            .collect::<Vec<_>>(),
        [Role::User, Role::Assistant, Role::Tool, Role::Assistant]
    );
}

#[tokio::test]
async fn auto_compaction_uses_persisted_provider_usage_as_its_token_anchor() {
    let provider = Arc::new(SequenceProvider {
        turns: Mutex::new(vec![
            ModelTurn::text("durable summary"),
            ModelTurn::text("answer"),
        ]),
        request_count: AtomicCounter::default(),
    });
    let tools = Arc::new(ToolRunner::new(Arc::new(FakeTools), 8));
    let store = Arc::new(Mutex::new(SessionStore::open_in_memory().unwrap()));
    let session = store
        .lock()
        .unwrap()
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_item(
            &session.id,
            Role::User,
            vec![ContentBlock::Text("old request".into())],
        )
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_assistant_item(
            &session.id,
            vec![ContentBlock::Text("old answer".into())],
            Some(Usage {
                total_tokens: Some(90),
                ..Usage::default()
            }),
        )
        .unwrap();
    let agent = Agent::new(
        provider.clone(),
        tools,
        store.clone(),
        session.id.clone(),
        "system".into(),
        10,
    )
    .with_context_budget(Some(100), 20);

    let result = agent
        .submit(
            "new request",
            EventSink::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.final_text.as_deref(), Some("answer"));
    assert_eq!(
        provider
            .request_count
            .0
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    let branch = store.lock().unwrap().active_branch(&session.id).unwrap();
    let texts = branch
        .iter()
        .flat_map(|item| &item.blocks)
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        texts[0].starts_with(CONVERSATION_SUMMARY_PREFIX),
        "{texts:?}"
    );
    assert!(texts.contains(&"new request"), "{texts:?}");
    assert!(texts.contains(&"answer"), "{texts:?}");
    assert!(!texts.contains(&"old request"), "{texts:?}");
}

#[tokio::test]
async fn auto_compaction_does_not_reestimate_history_before_a_valid_usage_anchor() {
    let provider = Arc::new(SequenceProvider {
        turns: Mutex::new(vec![ModelTurn::text("answer")]),
        request_count: AtomicCounter::default(),
    });
    let tools = Arc::new(ToolRunner::new(Arc::new(FakeTools), 8));
    let store = Arc::new(Mutex::new(SessionStore::open_in_memory().unwrap()));
    let session = store
        .lock()
        .unwrap()
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_item(
            &session.id,
            Role::User,
            vec![ContentBlock::Text("x".repeat(10_000))],
        )
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_assistant_item(
            &session.id,
            vec![ContentBlock::Text("old answer".into())],
            Some(Usage {
                total_tokens: Some(20),
                ..Usage::default()
            }),
        )
        .unwrap();
    let agent = Agent::new(
        provider.clone(),
        tools,
        store.clone(),
        session.id.clone(),
        "system".into(),
        10,
    )
    .with_context_budget(Some(100), 20);

    let result = agent
        .submit(
            "small follow-up",
            EventSink::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.final_text.as_deref(), Some("answer"));
    assert_eq!(
        provider
            .request_count
            .0
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let branch = store.lock().unwrap().active_branch(&session.id).unwrap();
    assert!(branch.iter().any(|item| {
        item.blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::Text(text) if text.len() == 10_000))
    }));
}

#[test]
fn context_status_combines_provider_usage_with_trailing_estimates() {
    let tools = Arc::new(ToolRunner::new(Arc::new(FakeTools), 8));
    let store = Arc::new(Mutex::new(SessionStore::open_in_memory().unwrap()));
    let session = store
        .lock()
        .unwrap()
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_assistant_item(
            &session.id,
            vec![ContentBlock::Text("anchor".into())],
            Some(Usage {
                total_tokens: Some(100),
                ..Usage::default()
            }),
        )
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_item(
            &session.id,
            Role::Tool,
            vec![ContentBlock::ToolResult(ToolResult::success(
                "call",
                "x".repeat(40),
            ))],
        )
        .unwrap();
    let provider = Arc::new(SequenceProvider {
        turns: Mutex::new(Vec::new()),
        request_count: AtomicCounter::default(),
    });
    let agent = Agent::new(provider, tools, store, session.id, "system".into(), 10)
        .with_context_budget(Some(1000), 100);

    let status = agent.context_status().unwrap();
    assert_eq!(status.used_tokens, 110);
    assert_eq!(status.provider_tokens, Some(100));
    assert_eq!(status.estimated_tokens, 10);
    assert_eq!(status.context_window, Some(1000));
    assert_eq!(status.compact_at, Some(900));
}

#[tokio::test]
async fn auto_compaction_ignores_all_zero_usage() {
    let provider = Arc::new(SequenceProvider {
        turns: Mutex::new(vec![
            ModelTurn::text("durable summary"),
            ModelTurn::text("answer"),
        ]),
        request_count: AtomicCounter::default(),
    });
    let tools = Arc::new(ToolRunner::new(Arc::new(FakeTools), 8));
    let store = Arc::new(Mutex::new(SessionStore::open_in_memory().unwrap()));
    let session = store
        .lock()
        .unwrap()
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_assistant_item(
            &session.id,
            vec![ContentBlock::Text("x".repeat(400))],
            Some(Usage::default()),
        )
        .unwrap();
    let agent = Agent::new(
        provider.clone(),
        tools,
        store.clone(),
        session.id.clone(),
        "system".into(),
        10,
    )
    .with_context_budget(Some(100), 20);

    agent
        .submit("continue", EventSink::default(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        provider
            .request_count
            .0
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    let branch = store.lock().unwrap().active_branch(&session.id).unwrap();
    assert!(is_summary(&branch[0].blocks));
}

#[tokio::test]
async fn manual_compaction_replaces_the_active_branch_without_an_agent_turn() {
    let provider = Arc::new(SequenceProvider {
        turns: Mutex::new(vec![ModelTurn::text("manual summary")]),
        request_count: AtomicCounter::default(),
    });
    let tools = Arc::new(ToolRunner::new(Arc::new(FakeTools), 8));
    let store = Arc::new(Mutex::new(SessionStore::open_in_memory().unwrap()));
    let session = store
        .lock()
        .unwrap()
        .create_session(NewSession::new("/repo", "responses", "model"))
        .unwrap();
    store
        .lock()
        .unwrap()
        .append_item(
            &session.id,
            Role::User,
            vec![ContentBlock::Text("keep this context".into())],
        )
        .unwrap();
    let agent = Agent::new(
        provider.clone(),
        tools,
        store.clone(),
        session.id.clone(),
        "system".into(),
        10,
    );

    assert!(
        agent
            .compact(EventSink::default(), CancellationToken::new())
            .await
            .unwrap()
    );
    assert_eq!(
        provider
            .request_count
            .0
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let branch = store.lock().unwrap().active_branch(&session.id).unwrap();
    assert_eq!(branch.len(), 1);
    assert!(is_summary(&branch[0].blocks));
}

fn is_summary(blocks: &[ContentBlock]) -> bool {
    blocks.iter().any(
        |block| matches!(block, ContentBlock::Text(text) if text.starts_with(CONVERSATION_SUMMARY_PREFIX)),
    )
}
