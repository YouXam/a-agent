use std::sync::{Arc, Mutex};

use a_agent::agent::Agent;
use a_agent::model::{ContentBlock, ModelRequest, ModelTurn, Role, ToolCall, ToolResult};
use a_agent::provider::{EventSink, Provider};
use a_agent::session::{NewSession, SessionStore};
use a_agent::tools::runner::{ToolExecutor, ToolRunner};
use async_trait::async_trait;
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
                &format!("cmd {index}"),
                Some(index),
                index as i64,
                Some(10),
                None,
            )
            .unwrap();
    }
    let recent = store.recent_shell_history("/repo", 2).unwrap();
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

    let result = agent
        .submit("fix it", EventSink::default(), CancellationToken::new())
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
