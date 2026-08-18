use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;

use crate::model::{ContentBlock, ModelMessage, ModelRequest, Role, StreamEvent};
use crate::provider::{EventSink, Provider};
use crate::session::SessionStore;
use crate::tools::runner::ToolRunner;

pub struct Agent {
    provider: Arc<dyn Provider>,
    tools: Arc<ToolRunner>,
    store: Arc<Mutex<SessionStore>>,
    session_id: String,
    system_prompt: String,
    max_cycles: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentResult {
    pub final_text: Option<String>,
    pub cycles: usize,
}

impl Agent {
    pub fn new(
        provider: Arc<dyn Provider>,
        tools: Arc<ToolRunner>,
        store: Arc<Mutex<SessionStore>>,
        session_id: String,
        system_prompt: String,
        max_cycles: usize,
    ) -> Self {
        Self {
            provider,
            tools,
            store,
            session_id,
            system_prompt,
            max_cycles,
        }
    }

    pub async fn submit(
        &self,
        prompt: &str,
        events: EventSink,
        cancel: CancellationToken,
    ) -> Result<AgentResult> {
        self.store
            .lock()
            .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
            .append_item(
                &self.session_id,
                Role::User,
                vec![ContentBlock::Text(prompt.into())],
            )?;

        for cycle in 1..=self.max_cycles {
            if cancel.is_cancelled() {
                anyhow::bail!("agent turn cancelled");
            }
            let messages = self
                .store
                .lock()
                .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
                .active_branch(&self.session_id)?
                .into_iter()
                .map(|item| ModelMessage {
                    role: item.role,
                    blocks: item.blocks,
                })
                .collect();
            let request = ModelRequest {
                system_prompt: self.system_prompt.clone(),
                messages,
            };
            let turn = self
                .provider
                .stream_turn(request, events.clone(), cancel.clone())
                .await?;
            if !turn.blocks.is_empty() {
                self.store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
                    .append_item(&self.session_id, Role::Assistant, turn.blocks.clone())?;
            }
            if let Some(state) = &turn.provider_state {
                self.store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
                    .set_provider_state(&self.session_id, "continuation", state)?;
            }
            if turn.tool_calls.is_empty() {
                events.emit(StreamEvent::Done);
                return Ok(AgentResult {
                    final_text: turn.final_text(),
                    cycles: cycle,
                });
            }
            for call in &turn.tool_calls {
                events.emit(StreamEvent::ToolExecutionStart {
                    id: call.id.clone(),
                });
            }
            let results = self
                .tools
                .execute_with(turn.tool_calls, events.clone(), cancel.clone())
                .await;
            for result in results {
                events.emit(StreamEvent::ToolExecutionEnd {
                    id: result.call_id.clone(),
                    result: result.clone(),
                });
                self.store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
                    .append_item(
                        &self.session_id,
                        Role::Tool,
                        vec![ContentBlock::ToolResult(result)],
                    )?;
            }
        }
        Err(anyhow::anyhow!(
            "maximum agent cycles ({}) exceeded",
            self.max_cycles
        ))
        .context("maximum agent cycles reached before a final response")
    }
}
