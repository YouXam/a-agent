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
    context_window: Option<u64>,
    max_output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentResult {
    pub final_text: Option<String>,
    pub cycles: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextStatus {
    pub used_tokens: u64,
    pub provider_tokens: Option<u64>,
    pub estimated_tokens: u64,
    pub context_window: Option<u64>,
    pub compact_at: Option<u64>,
    pub max_output_tokens: u64,
}

struct ContextEstimate {
    total: u64,
    provider: Option<u64>,
    estimated: u64,
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
            context_window: None,
            max_output_tokens: 0,
        }
    }

    pub fn with_context_budget(
        mut self,
        context_window: Option<u64>,
        max_output_tokens: u64,
    ) -> Self {
        self.context_window = context_window;
        self.max_output_tokens = max_output_tokens;
        self
    }

    pub async fn submit(
        &self,
        prompt: &str,
        events: EventSink,
        cancel: CancellationToken,
    ) -> Result<AgentResult> {
        self.compact_if_needed(Some(prompt), &events, &cancel)
            .await?;
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
            self.compact_if_needed(None, &events, &cancel).await?;
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
                include_tools: true,
            };
            events.emit(StreamEvent::GenerationStart);
            let turn = self
                .provider
                .stream_turn(request, events.clone(), cancel.clone())
                .await?;
            if !turn.blocks.is_empty() {
                self.store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
                    .append_assistant_item(&self.session_id, turn.blocks.clone(), turn.usage)?;
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

    async fn compact_if_needed(
        &self,
        additional_prompt: Option<&str>,
        events: &EventSink,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let Some(context_window) = self.context_window else {
            return Ok(());
        };
        let threshold = context_window.saturating_sub(self.max_output_tokens);
        let branch = self
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
            .active_branch(&self.session_id)?;
        if let Some(summary_index) = branch.iter().rposition(is_compaction_summary)
            && !branch[summary_index + 1..].iter().any(has_valid_usage)
        {
            return Ok(());
        }
        if branch.is_empty()
            || estimate_context_tokens(&self.system_prompt, &branch, additional_prompt).total
                <= threshold
        {
            return Ok(());
        }
        self.compact_branch(branch, events, cancel).await
    }

    async fn compact_branch(
        &self,
        branch: Vec<crate::model::ConversationItem>,
        events: &EventSink,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let messages = branch
            .iter()
            .map(|item| ModelMessage {
                role: item.role,
                blocks: item.blocks.clone(),
            })
            .collect();
        let request = ModelRequest {
            system_prompt: "Summarize this coding-agent conversation for continuation. Preserve user goals, decisions, modified files, tool results, failures, unresolved work, and exact technical constraints. Return only the compact summary and do not call tools.".into(),
            messages,
            include_tools: false,
        };
        events.emit(StreamEvent::GenerationStart);
        let result = self
            .provider
            .stream_turn(request, EventSink::default(), cancel.clone())
            .await;
        events.emit(StreamEvent::Done);
        let turn = result.context("compact conversation")?;
        if !turn.tool_calls.is_empty() {
            anyhow::bail!("compaction model attempted to call tools");
        }
        let summary = turn
            .final_text()
            .context("compaction model returned no summary")?;
        self.store
            .lock()
            .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
            .replace_branch_with_summary(&self.session_id, &summary)?;
        Ok(())
    }

    pub async fn compact(&self, events: EventSink, cancel: CancellationToken) -> Result<bool> {
        let branch = self
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
            .active_branch(&self.session_id)?;
        if branch.is_empty() {
            return Ok(false);
        }
        self.compact_branch(branch, &events, &cancel).await?;
        Ok(true)
    }

    pub fn context_status(&self) -> Result<ContextStatus> {
        let branch = self
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
            .active_branch(&self.session_id)?;
        let estimate = estimate_context_tokens(&self.system_prompt, &branch, None);
        Ok(ContextStatus {
            used_tokens: estimate.total,
            provider_tokens: estimate.provider,
            estimated_tokens: estimate.estimated,
            context_window: self.context_window,
            compact_at: self
                .context_window
                .map(|window| window.saturating_sub(self.max_output_tokens)),
            max_output_tokens: self.max_output_tokens,
        })
    }

    pub fn record_interruption(&self) -> Result<()> {
        self.store
            .lock()
            .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
            .append_turn_interrupted(&self.session_id)?;
        Ok(())
    }
}

fn is_compaction_summary(item: &crate::model::ConversationItem) -> bool {
    item.blocks.iter().any(|block| {
        matches!(block, ContentBlock::Text(text) if text.starts_with(crate::session::CONVERSATION_SUMMARY_PREFIX))
    })
}

fn has_valid_usage(item: &crate::model::ConversationItem) -> bool {
    item.role == Role::Assistant
        && item
            .usage
            .and_then(|usage| usage.context_tokens())
            .is_some_and(|tokens| tokens > 0)
}

fn estimate_context_tokens(
    system_prompt: &str,
    branch: &[crate::model::ConversationItem],
    additional_prompt: Option<&str>,
) -> ContextEstimate {
    let usage_anchor = branch.iter().enumerate().rev().find_map(|(index, item)| {
        (item.role == Role::Assistant)
            .then_some(item.usage)
            .flatten()
            .and_then(|usage| usage.context_tokens())
            .filter(|tokens| *tokens > 0)
            .map(|tokens| (index, tokens))
    });
    let (start, provider, mut estimated) = usage_anchor.map_or_else(
        || {
            let tools =
                serde_json::to_string(&crate::provider::tool_definitions()).unwrap_or_default();
            (
                0,
                None,
                estimate_text_tokens(system_prompt) + estimate_text_tokens(&tools),
            )
        },
        |(index, tokens)| (index + 1, Some(tokens), 0),
    );
    for item in &branch[start..] {
        estimated += estimate_blocks_tokens(&item.blocks);
    }
    if let Some(prompt) = additional_prompt {
        estimated += estimate_text_tokens(prompt);
    }
    ContextEstimate {
        total: provider.unwrap_or(0) + estimated,
        provider,
        estimated,
    }
}

fn estimate_blocks_tokens(blocks: &[ContentBlock]) -> u64 {
    let characters = blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) | ContentBlock::Reasoning(text) => text.chars().count(),
            ContentBlock::ToolCall(call) => {
                call.name.chars().count() + call.arguments.chars().count()
            }
            ContentBlock::ToolResult(result) => result.output.chars().count(),
        })
        .sum::<usize>();
    characters.div_ceil(4) as u64
}

fn estimate_text_tokens(text: &str) -> u64 {
    text.chars().count().div_ceil(4) as u64
}
