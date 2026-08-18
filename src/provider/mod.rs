pub mod anthropic;
pub mod chat_completion;
pub mod responses;

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::{ProviderConfig, ProviderKind};
use crate::model::{ModelRequest, ModelTurn, StreamEvent};

#[derive(Clone)]
pub struct EventSink(Arc<dyn Fn(StreamEvent) + Send + Sync>);

impl EventSink {
    pub fn new(callback: impl Fn(StreamEvent) + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }

    pub fn emit(&self, event: StreamEvent) {
        (self.0)(event);
    }
}

impl Default for EventSink {
    fn default() -> Self {
        Self::new(|_| {})
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn stream_turn(
        &self,
        request: ModelRequest,
        events: EventSink,
        cancel: CancellationToken,
    ) -> anyhow::Result<ModelTurn>;
}

pub fn create_provider(config: ProviderConfig) -> anyhow::Result<Arc<dyn Provider>> {
    let api_key = config.resolve_api_key()?;
    match config.kind {
        ProviderKind::Responses => Ok(Arc::new(responses::ResponsesProvider::new(
            config, api_key,
        )?)),
        ProviderKind::Anthropic => Ok(Arc::new(anthropic::AnthropicProvider::new(
            config, api_key,
        )?)),
        ProviderKind::Chatcompletion => Ok(Arc::new(chat_completion::ChatCompletionProvider::new(
            config, api_key,
        )?)),
    }
}

pub(crate) fn tool_definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "read",
            "description": "Read a UTF-8 text file with line numbers. Use offset and limit for targeted ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1}
                },
                "required": ["path"],
                "additionalProperties": false
            }
        }),
        serde_json::json!({
            "name": "apply_patch",
            "description": "Apply a deterministic V4A patch. Supports Add File, Delete File, and Update File operations.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": false
            }
        }),
        serde_json::json!({
            "name": "bash",
            "description": "Run a shell command in the current workspace and return bounded stdout, stderr, and exit status.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": false
            }
        }),
    ]
}

pub(crate) fn merge_request_fields(
    target: &mut serde_json::Map<String, serde_json::Value>,
    config: &ProviderConfig,
) {
    for (key, value) in &config.request {
        target.insert(key.clone(), value.clone());
    }
}
