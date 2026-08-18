use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContentBlock {
    Text(String),
    Reasoning(String),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub output: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            output: output.into(),
            is_error: false,
        }
    }

    pub fn error(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            output: output.into(),
            is_error: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMessage {
    pub role: Role,
    pub blocks: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub system_prompt: String,
    pub messages: Vec<ModelMessage>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelTurn {
    pub blocks: Vec<ContentBlock>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub provider_state: Option<serde_json::Value>,
}

impl ModelTurn {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![ContentBlock::Text(text.into())],
            ..Self::default()
        }
    }

    pub fn with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            blocks: tool_calls
                .iter()
                .cloned()
                .map(ContentBlock::ToolCall)
                .collect(),
            tool_calls,
            ..Self::default()
        }
    }

    pub fn final_text(&self) -> Option<String> {
        let text = self
            .blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        (!text.is_empty()).then_some(text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    TextDelta { delta: String },
    ReasoningDelta { delta: String },
    ToolCallStart { id: String, name: String },
    ToolCallArgsDelta { id: String, delta: String },
    ToolCallEnd { id: String },
    ToolExecutionStart { id: String },
    ToolExecutionOutput { id: String, delta: String },
    ToolExecutionEnd { id: String, result: ToolResult },
    Usage(Usage),
    Done,
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationItem {
    pub id: String,
    pub session_id: String,
    pub parent_id: Option<String>,
    pub role: Role,
    pub blocks: Vec<ContentBlock>,
    pub created_at: i64,
}
