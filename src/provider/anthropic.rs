use std::collections::BTreeMap;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use threatflux_anthropic_sdk::models::message::MessageRequest;
use threatflux_anthropic_sdk::{Client, Config, RequestOptions};
use tokio_util::sync::CancellationToken;

use crate::config::ProviderConfig;
use crate::model::{ContentBlock, ModelRequest, ModelTurn, Role, StreamEvent, ToolCall, Usage};

use super::{EventSink, Provider, merge_request_fields, tool_definitions};

pub struct AnthropicProvider {
    client: Client,
    config: ProviderConfig,
    options: RequestOptions,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig, api_key: String) -> Result<Self> {
        let mut sdk_config = Config::new(api_key)?.with_default_model(config.model.clone());
        if let Some(base_url) = &config.base_url {
            sdk_config = sdk_config.with_base_url(url::Url::parse(base_url)?);
        }
        let client = Client::try_new(sdk_config)?;
        let mut options = RequestOptions::new();
        for (key, value) in &config.headers {
            options = options.with_header(key, value);
        }
        Ok(Self {
            client,
            config,
            options,
        })
    }

    fn request(&self, request: ModelRequest) -> Result<MessageRequest> {
        let mut messages: Vec<Value> = Vec::new();
        for message in request.messages {
            let (role, content) = match message.role {
                Role::User => ("user", message.blocks.into_iter().filter_map(|block| match block { ContentBlock::Text(text) => Some(serde_json::json!({"type":"text","text":text})), _ => None }).collect::<Vec<_>>()),
                Role::Assistant => ("assistant", message.blocks.into_iter().filter_map(|block| match block {
                    ContentBlock::Text(text) => Some(serde_json::json!({"type":"text","text":text})),
                    ContentBlock::ToolCall(call) => Some(serde_json::json!({"type":"tool_use","id":call.id,"name":call.name,"input":serde_json::from_str::<Value>(&call.arguments).unwrap_or(Value::String(call.arguments))})),
                    _ => None,
                }).collect()),
                Role::Tool => ("user", message.blocks.into_iter().filter_map(|block| match block { ContentBlock::ToolResult(result) => Some(serde_json::json!({"type":"tool_result","tool_use_id":result.call_id,"content":result.output,"is_error":result.is_error})), _ => None }).collect()),
                Role::System => continue,
            };
            if content.is_empty() {
                continue;
            }
            if messages
                .last()
                .and_then(|item| item.get("role"))
                .and_then(Value::as_str)
                == Some(role)
            {
                messages
                    .last_mut()
                    .and_then(|item| item.get_mut("content"))
                    .and_then(Value::as_array_mut)
                    .expect("message content array")
                    .extend(content);
            } else {
                messages.push(serde_json::json!({"role":role,"content":content}));
            }
        }
        let tools = tool_definitions()
            .into_iter()
            .map(|mut tool| {
                let object = tool.as_object_mut().expect("tool definition object");
                let parameters = object.remove("parameters").expect("parameters");
                object.insert("input_schema".into(), parameters);
                tool
            })
            .collect();
        let mut body = serde_json::Map::new();
        merge_request_fields(&mut body, &self.config);
        body.insert("model".into(), Value::String(self.config.model.clone()));
        body.insert("max_tokens".into(), Value::from(self.config.max_tokens));
        body.insert("system".into(), Value::String(request.system_prompt));
        body.insert("messages".into(), Value::Array(messages));
        body.insert("tools".into(), Value::Array(tools));
        body.insert("stream".into(), Value::Bool(true));
        Ok(serde_json::from_value(Value::Object(body))?)
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn stream_turn(
        &self,
        request: ModelRequest,
        events: EventSink,
        cancel: CancellationToken,
    ) -> Result<ModelTurn> {
        let messages = self.client.messages();
        let create = messages.create_stream(self.request(request)?, Some(self.options.clone()));
        tokio::pin!(create);
        let mut stream = tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("Anthropic request cancelled"),
            result = &mut create => result?,
        };
        let mut values = Vec::new();
        let mut live = AnthropicLive::default();
        loop {
            tokio::select! {
                _ = cancel.cancelled() => anyhow::bail!("Anthropic request cancelled"),
                item = stream.next() => match item {
                    Some(Ok(event)) => {
                        let value = serde_json::to_value(event)?;
                        live.emit(&value, &events);
                        values.push(value);
                    }
                    Some(Err(error)) => return Err(error.into()),
                    None => break,
                }
            }
        }
        normalize_events(values).map(|(turn, _)| turn)
    }
}

#[derive(Default)]
struct AnthropicLive {
    calls: BTreeMap<usize, String>,
}

impl AnthropicLive {
    fn emit(&mut self, value: &Value, sink: &EventSink) {
        let index = value
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "content_block_start" if value["content_block"]["type"] == "tool_use" => {
                let block = &value["content_block"];
                let id = string(block, "id");
                self.calls.insert(index, id.clone());
                sink.emit(StreamEvent::ToolCallStart {
                    id: id.clone(),
                    name: string(block, "name"),
                });
                let input = block
                    .get("input")
                    .filter(|value| !value.as_object().is_some_and(|object| object.is_empty()))
                    .and_then(|value| serde_json::to_string(value).ok());
                if let Some(delta) = input {
                    sink.emit(StreamEvent::ToolCallArgsDelta { id, delta });
                }
            }
            "content_block_delta" => match value["delta"]["type"].as_str().unwrap_or_default() {
                "text_delta" => sink.emit(StreamEvent::TextDelta {
                    delta: string(&value["delta"], "text"),
                }),
                "thinking_delta" => sink.emit(StreamEvent::ReasoningDelta {
                    delta: string(&value["delta"], "thinking"),
                }),
                "input_json_delta" => {
                    if let Some(id) = self.calls.get(&index) {
                        sink.emit(StreamEvent::ToolCallArgsDelta {
                            id: id.clone(),
                            delta: string(&value["delta"], "partial_json"),
                        });
                    }
                }
                _ => {}
            },
            "content_block_stop" => {
                if let Some(id) = self.calls.get(&index) {
                    sink.emit(StreamEvent::ToolCallEnd { id: id.clone() });
                }
            }
            "message_stop" => sink.emit(StreamEvent::Done),
            "error" => sink.emit(StreamEvent::Error {
                message: value.to_string(),
            }),
            _ => {}
        }
    }
}

enum PendingBlock {
    Text(String),
    Reasoning(String),
    Tool(ToolCall),
    Ignore,
}

pub fn normalize_events(values: Vec<Value>) -> Result<(ModelTurn, Vec<StreamEvent>)> {
    let mut blocks = BTreeMap::new();
    let mut events = Vec::new();
    let mut usage = Usage::default();
    let mut has_usage = false;

    for value in values {
        let index = value
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "message_start" => {
                let raw = &value["message"]["usage"];
                usage.input_tokens = raw.get("input_tokens").and_then(Value::as_u64);
                usage.cached_tokens = raw.get("cache_read_input_tokens").and_then(Value::as_u64);
                has_usage = true;
            }
            "content_block_start" => {
                let block = &value["content_block"];
                match block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                {
                    "text" => {
                        blocks.insert(index, PendingBlock::Text(string(block, "text")));
                    }
                    "thinking" | "redacted_thinking" => {
                        blocks.insert(index, PendingBlock::Reasoning(string(block, "thinking")));
                    }
                    "tool_use" => {
                        let input = block
                            .get("input")
                            .cloned()
                            .unwrap_or(Value::Object(Default::default()));
                        let arguments = if input.as_object().is_some_and(|value| value.is_empty()) {
                            String::new()
                        } else {
                            serde_json::to_string(&input)?
                        };
                        let call =
                            ToolCall::new(string(block, "id"), string(block, "name"), arguments);
                        events.push(StreamEvent::ToolCallStart {
                            id: call.id.clone(),
                            name: call.name.clone(),
                        });
                        if !call.arguments.is_empty() {
                            events.push(StreamEvent::ToolCallArgsDelta {
                                id: call.id.clone(),
                                delta: call.arguments.clone(),
                            });
                        }
                        blocks.insert(index, PendingBlock::Tool(call));
                    }
                    _ => {
                        blocks.insert(index, PendingBlock::Ignore);
                    }
                }
            }
            "content_block_delta" => {
                let delta = &value["delta"];
                match (
                    blocks.get_mut(&index),
                    delta.get("type").and_then(Value::as_str),
                ) {
                    (Some(PendingBlock::Text(text)), Some("text_delta")) => {
                        let delta = string(delta, "text");
                        text.push_str(&delta);
                        events.push(StreamEvent::TextDelta { delta });
                    }
                    (Some(PendingBlock::Reasoning(text)), Some("thinking_delta")) => {
                        let delta = string(delta, "thinking");
                        text.push_str(&delta);
                        events.push(StreamEvent::ReasoningDelta { delta });
                    }
                    (Some(PendingBlock::Tool(call)), Some("input_json_delta")) => {
                        let delta = string(delta, "partial_json");
                        call.arguments.push_str(&delta);
                        events.push(StreamEvent::ToolCallArgsDelta {
                            id: call.id.clone(),
                            delta,
                        });
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let Some(PendingBlock::Tool(call)) = blocks.get(&index) {
                    events.push(StreamEvent::ToolCallEnd {
                        id: call.id.clone(),
                    });
                }
            }
            "message_delta" => {
                usage.output_tokens = value["usage"].get("output_tokens").and_then(Value::as_u64);
                has_usage = true;
            }
            "error" => anyhow::bail!("provider error: {}", value["error"]),
            _ => {}
        }
    }

    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    for block in blocks.into_values() {
        match block {
            PendingBlock::Text(value) if !value.is_empty() => {
                content.push(ContentBlock::Text(value))
            }
            PendingBlock::Reasoning(value) if !value.is_empty() => {
                content.push(ContentBlock::Reasoning(value))
            }
            PendingBlock::Tool(call) => {
                tool_calls.push(call.clone());
                content.push(ContentBlock::ToolCall(call));
            }
            _ => {}
        }
    }
    let usage = has_usage.then_some(usage);
    if let Some(usage) = usage {
        events.push(StreamEvent::Usage(usage));
    }
    events.push(StreamEvent::Done);
    Ok((
        ModelTurn {
            blocks: content,
            tool_calls,
            usage,
            provider_state: None,
        },
        events,
    ))
}

fn string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}
