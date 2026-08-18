use std::collections::HashMap;

use anyhow::{Context, Result};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::stream::StreamResponse;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::config::ProviderConfig;
use crate::model::{ContentBlock, ModelRequest, ModelTurn, Role, StreamEvent, ToolCall, Usage};

use super::{EventSink, Provider, merge_request_fields, tool_definitions};

pub struct ResponsesProvider {
    client: Client<OpenAIConfig>,
    config: ProviderConfig,
}

impl ResponsesProvider {
    pub fn new(config: ProviderConfig, api_key: String) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".into());
        let mut sdk_config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url.trim_end_matches('/'));
        for (key, value) in &config.headers {
            let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())?;
            sdk_config = sdk_config.with_header(name, value.as_str())?;
        }
        Ok(Self {
            client: Client::with_config(sdk_config),
            config,
        })
    }

    fn request_body(&self, request: ModelRequest) -> Value {
        let mut input = Vec::new();
        for message in request.messages {
            match message.role {
                Role::User => {
                    let text = text_blocks(&message.blocks);
                    if !text.is_empty() {
                        input.push(serde_json::json!({"role":"user","content":text}));
                    }
                }
                Role::Assistant => {
                    let text = text_blocks(&message.blocks);
                    if !text.is_empty() {
                        input.push(serde_json::json!({"role":"assistant","content":text}));
                    }
                    for block in message.blocks {
                        if let ContentBlock::ToolCall(call) = block {
                            input.push(serde_json::json!({
                                "type":"function_call", "call_id":call.id,
                                "name":call.name, "arguments":call.arguments
                            }));
                        }
                    }
                }
                Role::Tool => {
                    for block in message.blocks {
                        if let ContentBlock::ToolResult(result) = block {
                            input.push(serde_json::json!({
                                "type":"function_call_output", "call_id":result.call_id,
                                "output":result.output
                            }));
                        }
                    }
                }
                Role::System => {}
            }
        }
        let tools = tool_definitions()
            .into_iter()
            .map(|mut tool| {
                tool.as_object_mut()
                    .expect("tool definition is an object")
                    .insert("type".into(), Value::String("function".into()));
                tool
            })
            .collect::<Vec<_>>();
        let mut body = serde_json::Map::new();
        merge_request_fields(&mut body, &self.config);
        body.insert("model".into(), Value::String(self.config.model.clone()));
        body.insert("instructions".into(), Value::String(request.system_prompt));
        body.insert("input".into(), Value::Array(input));
        body.insert("tools".into(), Value::Array(tools));
        body.insert("stream".into(), Value::Bool(true));
        Value::Object(body)
    }
}

#[async_trait]
impl Provider for ResponsesProvider {
    async fn stream_turn(
        &self,
        request: ModelRequest,
        events: EventSink,
        cancel: CancellationToken,
    ) -> Result<ModelTurn> {
        let responses = self.client.responses();
        let create = responses.create_stream_byot(self.request_body(request));
        tokio::pin!(create);
        let mut stream: StreamResponse<Value> = tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("Responses API request cancelled"),
            result = &mut create => result.context("start Responses API stream")?,
        };
        let mut values = Vec::new();
        let mut live = ResponsesLive::default();
        loop {
            tokio::select! {
                _ = cancel.cancelled() => anyhow::bail!("Responses API request cancelled"),
                item = stream.next() => match item {
                    Some(Ok(value)) => {
                        live.emit(&value, &events);
                        values.push(value);
                    }
                    Some(Err(error)) => return Err(error).context("read Responses API stream"),
                    None => break,
                }
            }
        }
        normalize_events(values).map(|(turn, _)| turn)
    }
}

#[derive(Default)]
struct ResponsesLive {
    calls: HashMap<String, String>,
}

impl ResponsesLive {
    fn emit(&mut self, value: &Value, sink: &EventSink) {
        match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "response.output_text.delta" => sink.emit(StreamEvent::TextDelta {
                delta: string(value, "delta"),
            }),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                sink.emit(StreamEvent::ReasoningDelta {
                    delta: string(value, "delta"),
                })
            }
            "response.output_item.added" if value["item"]["type"] == "function_call" => {
                let item = &value["item"];
                let item_id = string(item, "id");
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or(&item_id)
                    .to_owned();
                self.calls.insert(item_id, id.clone());
                sink.emit(StreamEvent::ToolCallStart {
                    id: id.clone(),
                    name: string(item, "name"),
                });
                let arguments = string(item, "arguments");
                if !arguments.is_empty() {
                    sink.emit(StreamEvent::ToolCallArgsDelta {
                        id,
                        delta: arguments,
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(id) = self.calls.get(&string(value, "item_id")) {
                    sink.emit(StreamEvent::ToolCallArgsDelta {
                        id: id.clone(),
                        delta: string(value, "delta"),
                    });
                }
            }
            "response.output_item.done" if value["item"]["type"] == "function_call" => {
                let item_id = string(&value["item"], "id");
                if let Some(id) = self.calls.get(&item_id) {
                    sink.emit(StreamEvent::ToolCallEnd { id: id.clone() });
                }
            }
            "response.completed" => {
                let raw = &value["response"]["usage"];
                sink.emit(StreamEvent::Usage(Usage {
                    input_tokens: raw.get("input_tokens").and_then(Value::as_u64),
                    output_tokens: raw.get("output_tokens").and_then(Value::as_u64),
                    cached_tokens: raw
                        .pointer("/input_tokens_details/cached_tokens")
                        .and_then(Value::as_u64),
                }));
                sink.emit(StreamEvent::Done);
            }
            "error" => sink.emit(StreamEvent::Error {
                message: value.to_string(),
            }),
            _ => {}
        }
    }
}

fn text_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn normalize_events(values: Vec<Value>) -> Result<(ModelTurn, Vec<StreamEvent>)> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls: HashMap<String, ToolCall> = HashMap::new();
    let mut order = Vec::new();
    let mut stream_events = Vec::new();
    let mut usage = None;

    for value in values {
        match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "response.output_text.delta" => {
                let delta = string(&value, "delta");
                text.push_str(&delta);
                stream_events.push(StreamEvent::TextDelta { delta });
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let delta = string(&value, "delta");
                reasoning.push_str(&delta);
                stream_events.push(StreamEvent::ReasoningDelta { delta });
            }
            "response.output_item.added" => {
                let item = &value["item"];
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let item_id = string(item, "id");
                    let call = ToolCall::new(
                        item.get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or(&item_id),
                        string(item, "name"),
                        string(item, "arguments"),
                    );
                    stream_events.push(StreamEvent::ToolCallStart {
                        id: call.id.clone(),
                        name: call.name.clone(),
                    });
                    if !call.arguments.is_empty() {
                        stream_events.push(StreamEvent::ToolCallArgsDelta {
                            id: call.id.clone(),
                            delta: call.arguments.clone(),
                        });
                    }
                    order.push(item_id.clone());
                    calls.insert(item_id, call);
                }
            }
            "response.function_call_arguments.delta" => {
                let item_id = string(&value, "item_id");
                let delta = string(&value, "delta");
                let call = calls.get_mut(&item_id).with_context(|| {
                    format!("arguments for unknown function call item {item_id}")
                })?;
                call.arguments.push_str(&delta);
                stream_events.push(StreamEvent::ToolCallArgsDelta {
                    id: call.id.clone(),
                    delta,
                });
            }
            "response.output_item.done" => {
                let item = &value["item"];
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let item_id = string(item, "id");
                    let final_arguments = string(item, "arguments");
                    if let Some(call) = calls.get_mut(&item_id) {
                        if !final_arguments.is_empty() {
                            call.arguments = final_arguments;
                        }
                        stream_events.push(StreamEvent::ToolCallEnd {
                            id: call.id.clone(),
                        });
                    }
                }
            }
            "response.completed" => {
                let raw = &value["response"]["usage"];
                usage = Some(Usage {
                    input_tokens: raw.get("input_tokens").and_then(Value::as_u64),
                    output_tokens: raw.get("output_tokens").and_then(Value::as_u64),
                    cached_tokens: raw
                        .pointer("/input_tokens_details/cached_tokens")
                        .and_then(Value::as_u64),
                });
            }
            "response.failed" | "response.incomplete" => {
                anyhow::bail!("provider response did not complete: {}", value);
            }
            "error" => {
                let message = string(&value, "message");
                let code = string(&value, "code");
                anyhow::bail!("provider error: {message} ({code})");
            }
            _ => {}
        }
    }
    let tool_calls = order
        .into_iter()
        .filter_map(|id| calls.remove(&id))
        .collect::<Vec<_>>();
    let mut blocks = Vec::new();
    if !reasoning.is_empty() {
        blocks.push(ContentBlock::Reasoning(reasoning));
    }
    if !text.is_empty() {
        blocks.push(ContentBlock::Text(text));
    }
    blocks.extend(tool_calls.iter().cloned().map(ContentBlock::ToolCall));
    if let Some(usage) = usage {
        stream_events.push(StreamEvent::Usage(usage));
    }
    stream_events.push(StreamEvent::Done);
    Ok((
        ModelTurn {
            blocks,
            tool_calls,
            usage,
            provider_state: None,
        },
        stream_events,
    ))
}

fn string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}
