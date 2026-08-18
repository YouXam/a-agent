use std::collections::BTreeMap;

use anyhow::Result;
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

pub struct ChatCompletionProvider {
    client: Client<OpenAIConfig>,
    config: ProviderConfig,
}

impl ChatCompletionProvider {
    pub fn new(config: ProviderConfig, api_key: String) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".into());
        let mut sdk_config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url.trim_end_matches('/'));
        for (key, value) in &config.headers {
            sdk_config = sdk_config.with_header(
                reqwest::header::HeaderName::from_bytes(key.as_bytes())?,
                value.as_str(),
            )?;
        }
        Ok(Self {
            client: Client::with_config(sdk_config),
            config,
        })
    }

    fn request_body(&self, request: ModelRequest) -> Value {
        let mut messages =
            vec![serde_json::json!({"role":"system","content":request.system_prompt})];
        for message in request.messages {
            match message.role {
                Role::User => messages.push(
                    serde_json::json!({"role":"user","content":text_blocks(&message.blocks)}),
                ),
                Role::Assistant => {
                    let calls = message.blocks.iter().filter_map(|block| match block {
                        ContentBlock::ToolCall(call) => Some(serde_json::json!({
                            "id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}
                        })), _ => None
                    }).collect::<Vec<_>>();
                    let text = text_blocks(&message.blocks);
                    let mut item = serde_json::json!({"role":"assistant","content":if text.is_empty() { Value::Null } else { Value::String(text) }});
                    if !calls.is_empty() {
                        item["tool_calls"] = Value::Array(calls);
                    }
                    messages.push(item);
                }
                Role::Tool => {
                    for block in message.blocks {
                        if let ContentBlock::ToolResult(result) = block {
                            messages.push(serde_json::json!({"role":"tool","tool_call_id":result.call_id,"content":result.output}));
                        }
                    }
                }
                Role::System => {}
            }
        }
        let tools = if request.include_tools {
            tool_definitions()
                .into_iter()
                .map(|tool| serde_json::json!({"type":"function","function":tool}))
                .collect()
        } else {
            Vec::new()
        };
        let mut body = serde_json::Map::new();
        merge_request_fields(&mut body, &self.config);
        body.entry("stream_options")
            .or_insert_with(|| serde_json::json!({"include_usage":true}));
        body.insert("model".into(), Value::String(self.config.model.clone()));
        body.insert("max_tokens".into(), Value::from(self.config.max_tokens));
        body.insert("messages".into(), Value::Array(messages));
        body.insert("tools".into(), Value::Array(tools));
        body.insert("stream".into(), Value::Bool(true));
        Value::Object(body)
    }
}

#[async_trait]
impl Provider for ChatCompletionProvider {
    async fn stream_turn(
        &self,
        request: ModelRequest,
        events: EventSink,
        cancel: CancellationToken,
    ) -> Result<ModelTurn> {
        let chat = self.client.chat();
        let create = chat.create_stream_byot(self.request_body(request));
        tokio::pin!(create);
        let mut stream: StreamResponse<Value> = tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("Chat Completions request cancelled"),
            result = &mut create => result?,
        };
        let mut values = Vec::new();
        let mut live = ChatLive::default();
        loop {
            tokio::select! {
                _ = cancel.cancelled() => anyhow::bail!("Chat Completions request cancelled"),
                item = stream.next() => match item {
                    Some(Ok(value)) => { live.emit(&value, &events); values.push(value); }
                    Some(Err(error)) => return Err(error.into()),
                    None => break,
                }
            }
        }
        normalize_events(values).map(|(turn, _)| turn)
    }
}

#[derive(Default)]
struct ChatLive {
    calls: BTreeMap<usize, (String, String, bool)>,
}

impl ChatLive {
    fn emit(&mut self, value: &Value, sink: &EventSink) {
        if let Some(raw) = value.get("usage") {
            sink.emit(StreamEvent::Usage(normalize_usage(raw)));
        }
        let Some(delta) = value.pointer("/choices/0/delta") else {
            return;
        };
        if let Some(part) = delta.get("content").and_then(Value::as_str) {
            sink.emit(StreamEvent::TextDelta { delta: part.into() });
        }
        if let Some(part) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(Value::as_str)
        {
            sink.emit(StreamEvent::ReasoningDelta { delta: part.into() });
        }
        for raw in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = raw.get("index").and_then(Value::as_u64).unwrap_or_default() as usize;
            let state = self.calls.entry(index).or_default();
            if let Some(id) = raw.get("id").and_then(Value::as_str) {
                state.0.push_str(id);
            }
            if let Some(name) = raw.pointer("/function/name").and_then(Value::as_str) {
                state.1.push_str(name);
            }
            if !state.2 && !state.0.is_empty() && !state.1.is_empty() {
                state.2 = true;
                sink.emit(StreamEvent::ToolCallStart {
                    id: state.0.clone(),
                    name: state.1.clone(),
                });
            }
            if let Some(part) = raw.pointer("/function/arguments").and_then(Value::as_str) {
                sink.emit(StreamEvent::ToolCallArgsDelta {
                    id: state.0.clone(),
                    delta: part.into(),
                });
            }
        }
        if value
            .pointer("/choices/0/finish_reason")
            .is_some_and(|value| !value.is_null())
        {
            for (id, _, started) in self.calls.values() {
                if *started {
                    sink.emit(StreamEvent::ToolCallEnd { id: id.clone() });
                }
            }
            sink.emit(StreamEvent::Done);
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
    let mut calls: BTreeMap<usize, ToolCall> = BTreeMap::new();
    let mut started = BTreeMap::new();
    let mut events = Vec::new();
    let mut usage = None;

    for value in values {
        if let Some(error) = value.get("error") {
            anyhow::bail!("provider error: {error}");
        }
        if let Some(raw) = value.get("usage") {
            usage = Some(normalize_usage(raw));
        }
        let Some(delta) = value.pointer("/choices/0/delta") else {
            continue;
        };
        if let Some(part) = delta.get("content").and_then(Value::as_str) {
            text.push_str(part);
            events.push(StreamEvent::TextDelta { delta: part.into() });
        }
        if let Some(part) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(Value::as_str)
        {
            reasoning.push_str(part);
            events.push(StreamEvent::ReasoningDelta { delta: part.into() });
        }
        for raw in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = raw.get("index").and_then(Value::as_u64).unwrap_or_default() as usize;
            let call = calls
                .entry(index)
                .or_insert_with(|| ToolCall::new("", "", ""));
            if let Some(id) = raw.get("id").and_then(Value::as_str) {
                call.id.push_str(id);
            }
            if let Some(name) = raw.pointer("/function/name").and_then(Value::as_str) {
                call.name.push_str(name);
            }
            if !started.get(&index).copied().unwrap_or(false)
                && !call.id.is_empty()
                && !call.name.is_empty()
            {
                events.push(StreamEvent::ToolCallStart {
                    id: call.id.clone(),
                    name: call.name.clone(),
                });
                started.insert(index, true);
            }
            if let Some(part) = raw.pointer("/function/arguments").and_then(Value::as_str) {
                call.arguments.push_str(part);
                events.push(StreamEvent::ToolCallArgsDelta {
                    id: call.id.clone(),
                    delta: part.into(),
                });
            }
        }
    }
    let tool_calls = calls.into_values().collect::<Vec<_>>();
    for call in &tool_calls {
        events.push(StreamEvent::ToolCallEnd {
            id: call.id.clone(),
        });
    }
    let mut blocks = Vec::new();
    if !reasoning.is_empty() {
        blocks.push(ContentBlock::Reasoning(reasoning));
    }
    if !text.is_empty() {
        blocks.push(ContentBlock::Text(text));
    }
    blocks.extend(tool_calls.iter().cloned().map(ContentBlock::ToolCall));
    if let Some(usage) = usage {
        events.push(StreamEvent::Usage(usage));
    }
    events.push(StreamEvent::Done);
    Ok((
        ModelTurn {
            blocks,
            tool_calls,
            usage,
            provider_state: None,
        },
        events,
    ))
}

fn normalize_usage(raw: &Value) -> Usage {
    let cached_tokens = raw
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .or_else(|| raw.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
        .or_else(|| raw.get("cached_tokens").and_then(Value::as_u64));
    let cache_write_tokens = raw
        .pointer("/prompt_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64);
    Usage {
        input_tokens: raw
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .map(|input| {
                input.saturating_sub(cached_tokens.unwrap_or(0) + cache_write_tokens.unwrap_or(0))
            }),
        output_tokens: raw.get("completion_tokens").and_then(Value::as_u64),
        cached_tokens,
        cache_write_tokens,
        total_tokens: raw.get("total_tokens").and_then(Value::as_u64),
    }
}
