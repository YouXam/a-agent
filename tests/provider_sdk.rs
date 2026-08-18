use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use a_agent::config::{ProviderConfig, ProviderKind};
use a_agent::model::{ContentBlock, ModelMessage, ModelRequest, Role, StreamEvent};
use a_agent::provider::anthropic::AnthropicProvider;
use a_agent::provider::chat_completion::ChatCompletionProvider;
use a_agent::provider::responses::ResponsesProvider;
use a_agent::provider::{EventSink, Provider, create_provider};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config(kind: ProviderKind, base_url: String) -> ProviderConfig {
    ProviderConfig {
        kind,
        base_url: Some(base_url),
        model: "test-model".into(),
        api_key_env: "TEST_KEY".into(),
        api_key: None,
        headers: BTreeMap::from([("X-Tenant".into(), "acme".into())]),
        max_tokens: 1024,
        request: BTreeMap::new(),
    }
}

#[tokio::test]
async fn configured_api_key_takes_precedence_without_reading_the_environment() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer config-secret"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
                    "data: [DONE]\n\n"
                )),
        )
        .mount(&server)
        .await;
    let mut provider_config = config(ProviderKind::Responses, format!("{}/v1", server.uri()));
    provider_config.api_key = Some("config-secret".into());
    provider_config.api_key_env = "A_AGENT_INTENTIONALLY_MISSING_KEY".into();
    let provider = create_provider(provider_config).unwrap();
    let turn = provider
        .stream_turn(request(), EventSink::default(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(turn.final_text().as_deref(), Some("ok"));
}

fn request() -> ModelRequest {
    ModelRequest {
        system_prompt: "system rules".into(),
        messages: vec![ModelMessage {
            role: Role::User,
            blocks: vec![ContentBlock::Text("hello".into())],
        }],
        include_tools: true,
    }
}

fn sink() -> (EventSink, Arc<Mutex<Vec<StreamEvent>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let output = seen.clone();
    (
        EventSink::new(move |event| output.lock().unwrap().push(event)),
        seen,
    )
}

#[tokio::test]
async fn responses_sdk_honors_custom_base_headers_and_streams() {
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n",
        "data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer secret"))
        .and(header("x-tenant", "acme"))
        .and(body_partial_json(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "instructions": "system rules",
            "max_output_tokens": 1024
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = ResponsesProvider::new(
        config(ProviderKind::Responses, format!("{}/v1/", server.uri())),
        "secret".into(),
    )
    .unwrap();
    let (events, seen) = sink();
    let turn = provider
        .stream_turn(request(), events, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(turn.final_text().as_deref(), Some("hello"));
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { delta } if delta == "hello"))
    );

    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn chat_completion_sdk_honors_custom_base_and_normalizes_stream() {
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("x-tenant", "acme"))
        .and(body_partial_json(
            serde_json::json!({"model":"test-model","stream":true,"max_tokens":1024}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = ChatCompletionProvider::new(
        config(ProviderKind::Chatcompletion, format!("{}/v1", server.uri())),
        "secret".into(),
    )
    .unwrap();
    let turn = provider
        .stream_turn(request(), EventSink::default(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(turn.final_text().as_deref(), Some("hi"));
}

#[tokio::test]
async fn anthropic_sdk_honors_custom_origin_headers_and_streams() {
    let server = MockServer::start().await;
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"test-model\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\",\"citations\":null}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hey\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-secret"))
        .and(header("x-tenant", "acme"))
        .and(body_partial_json(serde_json::json!({
            "model":"test-model",
            "stream":true,
            "system":"system rules",
            "max_tokens":1024
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(
        config(ProviderKind::Anthropic, server.uri()),
        "sk-ant-secret".into(),
    )
    .unwrap();
    let turn = provider
        .stream_turn(request(), EventSink::default(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(turn.final_text().as_deref(), Some("hey"));

    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn provider_cancellation_interrupts_waiting_for_response_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(600))
                .insert_header("content-type", "text/event-stream")
                .set_body_string("data: [DONE]\n\n"),
        )
        .mount(&server)
        .await;
    let provider = ResponsesProvider::new(
        config(ProviderKind::Responses, format!("{}/v1", server.uri())),
        "secret".into(),
    )
    .unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(25)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    let error = provider
        .stream_turn(request(), EventSink::default(), cancel)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    assert!(started.elapsed() < Duration::from_millis(250));
}
