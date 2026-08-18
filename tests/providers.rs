use a_agent::model::{ContentBlock, StreamEvent};
use a_agent::provider::{anthropic, chat_completion, responses};
use serde_json::json;

#[test]
fn responses_normalizes_text_reasoning_tools_and_usage() {
    let values = vec![
        json!({"type":"response.output_text.delta","delta":"done"}),
        json!({"type":"response.reasoning_summary_text.delta","delta":"checking"}),
        json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item_1","call_id":"call_1","name":"read","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","item_id":"item_1","delta":"{\"path\":"}),
        json!({"type":"response.function_call_arguments.delta","item_id":"item_1","delta":"\"a.rs\"}"}),
        json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item_1","call_id":"call_1","name":"read","arguments":"{\"path\":\"a.rs\"}"}}),
        json!({"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":3}}}}),
    ];
    let (turn, events) = responses::normalize_events(values).unwrap();
    assert_eq!(turn.tool_calls[0].arguments, "{\"path\":\"a.rs\"}");
    assert!(
        turn.blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::Text(text) if text == "done"))
    );
    assert_eq!(turn.usage.unwrap().cached_tokens, Some(3));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::ReasoningDelta { .. }))
    );
}

#[test]
fn anthropic_normalizes_fragmented_tool_json() {
    let values = vec![
        json!({"type":"message_start","message":{"usage":{"input_tokens":8,"cache_read_input_tokens":2}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool_1","name":"bash","input":{}}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"pwd\"}"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","usage":{"output_tokens":5}}),
        json!({"type":"message_stop"}),
    ];
    let (turn, _) = anthropic::normalize_events(values).unwrap();
    assert_eq!(turn.tool_calls[0].arguments, "{\"command\":\"pwd\"}");
    assert_eq!(turn.usage.unwrap().input_tokens, Some(8));
}

#[test]
fn chat_completion_normalizes_indexed_tool_deltas_and_reasoning() {
    let values = vec![
        json!({"choices":[{"delta":{"reasoning_content":"think","tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"pa"}}]}}]}),
        json!({"choices":[{"delta":{"content":"ok","tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":6,"prompt_tokens_details":{"cached_tokens":1}}}),
    ];
    let (turn, events) = chat_completion::normalize_events(values).unwrap();
    assert_eq!(turn.tool_calls[0].arguments, "{\"path\":\"x\"}");
    assert!(
        turn.blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::Text(text) if text == "ok"))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::ReasoningDelta { .. }))
    );
}

#[test]
fn provider_errors_are_actionable() {
    let error = responses::normalize_events(vec![
        json!({"type":"error","message":"bad request","code":"invalid"}),
    ])
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("bad request"));
    assert!(message.contains("invalid"));
}
