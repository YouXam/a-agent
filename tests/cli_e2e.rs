use std::fs;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use a_agent::fish::install_to;
use a_agent::session::TURN_INTERRUPTED_NOTICE;
use tempfile::tempdir;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn one_turn_cli_persists_and_resumes_without_eager_file_content() {
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello from model\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":3}}}\n\n",
        "data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    let state = temp.path().join("state");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(repo.join("src")).unwrap();
    fs::write(repo.join("src/target.rs"), "EAGER_CONTENT_MUST_NOT_APPEAR").unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            r#"
[provider]
type = "responses"
base_url = "{}/v1"
model = "test-model"
api_key_env = "TEST_API_KEY"
"#,
            server.uri()
        ),
    )
    .unwrap();

    let binary = env!("CARGO_BIN_EXE_a");
    let mut child = Command::new(binary)
        .args(["-1", "src/target.rs", "answer briefly"])
        .current_dir(&repo)
        .env("HOME", &home)
        .env("XDG_STATE_HOME", &state)
        .env("TEST_API_KEY", "secret")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"compiler failed at useful tail")
        .await
        .unwrap();
    let first = child.wait_with_output().await.unwrap();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("hello from model"));
    assert!(state.join("a/sessions.db").is_file());

    let second = Command::new(binary)
        .args(["-r", "-1", "continue"])
        .current_dir(&repo)
        .env("HOME", &home)
        .env("XDG_STATE_HOME", &state)
        .env("TEST_API_KEY", "secret")
        .output()
        .await
        .unwrap();
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let first_body = String::from_utf8(requests[0].body.clone()).unwrap();
    assert!(first_body.contains("src/target.rs"));
    assert!(first_body.contains("compiler failed at useful tail"));
    assert!(!first_body.contains("EAGER_CONTENT_MUST_NOT_APPEAR"));
    let second_body = String::from_utf8(requests[1].body.clone()).unwrap();
    assert!(second_body.contains("hello from model"));
    assert!(second_body.contains("continue"));
}

#[tokio::test]
async fn fish_session_keys_isolate_conversations_in_the_same_cwd() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicUsize::new(0));
    let response_count = count.clone();
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_: &wiremock::Request| {
            let text = match response_count.fetch_add(1, Ordering::SeqCst) {
                0 => "reply-from-one",
                1 => "reply-from-two",
                _ => "reply-from-one-again",
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {{\"type\":\"response.output_text.delta\",\"delta\":{}}}\n\ndata: {{\"type\":\"response.completed\",\"response\":{{\"usage\":{{}}}}}}\n\ndata: [DONE]\n\n",
                    serde_json::to_string(text).unwrap()
                ))
        })
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    let state = temp.path().join("state");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            "[provider]\ntype = \"responses\"\nbase_url = \"{}/v1\"\nmodel = \"test-model\"\napi_key = \"secret\"\n",
            server.uri()
        ),
    )
    .unwrap();

    for (key, prompt) in [
        ("fish-one", "first prompt"),
        ("fish-two", "second prompt"),
        ("fish-one", "third prompt"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_a"))
            .args(["--fish-ai", "--fish-session-key", key, "-1", prompt])
            .current_dir(&repo)
            .env("HOME", &home)
            .env("XDG_STATE_HOME", &state)
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3);
    let second = String::from_utf8(requests[1].body.clone()).unwrap();
    let third = String::from_utf8(requests[2].body.clone()).unwrap();
    assert!(!second.contains("reply-from-one"), "{second}");
    assert!(third.contains("reply-from-one"), "{third}");
    assert!(!third.contains("reply-from-two"), "{third}");
}

#[tokio::test]
async fn one_turn_cli_completes_a_model_tool_model_cycle() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicUsize::new(0));
    let response_count = count.clone();
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_: &wiremock::Request| {
            let body = if response_count.fetch_add(1, Ordering::SeqCst) == 0 {
                concat!(
                    "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}\n\n",
                    "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n",
                    "data: [DONE]\n\n"
                )
            } else {
                concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"tool cycle complete\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":3}}}\n\n",
                    "data: [DONE]\n\n"
                )
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body)
        })
        .mount(&server).await;

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(repo.join("a.rs"), "code\n").unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            r#"
[provider]
type = "responses"
base_url = "{}/v1"
model = "test-model"
api_key_env = "TEST_API_KEY"
"#,
            server.uri()
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_a"))
        .args(["-1", "inspect a.rs"])
        .current_dir(&repo)
        .env("HOME", &home)
        .env("XDG_STATE_HOME", temp.path().join("state"))
        .env("TEST_API_KEY", "secret")
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("tool cycle complete"));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let second = String::from_utf8(requests[1].body.clone()).unwrap();
    assert!(second.contains("function_call_output"));
    assert!(second.contains("1: code"));
}

#[tokio::test]
async fn first_agent_run_reports_the_generated_config_path() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    fs::create_dir_all(&repo).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_a"))
        .args(["-1", "hello"])
        .current_dir(&repo)
        .env("HOME", &home)
        .env("XDG_STATE_HOME", temp.path().join("state"))
        .env_remove("OPENAI_API_KEY")
        .output()
        .await
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Created config"));
    assert!(stderr.contains(".config/a/config.toml"));
    assert!(home.join(".config/a/config.toml").is_file());
}

#[cfg(unix)]
#[tokio::test]
async fn interactive_tui_is_inline_aligned_and_does_not_duplicate_input() {
    if Command::new("tmux")
        .arg("--version")
        .output()
        .await
        .is_err()
    {
        return;
    }
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"checking\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Ready.\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        "data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            r#"
[provider]
type = "responses"
base_url = "{}/v1"
model = "test-model"
api_key = "secret"
"#,
            server.uri()
        ),
    )
    .unwrap();

    let socket = format!("a-agent-test-{}", std::process::id());
    let session = "a-agent-tui";
    let started = Command::new("tmux")
        .args([
            "-L",
            &socket,
            "new-session",
            "-d",
            "-x",
            "80",
            "-y",
            "24",
            "-s",
            session,
            "-c",
            repo.to_str().unwrap(),
            "-e",
            &format!("HOME={}", home.display()),
            "-e",
            &format!("XDG_STATE_HOME={}", temp.path().join("state").display()),
            env!("CARGO_BIN_EXE_a"),
        ])
        .output()
        .await
        .unwrap();
    assert!(
        started.status.success(),
        "{}",
        String::from_utf8_lossy(&started.stderr)
    );
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(
        Command::new("tmux")
            .args(["-L", &socket, "send-keys", "-t", session, "-l", "hi"])
            .status()
            .await
            .unwrap()
            .success()
    );
    assert!(
        Command::new("tmux")
            .args(["-L", &socket, "send-keys", "-t", session, "Enter"])
            .status()
            .await
            .unwrap()
            .success()
    );
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let capture = Command::new("tmux")
        .args(["-L", &socket, "capture-pane", "-p", "-t", session])
        .output()
        .await
        .unwrap();
    let screen = String::from_utf8(capture.stdout).unwrap();
    let alternate = Command::new("tmux")
        .args([
            "-L",
            &socket,
            "display-message",
            "-p",
            "-t",
            session,
            "#{alternate_on}",
        ])
        .output()
        .await
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&alternate.stdout).trim(), "0");
    assert_eq!(screen.matches("hi").count(), 1, "{screen:?}");
    assert!(
        screen
            .lines()
            .any(|line| line.starts_with("› hi") || line.starts_with("> hi")),
        "{screen:?}"
    );
    assert!(
        screen.lines().any(|line| line.starts_with("▸ Reasoning")),
        "{screen:?}"
    );
    assert!(
        screen.lines().any(|line| line.starts_with("│ Ready.")),
        "{screen:?}"
    );

    assert!(
        Command::new("tmux")
            .args([
                "-L",
                &socket,
                "send-keys",
                "-t",
                session,
                "Escape",
                "Escape"
            ])
            .status()
            .await
            .unwrap()
            .success()
    );
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let rewind = Command::new("tmux")
        .args(["-L", &socket, "capture-pane", "-p", "-t", session])
        .output()
        .await
        .unwrap();
    let rewind = String::from_utf8(rewind.stdout).unwrap();
    assert!(rewind.contains("Rewind to:"), "{rewind:?}");
    assert!(rewind.contains("rewind>"), "{rewind:?}");

    let _ = Command::new("tmux")
        .args(["-L", &socket, "send-keys", "-t", session, "C-c"])
        .status()
        .await;
    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
}

#[cfg(unix)]
#[tokio::test]
async fn fresh_fish_records_shell_history_that_reaches_the_agent_request() {
    if Command::new("tmux").arg("-V").output().await.is_err() {
        return;
    }
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"history received\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        "data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    let state = temp.path().join("state");
    let bin = temp.path().join("bin");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::create_dir_all(&bin).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            r#"
[provider]
type = "responses"
base_url = "{}/v1"
model = "test-model"
api_key = "secret"
"#,
            server.uri()
        ),
    )
    .unwrap();
    install_to(&home).unwrap();
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_a"), bin.join("a")).unwrap();

    let socket = format!("a-fish-history-{}", std::process::id());
    let session = "fish-history";
    let fish_command = format!(
        "env HOME={} XDG_CONFIG_HOME={}/.config XDG_STATE_HOME={} PATH={}:{} fish",
        home.display(),
        home.display(),
        state.display(),
        bin.display(),
        std::env::var("PATH").unwrap()
    );
    let started = Command::new("tmux")
        .args([
            "-L",
            &socket,
            "new-session",
            "-d",
            "-x",
            "100",
            "-y",
            "30",
            "-s",
            session,
            "-c",
            repo.to_str().unwrap(),
            &fish_command,
        ])
        .output()
        .await
        .unwrap();
    assert!(
        started.status.success(),
        "{}",
        String::from_utf8_lossy(&started.stderr)
    );
    wait_for_fish_prompt(&socket, session).await;
    tmux_send_text(&socket, session, "false").await;
    tmux_send_key(&socket, session, "Enter").await;
    wait_for_fish_prompt(&socket, session).await;
    tmux_send_hex(&socket, session, "07").await;
    wait_for_tmux_text(&socket, session, "[AI]").await;
    tmux_send_text(&socket, session, "fix previous failure").await;
    tmux_send_key(&socket, session, "Enter").await;
    let pane = wait_for_tmux_text(&socket, session, "history received").await;

    let requests = server.received_requests().await.unwrap();
    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
    assert_eq!(requests.len(), 1, "{pane:?}");
    assert_eq!(pane.matches("fix previous failure").count(), 1, "{pane:?}");
    assert!(!pane.contains("a --fish-ai"), "{pane:?}");
    let body = String::from_utf8(requests[0].body.clone()).unwrap();
    let body: serde_json::Value = serde_json::from_str(&body).unwrap();
    let instructions = body["instructions"].as_str().unwrap();
    assert!(
        instructions.contains("Runtime shell context"),
        "{instructions}"
    );
    assert!(
        instructions.contains("command: \"false\""),
        "{instructions}"
    );
    assert!(instructions.contains("exit_code: 1"), "{instructions}");
    assert_eq!(body["input"][0]["content"], "fix previous failure");
}

#[cfg(unix)]
#[tokio::test]
async fn live_bash_output_auto_scrolls_and_commits_a_bounded_final_block() {
    if Command::new("tmux").arg("-V").output().await.is_err() {
        return;
    }
    let server = MockServer::start().await;
    let count = Arc::new(AtomicUsize::new(0));
    let response_count = count.clone();
    let command = "for i in 1 2 3 4 5 6 7 8; do echo live-$i; sleep 0.15; done";
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_: &wiremock::Request| {
            let body = if response_count.fetch_add(1, Ordering::SeqCst) == 0 {
                format!(
                    "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    serde_json::json!({
                        "type":"response.output_item.added",
                        "item":{"type":"function_call","id":"item_live","call_id":"call_live","name":"bash","arguments":serde_json::json!({"command":command}).to_string()}
                    }),
                    serde_json::json!({
                        "type":"response.output_item.done",
                        "item":{"type":"function_call","id":"item_live","call_id":"call_live","name":"bash","arguments":serde_json::json!({"command":command}).to_string()}
                    }),
                    serde_json::json!({"type":"response.completed","response":{"usage":{}}}),
                )
            } else {
                concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"stream complete\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
                    "data: [DONE]\n\n"
                )
                .to_owned()
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body)
        })
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            r#"
[provider]
type = "responses"
base_url = "{}/v1"
model = "test-model"
api_key = "secret"

[ui]
tool_live_output_lines = 2
tool_output_max_lines = 3
tool_output_max_bytes = 4096
"#,
            server.uri()
        ),
    )
    .unwrap();

    let socket = format!("a-live-output-{}", std::process::id());
    let session = "live-output";
    let shell = format!(
        "env HOME={} XDG_STATE_HOME={} bash --noprofile --norc",
        home.display(),
        temp.path().join("state").display()
    );
    let started = Command::new("tmux")
        .args([
            "-L",
            &socket,
            "new-session",
            "-d",
            "-x",
            "60",
            "-y",
            "24",
            "-s",
            session,
            "-c",
            repo.to_str().unwrap(),
            &shell,
        ])
        .output()
        .await
        .unwrap();
    assert!(started.status.success());
    tmux_send_text(
        &socket,
        session,
        &format!("{} -1 run", env!("CARGO_BIN_EXE_a")),
    )
    .await;
    tmux_send_key(&socket, session, "Enter").await;
    wait_for_tmux_text(&socket, session, "live-4").await;
    let live = tmux_pane(&socket, session).await;
    assert!(live.contains("live-4"), "{live:?}");
    assert!(!live.contains("live-1"), "{live:?}");
    let live_spinner_lines = live
        .lines()
        .filter(|line| {
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                .iter()
                .any(|spinner| line.starts_with(spinner))
        })
        .count();
    assert_eq!(live_spinner_lines, 1, "duplicate live frames: {live:?}");

    let final_pane = wait_for_tmux_text(&socket, session, "stream complete").await;
    let final_history = tmux_history(&socket, session).await;
    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
    assert!(final_pane.contains("live-8"), "{final_pane:?}");
    assert!(!final_pane.contains("live-1"), "{final_pane:?}");
    assert!(final_pane.contains("✓ bash  exit 0"), "{final_pane:?}");
    assert!(
        !["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            .iter()
            .any(|spinner| final_history.contains(spinner)),
        "transient spinner leaked into scrollback: {final_history:?}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn escape_cancels_a_running_bash_tool_and_returns_to_the_shell() {
    if Command::new("tmux").arg("-V").output().await.is_err() {
        return;
    }
    let server = MockServer::start().await;
    let command = "echo escape-started; sleep 10; echo escape-finished";
    let tool_body = format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({
            "type":"response.output_item.added",
            "item":{"type":"function_call","id":"item_escape","call_id":"call_escape","name":"bash","arguments":serde_json::json!({"command":command}).to_string()}
        }),
        serde_json::json!({
            "type":"response.output_item.done",
            "item":{"type":"function_call","id":"item_escape","call_id":"call_escape","name":"bash","arguments":serde_json::json!({"command":command}).to_string()}
        }),
        serde_json::json!({"type":"response.completed","response":{"usage":{}}}),
    );
    let count = Arc::new(AtomicUsize::new(0));
    let response_count = count.clone();
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_: &wiremock::Request| {
            let body = if response_count.fetch_add(1, Ordering::SeqCst) == 0 {
                tool_body.clone()
            } else {
                concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"fresh after cancel\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
                    "data: [DONE]\n\n"
                )
                .to_owned()
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body)
        })
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            "[provider]\ntype = \"responses\"\nbase_url = \"{}/v1\"\nmodel = \"test-model\"\napi_key = \"secret\"\n",
            server.uri()
        ),
    )
    .unwrap();

    let socket = format!("a-escape-{}", std::process::id());
    let session = "escape";
    let shell = format!(
        "env HOME={} XDG_STATE_HOME={} bash --noprofile --norc",
        home.display(),
        temp.path().join("state").display()
    );
    assert!(
        Command::new("tmux")
            .args([
                "-L",
                &socket,
                "new-session",
                "-d",
                "-x",
                "80",
                "-y",
                "24",
                "-s",
                session,
                "-c",
                repo.to_str().unwrap(),
                &shell,
            ])
            .status()
            .await
            .unwrap()
            .success()
    );
    tmux_send_text(
        &socket,
        session,
        &format!(
            "{} --fish-session-key escape-key -1 cancel",
            env!("CARGO_BIN_EXE_a")
        ),
    )
    .await;
    tmux_send_key(&socket, session, "Enter").await;
    wait_for_tmux_text(&socket, session, "escape-started").await;
    tmux_send_hex(&socket, session, "1b").await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let current = Command::new("tmux")
        .args([
            "-L",
            &socket,
            "display-message",
            "-p",
            "-t",
            session,
            "#{pane_current_command}",
        ])
        .output()
        .await
        .unwrap();
    let pane = tmux_history(&socket, session).await;
    assert_eq!(
        String::from_utf8_lossy(&current.stdout).trim(),
        "bash",
        "{pane:?}"
    );
    assert!(pane.contains("× bash  cancelled"), "{pane:?}");
    assert_eq!(pane.matches("escape-finished").count(), 1, "{pane:?}");

    tmux_send_text(
        &socket,
        session,
        &format!(
            "{} --fish-session-key escape-key -1 next",
            env!("CARGO_BIN_EXE_a")
        ),
    )
    .await;
    tmux_send_key(&socket, session, "Enter").await;
    let resumed = wait_for_tmux_text(&socket, session, "fresh after cancel").await;
    let requests = server.received_requests().await.unwrap();
    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
    assert_eq!(requests.len(), 2, "{resumed:?}");
    let resumed_body = String::from_utf8(requests[1].body.clone()).unwrap();
    assert!(
        resumed_body.contains(TURN_INTERRUPTED_NOTICE),
        "{resumed_body}"
    );
    assert!(resumed_body.contains("next"), "{resumed_body}");
}

async fn tmux_send_text(socket: &str, session: &str, text: &str) {
    assert!(
        Command::new("tmux")
            .args(["-L", socket, "send-keys", "-t", session, "-l", text])
            .status()
            .await
            .unwrap()
            .success()
    );
}

async fn tmux_send_key(socket: &str, session: &str, key: &str) {
    assert!(
        Command::new("tmux")
            .args(["-L", socket, "send-keys", "-t", session, key])
            .status()
            .await
            .unwrap()
            .success()
    );
}

async fn tmux_send_hex(socket: &str, session: &str, hex: &str) {
    assert!(
        Command::new("tmux")
            .args(["-L", socket, "send-keys", "-t", session, "-H", hex])
            .status()
            .await
            .unwrap()
            .success()
    );
}

async fn tmux_pane(socket: &str, session: &str) -> String {
    let output = Command::new("tmux")
        .args(["-L", socket, "capture-pane", "-p", "-t", session])
        .output()
        .await
        .unwrap();
    String::from_utf8(output.stdout).unwrap()
}

async fn tmux_history(socket: &str, session: &str) -> String {
    let output = Command::new("tmux")
        .args(["-L", socket, "capture-pane", "-p", "-S", "-", "-t", session])
        .output()
        .await
        .unwrap();
    String::from_utf8(output.stdout).unwrap()
}

async fn wait_for_fish_prompt(socket: &str, session: &str) {
    for _ in 0..80 {
        let pane = tmux_pane(socket, session).await;
        if pane
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .is_some_and(|line| {
                let line = line.trim_end();
                ['#', '>', '$', '❯']
                    .iter()
                    .any(|suffix| line.ends_with(*suffix))
            })
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!(
        "Fish prompt did not appear: {:?}",
        tmux_pane(socket, session).await
    );
}

async fn wait_for_tmux_text(socket: &str, session: &str, text: &str) -> String {
    for _ in 0..120 {
        let pane = tmux_pane(socket, session).await;
        if pane.contains(text) {
            return pane;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!(
        "tmux pane did not contain {text:?}: {:?}",
        tmux_pane(socket, session).await
    );
}
