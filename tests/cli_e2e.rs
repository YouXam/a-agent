use std::fs;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use a_agent::fish::install_to;
use a_agent::session::{SessionStore, TURN_INTERRUPTED_NOTICE};
use tempfile::tempdir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::process::Command;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn responses_config(base_url: &str) -> String {
    format!(
        "default_model = \"test\"\n\n[providers.test]\ntype = \"responses\"\nbase_url = \"{base_url}/v1\"\napi_key = \"secret\"\n\n[models.test]\nprovider = \"test\"\nmodel = \"test-model\"\neffort = \"low\"\nefforts = [\"low\", \"medium\", \"high\"]\n"
    )
}

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
        responses_config(&server.uri()),
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
        responses_config(&server.uri()),
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
async fn fish_slash_commands_persist_model_and_effort_without_model_requests() {
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"configured\"}\n\n",
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
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            r#"
default_model = "fast"

[providers.test]
type = "responses"
base_url = "{}/v1"
api_key = "secret"

[models.fast]
provider = "test"
model = "gpt-fast"
effort = "low"
efforts = ["low", "medium"]

[models.deep]
provider = "test"
model = "gpt-deep"
effort = "high"
efforts = ["high", "max"]
"#,
            server.uri()
        ),
    )
    .unwrap();

    for prompt in ["/model deep", "/effort max", "use current settings"] {
        let output = Command::new(env!("CARGO_BIN_EXE_a"))
            .args([
                "--fish-ai",
                "--fish-session-key",
                "slash-session",
                "-1",
                prompt,
            ])
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
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "gpt-deep");
    assert_eq!(body["reasoning"]["effort"], "max");
    let store = SessionStore::open(&state.join("a/sessions.db")).unwrap();
    let session = store
        .find_client_session(repo.to_str().unwrap(), "slash-session")
        .unwrap()
        .unwrap();
    assert_eq!(session.model_profile.as_deref(), Some("deep"));
    assert_eq!(session.effort.as_deref(), Some("max"));
}

#[tokio::test]
async fn fish_resume_command_rebinds_the_session_without_a_model_request() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    let state = temp.path().join("state");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
default_model = "test"
[providers.test]
type = "responses"
api_key = "secret"
[models.test]
provider = "test"
model = "test-model"
"#,
    )
    .unwrap();
    let target_id = {
        let mut store = SessionStore::open(&state.join("a/sessions.db")).unwrap();
        store
            .create_session(
                a_agent::session::NewSession::new(
                    repo.to_str().unwrap(),
                    "responses",
                    "test-model",
                )
                .with_client_session_key("fish-resume")
                .with_model_selection("test", None),
            )
            .unwrap();
        store
            .create_session(
                a_agent::session::NewSession::new(
                    repo.to_str().unwrap(),
                    "responses",
                    "test-model",
                )
                .with_model_selection("test", None),
            )
            .unwrap()
            .id
    };

    let output = Command::new(env!("CARGO_BIN_EXE_a"))
        .args([
            "--fish-ai",
            "--fish-session-key",
            "fish-resume",
            "-1",
            &format!("/resume {target_id}"),
        ])
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
    let store = SessionStore::open(&state.join("a/sessions.db")).unwrap();
    assert_eq!(
        store
            .find_client_session(repo.to_str().unwrap(), "fish-resume")
            .unwrap()
            .unwrap()
            .id,
        target_id
    );
}

#[tokio::test]
async fn compact_command_skips_the_api_for_an_empty_session() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    let state = temp.path().join("state");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
default_model = "test"
[providers.test]
type = "responses"
api_key = "secret"
[models.test]
provider = "test"
model = "test-model"
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_a"))
        .args(["-1", "/compact"])
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
    assert!(String::from_utf8_lossy(&output.stdout).contains("no conversation to compact"));
}

#[cfg(unix)]
#[tokio::test]
async fn fish_model_command_opens_an_arrow_key_selector() {
    if Command::new("tmux").arg("-V").output().await.is_err() {
        return;
    }
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    let state = temp.path().join("state");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
default_model = "fast"
[providers.test]
type = "responses"
api_key = "secret"
[models.deep]
provider = "test"
model = "gpt-deep"
[models.fast]
provider = "test"
model = "gpt-fast"
"#,
    )
    .unwrap();

    let socket = format!("a-model-select-{}", std::process::id());
    let session = "model-select";
    let shell = format!(
        "env HOME={} XDG_STATE_HOME={} bash --noprofile --norc",
        home.display(),
        state.display()
    );
    assert!(
        Command::new("tmux")
            .args([
                "-L",
                &socket,
                "new-session",
                "-d",
                "-x",
                "90",
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
            "{} --fish-ai --fish-session-key selector -1 /model",
            env!("CARGO_BIN_EXE_a")
        ),
    )
    .await;
    tmux_send_key(&socket, session, "Enter").await;
    wait_for_tmux_text(&socket, session, "Model").await;
    tmux_send_key(&socket, session, "Up").await;
    tmux_send_key(&socket, session, "Enter").await;
    wait_for_pane_command(&socket, session, "bash").await;

    let store = SessionStore::open(&state.join("a/sessions.db")).unwrap();
    let selected = store
        .find_client_session(repo.to_str().unwrap(), "selector")
        .unwrap()
        .unwrap();
    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
    assert_eq!(selected.model_profile.as_deref(), Some("deep"));
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
        responses_config(&server.uri()),
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
                .set_delay(std::time::Duration::from_millis(500))
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
        responses_config(&server.uri()),
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
    wait_for_agent_prompt(&socket, session).await;
    let initial_prompt = tmux_pane(&socket, session).await;
    assert!(initial_prompt.contains("multi · tab"), "{initial_prompt:?}");
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
    let loading = wait_for_tmux_text(&socket, session, "thinking").await;
    assert!(loading.contains("thinking"), "{loading:?}");
    let screen = wait_for_tmux_text(&socket, session, "Ready.").await;
    let history = tmux_history(&socket, session).await;
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
        screen.lines().any(|line| line.starts_with("a> hi")),
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
    assert!(!history.contains("thinking"), "{history:?}");
    assert!(
        !["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            .iter()
            .any(|spinner| history.contains(spinner)),
        "{history:?}"
    );

    tmux_send_key(&socket, session, "Escape").await;
    let single_escape = wait_for_tmux_text(&socket, session, "Rewind to:").await;
    assert!(single_escape.contains("rewind>"), "{single_escape:?}");
    tmux_send_key(&socket, session, "C-c").await;
    wait_for_agent_prompt(&socket, session).await;
    tmux_send_text(&socket, session, "retained").await;
    let after_single_escape = wait_for_tmux_text(&socket, session, "a> retained").await;
    assert!(
        after_single_escape.contains("a> retained"),
        "{after_single_escape:?}"
    );
    tmux_send_key(&socket, session, "C-u").await;

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
async fn interactive_generation_streams_reasoning_and_text_above_the_spinner() {
    if Command::new("tmux").arg("-V").output().await.is_err() {
        return;
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let count = stream.read(&mut buffer).await.unwrap();
            if count == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..count]);
        }
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        for data in [
            serde_json::json!({"type":"response.reasoning_summary_text.delta","delta":"streaming reason"}).to_string(),
            serde_json::json!({"type":"response.output_text.delta","delta":"first "}).to_string(),
            serde_json::json!({"type":"response.output_text.delta","delta":"second"}).to_string(),
            serde_json::json!({"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2}}}).to_string(),
        ] {
            let event = format!("data: {data}\n\n");
            stream
                .write_all(format!("{:x}\r\n", event.len()).as_bytes())
                .await
                .unwrap();
            stream.write_all(event.as_bytes()).await.unwrap();
            stream.write_all(b"\r\n").await.unwrap();
            stream.flush().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        }
        let done = "data: [DONE]\n\n";
        stream
            .write_all(format!("{:x}\r\n{done}\r\n0\r\n\r\n", done.len()).as_bytes())
            .await
            .unwrap();
    });

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        format!(
            "{}\n[ui]\nshow_reasoning = true\n",
            responses_config(&format!("http://{address}"))
        ),
    )
    .unwrap();

    let socket = format!("a-stream-test-{}", std::process::id());
    let session = "streaming";
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
                "-e",
                &format!("HOME={}", home.display()),
                "-e",
                &format!("XDG_STATE_HOME={}", temp.path().join("state").display()),
                env!("CARGO_BIN_EXE_a"),
            ])
            .status()
            .await
            .unwrap()
            .success()
    );
    wait_for_agent_prompt(&socket, session).await;
    tmux_send_text(&socket, session, "stream now").await;
    tmux_send_key(&socket, session, "Enter").await;

    let reasoning = wait_for_tmux_text(&socket, session, "streaming reason").await;
    assert!(reasoning.contains("▾ Reasoning"), "{reasoning:?}");
    assert!(reasoning.contains("thinking"), "{reasoning:?}");
    assert!(
        reasoning.find("streaming reason").unwrap() < reasoning.find("thinking").unwrap(),
        "{reasoning:?}"
    );

    let text = wait_for_tmux_text(&socket, session, "│ first ").await;
    assert!(text.contains("generating"), "{text:?}");
    assert!(
        text.find("│ first ").unwrap() < text.find("generating").unwrap(),
        "{text:?}"
    );

    let final_screen = wait_for_tmux_text(&socket, session, "│ first second").await;
    wait_for_agent_prompt(&socket, session).await;
    let history = tmux_history(&socket, session).await;
    assert_eq!(
        history.matches("streaming reason").count(),
        1,
        "{history:?}"
    );
    assert_eq!(history.matches("│ first second").count(), 1, "{history:?}");
    assert!(!history.contains("thinking"), "{history:?}");
    assert!(!history.contains("generating"), "{history:?}");
    assert!(final_screen.contains("│ first second"), "{final_screen:?}");

    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
    server.await.unwrap();
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
        responses_config(&server.uri()),
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
    wait_for_tmux_text(&socket, session, "once · tab").await;
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
            "{}\n\
[ui]
tool_live_output_lines = 2
tool_output_max_lines = 3
tool_output_max_bytes = 4096
",
            responses_config(&server.uri())
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
    assert!(!final_history.contains("● bash"), "{final_history:?}");
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
        responses_config(&server.uri()),
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

#[cfg(unix)]
#[tokio::test]
async fn standalone_input_history_is_shared_and_once_mode_exits_after_response() {
    if Command::new("tmux").arg("-V").output().await.is_err() {
        return;
    }
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"history response\"}\n\n",
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
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&repo).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        responses_config(&server.uri()),
    )
    .unwrap();

    let socket = format!("a-history-{}", std::process::id());
    let session = "history";
    let shell = format!(
        "env HOME={} XDG_STATE_HOME={} bash --noprofile --norc",
        home.display(),
        state.display()
    );
    assert!(
        Command::new("tmux")
            .args([
                "-L",
                &socket,
                "new-session",
                "-d",
                "-x",
                "90",
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

    for prompt in ["persisted history", "one shot"] {
        tmux_send_text(&socket, session, env!("CARGO_BIN_EXE_a")).await;
        tmux_send_key(&socket, session, "Enter").await;
        wait_for_agent_prompt(&socket, session).await;
        if prompt == "persisted history" {
            tmux_send_text(&socket, session, prompt).await;
        } else {
            tmux_send_key(&socket, session, "Up").await;
            let recalled = wait_for_tmux_text(&socket, session, "a> persisted history").await;
            assert!(recalled.contains("persisted history"), "{recalled:?}");
            tmux_send_key(&socket, session, "C-u").await;
            tmux_send_key(&socket, session, "Tab").await;
            wait_for_tmux_text(&socket, session, "once · tab").await;
            tmux_send_text(&socket, session, prompt).await;
        }
        tmux_send_key(&socket, session, "Enter").await;
        wait_for_tmux_text(&socket, session, "history response").await;
        if prompt == "persisted history" {
            tmux_send_key(&socket, session, "C-c").await;
            wait_for_pane_command(&socket, session, "bash").await;
        } else {
            wait_for_pane_command(&socket, session, "bash").await;
        }
    }

    let requests = server.received_requests().await.unwrap();
    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
    assert_eq!(requests.len(), 2);
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

async fn wait_for_pane_command(socket: &str, session: &str, command: &str) {
    for _ in 0..120 {
        let output = Command::new("tmux")
            .args([
                "-L",
                socket,
                "display-message",
                "-p",
                "-t",
                session,
                "#{pane_current_command}",
            ])
            .output()
            .await
            .unwrap();
        if String::from_utf8_lossy(&output.stdout).trim() == command {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("pane did not return to {command}");
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

async fn wait_for_agent_prompt(socket: &str, session: &str) {
    for _ in 0..120 {
        let pane = tmux_pane(socket, session).await;
        if pane
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .is_some_and(|line| {
                let line = line.trim();
                line.starts_with("a>") && line.contains("multi · tab")
            })
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!(
        "agent prompt did not appear: {:?}",
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
