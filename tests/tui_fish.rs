use std::io::{self, Write};
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};

use a_agent::fish::{fish_script, install_to};
use a_agent::model::{StreamEvent, ToolResult};
use a_agent::tui::{InlineRenderer, InputEditor, RenderLimits};
use crossterm::event::{KeyCode, KeyModifiers};
use tokio::process::Command;

#[derive(Clone, Default)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl SharedWriter {
    fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap().clone()
    }
}

#[test]
fn transcript_is_append_only_aligned_and_colored() {
    let writer = SharedWriter::default();
    let renderer = InlineRenderer::new(writer.clone(), false, true);
    renderer.render_user("hi").unwrap();
    let sink = renderer.event_sink();
    sink.emit(StreamEvent::ReasoningDelta {
        delta: "inspect first".into(),
    });
    sink.emit(StreamEvent::TextDelta {
        delta: "Hello from the model".into(),
    });
    sink.emit(StreamEvent::ToolCallStart {
        id: "c1".into(),
        name: "read".into(),
    });
    sink.emit(StreamEvent::ToolCallArgsDelta {
        id: "c1".into(),
        delta: r#"{"path":"src/main.rs","offset":10}"#.into(),
    });
    sink.emit(StreamEvent::ToolCallEnd { id: "c1".into() });
    sink.emit(StreamEvent::ToolExecutionStart { id: "c1".into() });
    sink.emit(StreamEvent::ToolExecutionEnd {
        id: "c1".into(),
        result: ToolResult::success("c1", "1: code"),
    });
    sink.emit(StreamEvent::Done);

    let bytes = writer.bytes();
    let plain = String::from_utf8(strip_ansi_escapes::strip(&bytes)).unwrap();
    assert_eq!(plain.matches("hi").count(), 1);
    assert!(plain.lines().any(|line| line.starts_with("› hi")));
    assert!(plain.lines().any(|line| line.starts_with("▸ Reasoning")));
    assert!(
        plain
            .lines()
            .any(|line| line.starts_with("│ Hello from the model"))
    );
    assert!(
        plain
            .lines()
            .any(|line| line.starts_with("● read  src/main.rs  from line 11")),
        "{plain}"
    );
    assert!(!plain.lines().any(|line| line.trim() == "input"));
    assert!(!plain.lines().any(|line| line.trim() == "output"));
    assert!(plain.contains("1: code"));
    assert!(plain.lines().any(|line| line.starts_with("✓ read")));
    assert!(bytes.windows(2).any(|bytes| bytes == b"\x1b["));
    for forbidden in [b"\x1b[?1049".as_slice(), b"\x1b[2J", b"\x1b[H", b"\x1b[2K"] {
        assert!(
            !bytes
                .windows(forbidden.len())
                .any(|window| window == forbidden)
        );
    }
}

#[test]
fn apply_patch_renders_file_operations_and_diff_summary() {
    let writer = SharedWriter::default();
    let renderer = InlineRenderer::new(writer.clone(), false, false);
    let sink = renderer.event_sink();
    sink.emit(StreamEvent::ToolCallStart {
        id: "patch-1".into(),
        name: "apply_patch".into(),
    });
    sink.emit(StreamEvent::ToolCallArgsDelta {
        id: "patch-1".into(),
        delta: serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: src/parser.rs\n@@\n-old\n+new\n*** Add File: src/parser_test.rs\n+test\n*** End Patch"
        })
        .to_string(),
    });
    sink.emit(StreamEvent::ToolCallEnd {
        id: "patch-1".into(),
    });
    sink.emit(StreamEvent::ToolExecutionEnd {
        id: "patch-1".into(),
        result: ToolResult::success(
            "patch-1",
            "src/parser.rs (+1 -1)\nsrc/parser_test.rs (+1 -0)",
        ),
    });
    let plain = String::from_utf8(writer.bytes()).unwrap();
    assert!(plain.lines().any(|line| line == "● apply_patch"), "{plain}");
    assert!(
        plain.lines().any(|line| line == "  M src/parser.rs"),
        "{plain}"
    );
    assert!(
        plain.lines().any(|line| line == "  A src/parser_test.rs"),
        "{plain}"
    );
    assert!(!plain.lines().any(|line| line.trim() == "input"));
    assert!(!plain.lines().any(|line| line.trim() == "output"));
    assert!(plain.contains("✓ apply_patch  2 files  +2 -1"), "{plain}");
}

#[test]
fn failed_apply_patch_reports_failure_instead_of_file_count() {
    let writer = SharedWriter::default();
    let renderer = InlineRenderer::new(writer.clone(), false, false);
    let sink = renderer.event_sink();
    sink.emit(StreamEvent::ToolCallStart {
        id: "patch-fail".into(),
        name: "apply_patch".into(),
    });
    sink.emit(StreamEvent::ToolCallArgsDelta {
        id: "patch-fail".into(),
        delta: serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: /tmp/outside.rs\n@@\n-old\n+new\n*** End Patch"
        })
        .to_string(),
    });
    sink.emit(StreamEvent::ToolCallEnd {
        id: "patch-fail".into(),
    });
    sink.emit(StreamEvent::ToolExecutionEnd {
        id: "patch-fail".into(),
        result: ToolResult::error("patch-fail", "patch path is outside the workspace"),
    });
    let plain = String::from_utf8(writer.bytes()).unwrap();
    assert!(plain.contains("× apply_patch  failed"), "{plain}");
    assert!(!plain.contains("× apply_patch  1 files"), "{plain}");
}

#[test]
fn tool_input_and_output_are_bounded_with_visible_truncation() {
    let writer = SharedWriter::default();
    let renderer = InlineRenderer::new_with_limits(
        writer.clone(),
        false,
        false,
        RenderLimits {
            tool_input_max_bytes: 40,
            tool_output_max_bytes: 48,
            tool_output_max_lines: 2,
            tool_live_output_lines: 2,
        },
    );
    let sink = renderer.event_sink();
    sink.emit(StreamEvent::ToolCallStart {
        id: "bash-1".into(),
        name: "bash".into(),
    });
    sink.emit(StreamEvent::ToolCallArgsDelta {
        id: "bash-1".into(),
        delta: serde_json::json!({"command":"printf a; printf b; cargo test --workspace --all-targets"}).to_string(),
    });
    sink.emit(StreamEvent::ToolCallEnd {
        id: "bash-1".into(),
    });
    sink.emit(StreamEvent::ToolExecutionOutput {
        id: "bash-1".into(),
        delta: (0..10).map(|index| format!("line {index}\n")).collect(),
    });
    sink.emit(StreamEvent::ToolExecutionEnd {
        id: "bash-1".into(),
        result: ToolResult::error("bash-1", "ignored\n[exit code: 1]"),
    });
    let plain = String::from_utf8(writer.bytes()).unwrap();
    let lines = plain.lines().collect::<Vec<_>>();
    let status_index = lines
        .iter()
        .position(|line| *line == "× bash  exit 1")
        .expect("completed Bash status missing");
    let command_index = lines
        .iter()
        .position(|line| line.starts_with("  $ "))
        .expect("Bash command missing");
    assert!(status_index < command_index, "{plain}");
    assert!(!plain.contains("● bash"), "{plain}");
    assert_eq!(plain.matches("× bash  exit 1").count(), 1, "{plain}");
    assert!(
        plain.lines().any(|line| line.starts_with("  $ ")),
        "{plain}"
    );
    assert!(!plain.lines().any(|line| line.trim() == "input"), "{plain}");
    assert!(
        !plain.lines().any(|line| line.trim() == "output"),
        "{plain}"
    );
    assert!(plain.contains("command truncated"), "{plain}");
    assert!(plain.contains("output truncated"), "{plain}");
    assert!(plain.contains("line 9"), "{plain}");
    assert!(!plain.contains("line 0"), "{plain}");
    assert!(plain.contains("exit 1"), "{plain}");
}

#[test]
fn bash_without_output_renders_one_italic_placeholder_without_metadata() {
    let writer = SharedWriter::default();
    let renderer = InlineRenderer::new(writer.clone(), false, true);
    let sink = renderer.event_sink();
    sink.emit(StreamEvent::ToolCallStart {
        id: "sleep".into(),
        name: "bash".into(),
    });
    sink.emit(StreamEvent::ToolCallArgsDelta {
        id: "sleep".into(),
        delta: serde_json::json!({"command":"sleep 20"}).to_string(),
    });
    sink.emit(StreamEvent::ToolCallEnd { id: "sleep".into() });
    sink.emit(StreamEvent::ToolExecutionEnd {
        id: "sleep".into(),
        result: ToolResult::success("sleep", "\n[exit code: 0]"),
    });

    let bytes = writer.bytes();
    let plain = String::from_utf8(strip_ansi_escapes::strip(&bytes)).unwrap();
    assert!(plain.contains("✓ bash  exit 0"), "{plain}");
    assert!(plain.contains("  $ sleep 20"), "{plain}");
    assert!(plain.contains("  (no output)"), "{plain}");
    assert!(!plain.contains("[exit code:"), "{plain}");
    assert!(
        bytes.windows(4).any(|bytes| bytes == b"\x1b[3m"),
        "{bytes:?}"
    );
}

#[test]
fn fish_script_has_metadata_hooks_dedicated_ai_read_and_mode_bindings() {
    let script = fish_script();
    assert!(script.contains("--on-event fish_preexec"));
    assert!(script.contains("--on-event fish_postexec"));
    assert!(script.contains("__record-shell"));
    assert!(script.contains("--fish-session-key"));
    assert!(script.contains("--one-turn"));
    assert!(!script.contains("--resume --one-turn"));
    assert!(script.contains("--fish-ai"));
    assert!(script.contains("A_FISH_AI_PROMPT"));
    assert!(script.contains("a> "));
    assert!(!script.contains("[AI] "));
    assert!(script.contains("read --local --line"));
    assert!(script.contains("--right-prompt"));
    assert!(script.contains("once · tab"));
    assert!(script.contains("multi · tab"));
    assert!(script.contains("__a_handle_tab"));
    assert!(script.contains("__a_ai_prompt_active"));
    assert!(!script.contains("--shell"));
    assert!(script.contains("bind -M default \\cg __a_ai_prompt"));
    assert!(script.contains("bind -M insert \\cg __a_ai_prompt"));
    assert!(!script.contains("bind -M default \\r"));
    assert!(!script.contains("bind -M insert \\r"));
}

#[cfg(unix)]
#[tokio::test]
async fn new_fish_has_immediate_ai_prompt_without_shell_completion_and_invokes_agent() {
    if Command::new("tmux").arg("-V").output().await.is_err() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let bin = temp.path().join("bin");
    let log = temp.path().join("calls.log");
    std::fs::create_dir_all(&bin).unwrap();
    let fake_a = bin.join("a");
    std::fs::write(
        &fake_a,
        "#!/bin/sh\nprintf '<call>' >> \"$A_AGENT_TEST_LOG\"\nprintf '%s|' \"$@\" >> \"$A_AGENT_TEST_LOG\"\nprintf 'prompt=%s|' \"$A_FISH_AI_PROMPT\" >> \"$A_AGENT_TEST_LOG\"\nprintf '\\n' >> \"$A_AGENT_TEST_LOG\"\nprintf 'agent invoked\\n'\n",
    )
    .unwrap();
    std::fs::set_permissions(&fake_a, std::fs::Permissions::from_mode(0o755)).unwrap();
    install_to(&home).unwrap();
    std::fs::write(
        home.join(".config/fish/config.fish"),
        "function fish_prompt\n    printf '[%s]# ' $status\nend\n",
    )
    .unwrap();

    let socket = format!("a-fish-test-{}", std::process::id());
    let session = "fish-mode";
    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap());
    let fish_command = format!(
        "env HOME={} XDG_CONFIG_HOME={}/.config PATH={} A_AGENT_TEST_LOG={} fish",
        home.display(),
        home.display(),
        path,
        log.display()
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
            "24",
            "-s",
            session,
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
    wait_for_prompt(&socket, session).await;
    tmux_type(&socket, session, "false").await;
    tmux_key(&socket, session, "Enter").await;
    wait_for_pane(&socket, session, "[1]#").await;
    tmux_key(&socket, session, "C-g").await;
    let ai_prompt = wait_for_last_line(&socket, session, |line| {
        line.contains("a>") && line.contains("once · tab")
    })
    .await;
    assert!(ai_prompt.contains("a>"), "{ai_prompt:?}");
    tmux_type(&socket, session, "git che").await;
    tmux_key(&socket, session, "Tab").await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let no_completion = tmux_capture(&socket, session).await;
    assert!(!no_completion.contains("checkout"), "{no_completion:?}");
    assert!(
        last_visible_line(&no_completion).contains("multi · tab"),
        "{no_completion:?}"
    );
    let colored = tmux_capture_escaped(&socket, session).await;
    let ai_line = colored
        .lines()
        .find(|line| line.contains("git che"))
        .expect("AI input line missing from colored capture");
    let input = &ai_line[ai_line.find("git che").unwrap()..];
    let input = input.split_once("\x1b[").map_or(input, |(input, _)| input);
    assert_eq!(input.trim_end(), "git che", "{ai_line:?}");
    tmux_key(&socket, session, "C-g").await;
    let normal_mode = wait_for_last_line(&socket, session, |line| line.contains("[1]#")).await;
    assert!(
        last_visible_line(&normal_mode).contains("git che"),
        "{normal_mode:?}"
    );
    assert!(!normal_mode.contains("a> git che"), "{normal_mode:?}");

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    tmux_key(&socket, session, "C-c").await;
    wait_for_prompt(&socket, session).await;
    tmux_key(&socket, session, "C-g").await;
    wait_for_last_line(&socket, session, |line| line.contains("a>")).await;
    tmux_type(&socket, session, "cancel me").await;
    tmux_key(&socket, session, "C-c").await;
    let interrupted = wait_for_last_line(&socket, session, |line| line.contains("[1]#")).await;
    assert!(interrupted.contains("a> cancel me"), "{interrupted:?}");

    tmux_key(&socket, session, "C-g").await;
    wait_for_last_line(&socket, session, |line| {
        line.contains("a>") && line.contains("once · tab")
    })
    .await;
    tmux_key(&socket, session, "Tab").await;
    wait_for_last_line(&socket, session, |line| line.contains("multi · tab")).await;
    tmux_type(&socket, session, "first turn").await;
    tmux_key(&socket, session, "Enter").await;
    wait_for_pane_count(&socket, session, "agent invoked", 1).await;
    wait_for_last_line(&socket, session, |line| {
        line.contains("a>") && line.contains("multi · tab")
    })
    .await;
    tmux_type(&socket, session, "echo ok").await;
    tmux_key(&socket, session, "Tab").await;
    wait_for_last_line(&socket, session, |line| line.contains("once · tab")).await;
    tmux_key(&socket, session, "Enter").await;
    let pane = wait_for_pane_count(&socket, session, "agent invoked", 2).await;
    let calls = std::fs::read_to_string(&log).unwrap();
    assert!(calls.contains("--fish-ai|--fish-session-key|"), "{calls:?}");
    assert!(calls.contains("--one-turn|prompt=first turn|"), "{calls:?}");
    assert!(calls.contains("--one-turn|prompt=echo ok|"), "{calls:?}");
    assert!(!pane.contains("a --fish-ai"), "{pane:?}");
    assert!(pane.contains("agent invoked"), "{pane:?}");
    let final_pane = wait_for_last_line(&socket, session, |line| line.contains("[1]#")).await;
    assert!(
        last_visible_line(&final_pane).contains("[1]#"),
        "{final_pane:?}"
    );
    let _ = Command::new("tmux")
        .args(["-L", &socket, "kill-server"])
        .status()
        .await;
}

async fn tmux_type(socket: &str, session: &str, text: &str) {
    assert!(
        Command::new("tmux")
            .args(["-L", socket, "send-keys", "-t", session, "-l", text])
            .status()
            .await
            .unwrap()
            .success()
    );
}

async fn tmux_key(socket: &str, session: &str, key: &str) {
    let mut arguments = vec!["-L", socket, "send-keys", "-t", session];
    if key == "C-g" {
        arguments.extend(["-H", "07"]);
    } else {
        arguments.push(key);
    }
    assert!(
        Command::new("tmux")
            .args(arguments)
            .status()
            .await
            .unwrap()
            .success()
    );
}

async fn tmux_capture(socket: &str, session: &str) -> String {
    let output = Command::new("tmux")
        .args(["-L", socket, "capture-pane", "-p", "-t", session])
        .output()
        .await
        .unwrap();
    String::from_utf8(output.stdout).unwrap()
}

async fn tmux_capture_escaped(socket: &str, session: &str) -> String {
    let output = Command::new("tmux")
        .args(["-L", socket, "capture-pane", "-p", "-e", "-t", session])
        .output()
        .await
        .unwrap();
    String::from_utf8(output.stdout).unwrap()
}

async fn wait_for_prompt(socket: &str, session: &str) {
    wait_for_last_line(socket, session, |line| line.trim_end().ends_with('#')).await;
}

async fn wait_for_pane(socket: &str, session: &str, needle: &str) -> String {
    for _ in 0..40 {
        let pane = tmux_capture(socket, session).await;
        if pane.contains(needle) {
            return pane;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!(
        "tmux pane did not contain {needle:?}: {:?}",
        tmux_capture(socket, session).await
    );
}

async fn wait_for_pane_count(socket: &str, session: &str, needle: &str, expected: usize) -> String {
    for _ in 0..80 {
        let pane = tmux_capture(socket, session).await;
        if pane.matches(needle).count() >= expected {
            return pane;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!(
        "tmux pane did not contain {expected} occurrences of {needle:?}: {:?}",
        tmux_capture(socket, session).await
    );
}

async fn wait_for_last_line(
    socket: &str,
    session: &str,
    predicate: impl Fn(&str) -> bool,
) -> String {
    for _ in 0..40 {
        let pane = tmux_capture(socket, session).await;
        if pane
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .is_some_and(&predicate)
        {
            return pane;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!(
        "tmux pane did not reach expected state: {:?}",
        tmux_capture(socket, session).await
    );
}

fn last_visible_line(pane: &str) -> &str {
    pane.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or_default()
}

#[test]
fn fish_installer_writes_the_conf_d_asset() {
    let home = tempfile::tempdir().unwrap();
    let path = install_to(home.path()).unwrap();
    assert_eq!(std::fs::read_to_string(path).unwrap(), fish_script());
}

#[test]
fn reasoning_toggle_configuration_accepts_ctrl_character_keys() {
    let editor = InputEditor::with_reasoning_toggle("ctrl-r").unwrap();
    assert!(editor.is_reasoning_toggle(KeyCode::Char('r'), KeyModifiers::CONTROL));
    assert!(!editor.is_reasoning_toggle(KeyCode::Char('o'), KeyModifiers::CONTROL));
    assert!(InputEditor::with_reasoning_toggle("alt-r").is_err());
    assert!(InputEditor::with_reasoning_toggle("ctrl-long").is_err());
}
