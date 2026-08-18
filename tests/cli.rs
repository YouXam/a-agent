use std::collections::HashSet;

use a_agent::cli::{CliArgs, parse_args_with};

fn parser(args: &[&str]) -> CliArgs {
    let paths = HashSet::from(["src/a.rs", "src/b.rs"]);
    parse_args_with(args.iter().map(ToString::to_string), |value| {
        paths.contains(value)
    })
    .expect("arguments should parse")
}

#[test]
fn parses_one_turn_prompt_and_targeted_files() {
    let parsed = parser(&["-1", "src/a.rs", "src/b.rs", "fix it"]);
    assert!(parsed.one_turn);
    assert_eq!(parsed.files, ["src/a.rs", "src/b.rs"]);
    assert_eq!(parsed.prompt.as_deref(), Some("fix it"));
}

#[test]
fn treats_one_existing_path_as_a_target() {
    let parsed = parser(&["src/a.rs"]);
    assert_eq!(parsed.files, ["src/a.rs"]);
    assert_eq!(parsed.prompt, None);
}

#[test]
fn distinguishes_cwd_resume_from_explicit_session() {
    let cwd_resume = parser(&["-r", "continue"]);
    assert!(cwd_resume.resume);
    assert_eq!(cwd_resume.prompt.as_deref(), Some("continue"));

    let explicit = parser(&["-r", "a_1234", "continue"]);
    assert_eq!(explicit.resume_session_id.as_deref(), Some("a_1234"));
    assert_eq!(explicit.prompt.as_deref(), Some("continue"));
}

#[test]
fn double_dash_stops_option_parsing() {
    let parsed = parser(&["-1", "--", "-r is text"]);
    assert_eq!(parsed.prompt.as_deref(), Some("-r is text"));
}

#[test]
fn rejects_unknown_options() {
    let error = parse_args_with(["--wat".to_owned()], |_| false).unwrap_err();
    assert!(error.to_string().contains("unknown option"));
}

#[test]
fn parses_hidden_fish_ai_mode() {
    let parsed = parser(&["--fish-ai", "-r", "-1", "fix it"]);
    assert!(parsed.fish_ai);
    assert!(parsed.resume);
    assert!(parsed.one_turn);
    assert_eq!(parsed.prompt.as_deref(), Some("fix it"));
}

#[test]
fn parses_internal_shell_history_record() {
    let parsed = parse_args_with(
        [
            "__record-shell",
            "--cwd",
            "/repo",
            "--command",
            "cargo test",
            "--exit-code",
            "101",
            "--started-at",
            "123",
            "--duration-ms",
            "45",
            "--pipe-status",
            "101 0",
        ]
        .into_iter()
        .map(str::to_owned),
        |_| false,
    )
    .unwrap();
    let record = parsed.shell_record.unwrap();
    assert_eq!(record.cwd, "/repo");
    assert_eq!(record.command, "cargo test");
    assert_eq!(record.exit_code, Some(101));
    assert_eq!(record.started_at, 123);
    assert_eq!(record.duration_ms, Some(45));
    assert_eq!(record.pipe_status.as_deref(), Some("101 0"));
}
