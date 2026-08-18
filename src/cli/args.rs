use std::fmt;
use std::path::Path;

pub const HELP: &str = r#"a - fast terminal coding agent

Usage:
  a [OPTIONS] [FILES...] [PROMPT]

Options:
  -1, --one-turn       Exit after one complete user/tool/model turn
  -r, --resume         Resume the latest session for the current directory
      --session ID     Resume a specific session
      --install-fish   Install Fish shell integration
  -h, --help           Show help
  -V, --version        Show version
"#;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CliArgs {
    pub one_turn: bool,
    pub resume: bool,
    pub resume_session_id: Option<String>,
    pub files: Vec<String>,
    pub prompt: Option<String>,
    pub help: bool,
    pub version: bool,
    pub install_fish: bool,
    pub fish_ai: bool,
    pub fish_session_key: Option<String>,
    pub shell_record: Option<ShellRecordArgs>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellRecordArgs {
    pub cwd: String,
    pub command: String,
    pub exit_code: Option<i32>,
    pub started_at: i64,
    pub duration_ms: Option<i64>,
    pub pipe_status: Option<String>,
    pub fish_session_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliArgs, CliError> {
    parse_args_with(args, |value| Path::new(value).exists())
}

pub fn parse_args_with(
    args: impl IntoIterator<Item = String>,
    is_path: impl Fn(&str) -> bool,
) -> Result<CliArgs, CliError> {
    let args = args.into_iter().collect::<Vec<_>>();
    if args.first().is_some_and(|value| value == "__record-shell") {
        return parse_shell_record(&args[1..]);
    }
    let mut parsed = CliArgs::default();
    let mut positional = Vec::new();
    let mut iter = args.into_iter().peekable();
    let mut options = true;

    while let Some(arg) = iter.next() {
        if options && arg == "--" {
            options = false;
            continue;
        }
        if options {
            match arg.as_str() {
                "-1" | "--one-turn" => parsed.one_turn = true,
                "-r" | "--resume" => parsed.resume = true,
                "-h" | "--help" => parsed.help = true,
                "-V" | "--version" => parsed.version = true,
                "--install-fish" => parsed.install_fish = true,
                "--fish-ai" => parsed.fish_ai = true,
                "--fish-session-key" => {
                    parsed.fish_session_key = Some(
                        iter.next()
                            .filter(|value| !value.is_empty())
                            .ok_or_else(|| {
                                CliError("--fish-session-key requires a value".into())
                            })?,
                    );
                }
                "--session" => {
                    parsed.resume = true;
                    parsed.resume_session_id = Some(
                        iter.next()
                            .ok_or_else(|| CliError("--session requires an ID".into()))?,
                    );
                }
                value if value.starts_with('-') => {
                    return Err(CliError(format!("unknown option: {value}")));
                }
                _ => positional.push(arg),
            }
        } else {
            positional.push(arg);
        }
    }

    if parsed.resume
        && parsed.resume_session_id.is_none()
        && positional
            .first()
            .is_some_and(|value| value.starts_with("a_"))
    {
        parsed.resume_session_id = Some(positional.remove(0));
    }

    let file_count = positional.iter().take_while(|value| is_path(value)).count();
    parsed.files.extend(positional.drain(..file_count));
    if !positional.is_empty() {
        parsed.prompt = Some(positional.join(" "));
    }
    Ok(parsed)
}

fn parse_shell_record(args: &[String]) -> Result<CliArgs, CliError> {
    let mut cwd = None;
    let mut command = None;
    let mut exit_code = None;
    let mut started_at = None;
    let mut duration_ms = None;
    let mut pipe_status = None;
    let mut fish_session_key = None;
    let mut index = 0;
    while index < args.len() {
        let key = &args[index];
        let value = args
            .get(index + 1)
            .ok_or_else(|| CliError(format!("{key} requires a value")))?;
        match key.as_str() {
            "--cwd" => cwd = Some(value.clone()),
            "--command" => command = Some(value.clone()),
            "--exit-code" => {
                exit_code = Some(
                    value
                        .parse()
                        .map_err(|_| CliError("--exit-code must be an integer".into()))?,
                )
            }
            "--started-at" => {
                started_at = Some(
                    value
                        .parse()
                        .map_err(|_| CliError("--started-at must be an integer".into()))?,
                )
            }
            "--duration-ms" => {
                duration_ms = Some(
                    value
                        .parse()
                        .map_err(|_| CliError("--duration-ms must be an integer".into()))?,
                )
            }
            "--pipe-status" => pipe_status = Some(value.clone()),
            "--fish-session-key" => fish_session_key = Some(value.clone()),
            _ => return Err(CliError(format!("unknown shell record option: {key}"))),
        }
        index += 2;
    }
    Ok(CliArgs {
        shell_record: Some(ShellRecordArgs {
            cwd: cwd.ok_or_else(|| CliError("__record-shell requires --cwd".into()))?,
            command: command.ok_or_else(|| CliError("__record-shell requires --command".into()))?,
            exit_code,
            started_at: started_at
                .ok_or_else(|| CliError("__record-shell requires --started-at".into()))?,
            duration_ms,
            pipe_status,
            fish_session_key,
        }),
        ..CliArgs::default()
    })
}
