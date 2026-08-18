use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::config::{Config, ProviderKind};
use crate::context::{
    ContextInput, build_system_prompt, discover_agents_for_targets, discover_skills,
};
use crate::fish;
use crate::model::ContentBlock;
use crate::provider::create_provider;
use crate::session::{NewSession, SessionStore, ShellHistoryItem, default_database_path};
use crate::tools::runner::{CoreToolExecutor, ToolRunner};
use crate::tui::{InlineRenderer, InputAction, InputEditor, RenderLimits};

use super::{CliArgs, args};

pub async fn run() -> Result<i32> {
    let mut timing = Timing::new();
    let mut args = args::parse_args(std::env::args().skip(1))?;
    if args.fish_ai && args.prompt.is_none() {
        args.prompt = std::env::var("A_FISH_AI_PROMPT")
            .ok()
            .filter(|prompt| !prompt.trim().is_empty());
    }
    timing.mark("argv_parse");
    if args.help {
        print!("{}", args::HELP);
        return Ok(0);
    }
    if args.version {
        println!("a {}", env!("CARGO_PKG_VERSION"));
        return Ok(0);
    }
    if args.install_fish {
        let path = fish::install()?;
        println!("Installed Fish integration at {}", path.display());
        return Ok(0);
    }

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")?;
    let database_path = default_database_path(&home);
    if let Some(record) = args.shell_record {
        let store = SessionStore::open(&database_path)?;
        let history_limit = Config::load_from(Path::new(&record.cwd), &home)
            .map(|config| config.session.shell_history_limit)
            .unwrap_or(5000);
        store.record_shell_history(
            &record.cwd,
            &record.command,
            record.exit_code,
            record.started_at,
            record.duration_ms,
            record.pipe_status.as_deref(),
        )?;
        store.prune_shell_history(history_limit)?;
        return Ok(0);
    }

    let cwd = std::env::current_dir()?
        .canonicalize()
        .context("resolve current directory")?;
    if !cwd.join(".a/config.toml").is_file()
        && let Some(path) = Config::ensure_user_config(&home)?
    {
        eprintln!("Created config at {}", path.display());
        eprintln!("Set OPENAI_API_KEY or edit the provider section before use.");
    }
    let config = Config::load_from(&cwd, &home)?;
    timing.mark("config");
    let stdin_context = read_stdin_tail(config.context.stdin_max_bytes)?;
    let targets = resolve_targets(&cwd, &args.files)?;
    let global_agents = home.join(".config/a/AGENTS.md");
    let agents = discover_agents_for_targets(&cwd, Some(&global_agents), &targets)?;
    timing.mark("agents_load");
    let project_root = cwd
        .ancestors()
        .find(|path| path.join(".git").exists())
        .unwrap_or(&cwd);
    let skills = discover_skills(
        &home.join(".config/a/skills"),
        &project_root.join(".a/skills"),
    )?;
    timing.mark("skills_index");
    let system_prompt = build_system_prompt(&ContextInput {
        cwd: cwd.clone(),
        agents,
        skills,
        targeted_files: targets,
        platform: std::env::consts::OS.into(),
        shell: std::env::var("SHELL").unwrap_or_else(|_| "unknown".into()),
    });

    let mut store = SessionStore::open(&database_path)?;
    timing.mark("sqlite_open");
    let cwd_text = cwd.to_string_lossy().into_owned();
    let session = resolve_session(&mut store, &args, &cwd_text, &config)?;
    timing.mark("session_lookup");
    let shell_history =
        store.recent_shell_history(&cwd_text, config.context.shell_history_count)?;
    let mut provider_config = config.provider.clone();
    if args.resume {
        provider_config.kind = ProviderKind::parse(&session.provider_type)?;
        provider_config.model = session.model.clone();
    }
    let provider = create_provider(provider_config)?;
    let executor = Arc::new(CoreToolExecutor::new(
        cwd,
        config.context.read_max_lines,
        Duration::from_secs(config.tools.bash_timeout_seconds),
        config.tools.max_output_bytes,
    ));
    let tools = Arc::new(ToolRunner::new(executor, config.tools.max_parallel));
    let store = Arc::new(Mutex::new(store));
    let agent = Agent::new(
        provider,
        tools,
        store.clone(),
        session.id.clone(),
        system_prompt,
        config.session.max_agent_cycles,
    );
    let renderer = InlineRenderer::stdout_with_limits(
        config.ui.show_reasoning,
        RenderLimits {
            tool_input_max_bytes: config.ui.tool_input_max_bytes,
            tool_output_max_bytes: config.ui.tool_output_max_bytes,
            tool_output_max_lines: config.ui.tool_output_max_lines,
            tool_live_output_lines: config.ui.tool_live_output_lines,
        },
    )?;
    let mut input = InputEditor::with_reasoning_toggle(&config.ui.reasoning_toggle)?;
    timing.mark("request_build");
    timing.print();

    if let Some(prompt) = args.prompt.as_deref() {
        if args.fish_ai {
            renderer.begin_turn()?;
        } else {
            renderer.render_user(prompt)?;
        }
        let contextual = contextual_prompt(prompt, stdin_context.as_deref(), &shell_history);
        if run_turn(&agent, &renderer, &contextual).await? {
            return Ok(130);
        }
        if args.one_turn {
            return Ok(0);
        }
    } else if args.one_turn {
        anyhow::bail!("--one-turn requires a prompt");
    }

    loop {
        match input.read_action()? {
            InputAction::Submit(prompt) if !prompt.trim().is_empty() => {
                renderer.begin_turn()?;
                let contextual = contextual_prompt(&prompt, None, &shell_history);
                if run_turn(&agent, &renderer, &contextual).await? {
                    return Ok(130);
                }
            }
            InputAction::Submit(_) => {}
            InputAction::ToggleReasoning => {
                renderer.toggle_reasoning()?;
            }
            InputAction::Rewind => {
                let checkpoints = store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
                    .user_checkpoints(&session.id)?;
                let choices = checkpoints
                    .into_iter()
                    .rev()
                    .map(|item| {
                        let label = item
                            .blocks
                            .iter()
                            .find_map(|block| match block {
                                ContentBlock::Text(text) => {
                                    Some(text.lines().last().unwrap_or(text).to_owned())
                                }
                                _ => None,
                            })
                            .unwrap_or_else(|| item.id.clone());
                        (item.id, label)
                    })
                    .collect::<Vec<_>>();
                if let Some(item_id) = input.select_checkpoint(&choices, &renderer)? {
                    store
                        .lock()
                        .map_err(|_| anyhow::anyhow!("session store lock poisoned"))?
                        .rewind(&session.id, &item_id)?;
                    renderer.render_status("rewound; the previous branch is preserved")?;
                }
            }
            InputAction::Interrupt => return Ok(130),
            InputAction::Eof => return Ok(0),
        }
    }
}

async fn run_turn(agent: &Agent, renderer: &InlineRenderer, prompt: &str) -> Result<bool> {
    let cancel = CancellationToken::new();
    let turn = agent.submit(prompt, renderer.event_sink(), cancel.clone());
    tokio::pin!(turn);
    tokio::select! {
        result = &mut turn => { result?; Ok(false) }
        signal = tokio::signal::ctrl_c() => {
            signal?;
            cancel.cancel();
            let _ = turn.await;
            renderer.render_status("cancelled")?;
            Ok(true)
        }
    }
}

fn resolve_session(
    store: &mut SessionStore,
    args: &CliArgs,
    cwd: &str,
    config: &Config,
) -> Result<crate::session::Session> {
    if let Some(id) = &args.resume_session_id {
        return store
            .get_session(id)?
            .with_context(|| format!("session not found: {id}"));
    }
    if args.resume
        && let Some(session) = store.find_latest_session(cwd)?
    {
        return Ok(session);
    }
    store.create_session(NewSession::new(
        cwd,
        config.provider.kind.as_str(),
        &config.provider.model,
    ))
}

fn resolve_targets(cwd: &Path, files: &[String]) -> Result<Vec<PathBuf>> {
    files
        .iter()
        .map(|file| {
            let path = Path::new(file);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                cwd.join(path)
            };
            path.canonicalize()
                .with_context(|| format!("resolve targeted path {}", path.display()))
        })
        .collect()
}

fn read_stdin_tail(max_bytes: usize) -> Result<Option<String>> {
    if io::stdin().is_terminal() {
        return Ok(None);
    }
    let mut input = io::stdin().lock();
    let mut tail = Vec::new();
    let mut buffer = [0_u8; 8192];
    let mut total = 0_usize;
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count;
        tail.extend_from_slice(&buffer[..count]);
        if tail.len() > max_bytes {
            let remove = tail.len() - max_bytes;
            tail.drain(..remove);
        }
    }
    if total == 0 {
        return Ok(None);
    }
    while tail
        .first()
        .is_some_and(|byte| byte & 0b1100_0000 == 0b1000_0000)
    {
        tail.remove(0);
    }
    let text = String::from_utf8_lossy(&tail);
    Ok(Some(if total > max_bytes {
        format!("[stdin truncated; showing last {max_bytes} bytes]\n{text}")
    } else {
        text.into_owned()
    }))
}

fn contextual_prompt(prompt: &str, stdin: Option<&str>, shell: &[ShellHistoryItem]) -> String {
    let mut sections = Vec::new();
    if !shell.is_empty() {
        let commands = shell
            .iter()
            .map(|item| {
                format!(
                    "- `{}` -> exit {}",
                    item.command,
                    item.exit_code
                        .map_or_else(|| "?".into(), |code| code.to_string())
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        sections.push(format!("Recent shell activity:\n{commands}"));
    }
    if let Some(stdin) = stdin {
        sections.push(format!("User-provided stdin:\n\n{stdin}"));
    }
    sections.push(prompt.to_owned());
    sections.join("\n\n")
}

struct Timing {
    enabled: bool,
    started: Instant,
    last: Instant,
    entries: Vec<(&'static str, Duration)>,
}

impl Timing {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            enabled: std::env::var_os("A_DEBUG_TIMING").is_some(),
            started: now,
            last: now,
            entries: Vec::new(),
        }
    }
    fn mark(&mut self, name: &'static str) {
        if self.enabled {
            let now = Instant::now();
            self.entries.push((name, now - self.last));
            self.last = now;
        }
    }
    fn print(&self) {
        if self.enabled {
            eprintln!("timing:");
            for (name, duration) in &self.entries {
                eprintln!("{name:<18} {:>7.2} ms", duration.as_secs_f64() * 1000.0);
            }
            eprintln!(
                "{:<18} {:>7.2} ms",
                "pre-network",
                self.started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}
