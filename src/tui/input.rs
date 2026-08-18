use std::borrow::Cow;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dialoguer::Select;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hint, Hinter};
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{
    Cmd, ConditionalEventHandler, Context, Editor, Event, EventContext, EventHandler, Helper,
    KeyCode, KeyEvent, Modifiers, RepeatCount,
};
use unicode_width::UnicodeWidthStr;

use super::InlineRenderer;

const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand::new("/model", "/model [profile]", "Switch model profile"),
    SlashCommand::new("/effort", "/effort [level]", "Set reasoning effort"),
    SlashCommand::new("/thinking", "/thinking", "Toggle reasoning visibility"),
    SlashCommand::new("/status", "/status", "Show session and model status"),
    SlashCommand::new("/clear", "/clear", "Clear the current conversation"),
    SlashCommand::new("/compact", "/compact", "Summarize the current conversation"),
    SlashCommand::new(
        "/resume",
        "/resume [session-id]",
        "Resume a session in this cwd",
    ),
    SlashCommand::new("/help", "/help", "Show available commands"),
];

struct SlashCommand {
    name: &'static str,
    usage: &'static str,
    description: &'static str,
}

impl SlashCommand {
    const fn new(name: &'static str, usage: &'static str, description: &'static str) -> Self {
        Self {
            name,
            usage,
            description,
        }
    }
}

#[derive(Default)]
struct PaletteState(Mutex<PaletteSelection>);

#[derive(Default)]
struct PaletteSelection {
    prefix: String,
    index: usize,
}

impl PaletteState {
    fn selected(&self, prefix: &str, count: usize) -> usize {
        let Ok(mut state) = self.0.lock() else {
            return 0;
        };
        if state.prefix != prefix {
            state.prefix = prefix.into();
            state.index = 0;
        }
        state.index = state.index.min(count.saturating_sub(1));
        state.index
    }

    fn move_selection(&self, prefix: &str, count: usize, direction: isize) {
        if count == 0 {
            return;
        }
        let Ok(mut state) = self.0.lock() else {
            return;
        };
        if state.prefix != prefix {
            state.prefix = prefix.into();
            state.index = 0;
        }
        state.index = (state.index as isize + direction).rem_euclid(count as isize) as usize;
    }

    fn clear(&self) {
        if let Ok(mut state) = self.0.lock() {
            state.prefix.clear();
            state.index = 0;
        }
    }
}

struct AgentHint(String);

impl Hint for AgentHint {
    fn display(&self) -> &str {
        &self.0
    }

    fn completion(&self) -> Option<&str> {
        None
    }
}

fn command_prefix(line: &str, position: usize) -> Option<&str> {
    if position != line.len() {
        return None;
    }
    let prefix = &line[..position];
    (prefix.starts_with('/') && !prefix.chars().any(char::is_whitespace)).then_some(prefix)
}

fn matching_commands(prefix: &str) -> Vec<&'static SlashCommand> {
    SLASH_COMMANDS
        .iter()
        .filter(|command| command.name.starts_with(prefix))
        .collect()
}

fn command_row(command: &SlashCommand, selected: bool, color: bool) -> String {
    let marker = if selected { '›' } else { ' ' };
    let row = format!("{marker} {:<22} {}", command.usage, command.description);
    if !color {
        return row;
    }
    if selected {
        format!("\x1b[1;96m{row}\x1b[0m")
    } else {
        format!("\x1b[90m{row}\x1b[0m")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputAction {
    Submit(String, InputMode),
    Rewind,
    ToggleReasoning,
    Interrupt,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Once,
    Multi,
}

#[derive(Clone)]
struct AgentHelper {
    multi: Arc<AtomicBool>,
    palette: Arc<PaletteState>,
}

impl Completer for AgentHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        position: usize,
        _context: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        let Some(prefix) = command_prefix(line, position) else {
            return Ok((0, Vec::new()));
        };
        let commands = matching_commands(prefix);
        let selected = self.palette.selected(prefix, commands.len());
        let candidates = commands
            .get(selected)
            .map(|command| Pair {
                display: command.usage.into(),
                replacement: format!("{} ", command.name),
            })
            .into_iter()
            .collect();
        Ok((0, candidates))
    }
}

impl Hinter for AgentHelper {
    type Hint = AgentHint;

    fn hint(&self, line: &str, position: usize, _context: &Context<'_>) -> Option<AgentHint> {
        if position != line.len() {
            self.palette.clear();
            return None;
        }
        if let Some(prefix) = command_prefix(line, position) {
            let commands = matching_commands(prefix);
            let selected = self.palette.selected(prefix, commands.len());
            let color = std::env::var_os("NO_COLOR").is_none();
            let rows = commands
                .iter()
                .enumerate()
                .map(|(index, command)| command_row(command, index == selected, color))
                .collect::<Vec<_>>();
            return Some(AgentHint(if rows.is_empty() {
                "\n  No matching commands".into()
            } else {
                format!("\n{}", rows.join("\n"))
            }));
        }
        self.palette.clear();
        let label = if self.multi.load(Ordering::SeqCst) {
            "multi · tab"
        } else {
            "once · tab"
        };
        let terminal_width = crossterm::terminal::size()
            .map(|(width, _)| usize::from(width))
            .unwrap_or(80);
        let used = 3 + UnicodeWidthStr::width(line) + UnicodeWidthStr::width(label);
        (terminal_width > used + 1)
            .then(|| AgentHint(format!("{}{label}", " ".repeat(terminal_width - used - 1))))
    }
}

impl Highlighter for AgentHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        if hint.starts_with('\n') {
            return Cow::Borrowed(hint);
        }
        if self.multi.load(Ordering::SeqCst) {
            Cow::Owned(format!("\x1b[1;95m{hint}\x1b[0m"))
        } else {
            Cow::Owned(format!("\x1b[90m{hint}\x1b[0m"))
        }
    }
}

impl Validator for AgentHelper {}
impl Helper for AgentHelper {}

struct RewindHandler(Arc<AtomicBool>);

impl ConditionalEventHandler for RewindHandler {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        _context: &EventContext,
    ) -> Option<Cmd> {
        self.0.store(true, Ordering::SeqCst);
        Some(Cmd::Interrupt)
    }
}

struct ReasoningHandler(Arc<Mutex<Option<(String, String)>>>);

impl ConditionalEventHandler for ReasoningHandler {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        context: &EventContext,
    ) -> Option<Cmd> {
        let position = context.pos();
        let line = context.line();
        if let Ok(mut pending) = self.0.lock() {
            *pending = Some((line[..position].to_owned(), line[position..].to_owned()));
        }
        Some(Cmd::Interrupt)
    }
}

struct TabHandler {
    multi: Arc<AtomicBool>,
    palette: Arc<PaletteState>,
}

impl ConditionalEventHandler for TabHandler {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        context: &EventContext,
    ) -> Option<Cmd> {
        if let Some(prefix) = command_prefix(context.line(), context.pos()) {
            let commands = matching_commands(prefix);
            self.palette.selected(prefix, commands.len());
            return Some(Cmd::Complete);
        }
        self.multi.fetch_xor(true, Ordering::SeqCst);
        Some(Cmd::Repaint)
    }
}

struct PaletteNavigation {
    palette: Arc<PaletteState>,
    direction: isize,
}

impl ConditionalEventHandler for PaletteNavigation {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        context: &EventContext,
    ) -> Option<Cmd> {
        let prefix = command_prefix(context.line(), context.pos())?;
        let count = matching_commands(prefix).len();
        self.palette.move_selection(prefix, count, self.direction);
        Some(Cmd::Repaint)
    }
}

pub struct InputEditor {
    editor: Editor<AgentHelper, DefaultHistory>,
    rewind_requested: Arc<AtomicBool>,
    reasoning_requested: Arc<Mutex<Option<(String, String)>>>,
    pending_initial: Option<(String, String)>,
    reasoning_key: char,
    multi: Arc<AtomicBool>,
    palette: Arc<PaletteState>,
}

impl InputEditor {
    pub fn with_reasoning_toggle(value: &str) -> io::Result<Self> {
        let reasoning_key = value
            .strip_prefix("ctrl-")
            .and_then(|value| {
                let mut characters = value.chars();
                let key = characters.next()?;
                characters.next().is_none().then_some(key)
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "reasoning toggle must use ctrl-<character>",
                )
            })?;
        let rewind_requested = Arc::new(AtomicBool::new(false));
        let reasoning_requested = Arc::new(Mutex::new(None));
        let multi = Arc::new(AtomicBool::new(true));
        let palette = Arc::new(PaletteState::default());
        let editor_config = rustyline::Config::builder()
            .keyseq_timeout(Some(500))
            .build();
        let mut editor = Editor::<AgentHelper, DefaultHistory>::with_config(editor_config)
            .map_err(io::Error::other)?;
        editor.set_helper(Some(AgentHelper {
            multi: multi.clone(),
            palette: palette.clone(),
        }));
        editor.bind_sequence(
            Event::KeySeq(vec![KeyEvent::from('\x1b'), KeyEvent::from('\x1b')]),
            EventHandler::Conditional(Box::new(RewindHandler(rewind_requested.clone()))),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Esc, Modifiers::ALT),
            EventHandler::Conditional(Box::new(RewindHandler(rewind_requested.clone()))),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Esc, Modifiers::NONE),
            EventHandler::Conditional(Box::new(RewindHandler(rewind_requested.clone()))),
        );
        editor.bind_sequence(
            KeyEvent::ctrl(reasoning_key),
            EventHandler::Conditional(Box::new(ReasoningHandler(reasoning_requested.clone()))),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Tab, Modifiers::NONE),
            EventHandler::Conditional(Box::new(TabHandler {
                multi: multi.clone(),
                palette: palette.clone(),
            })),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Up, Modifiers::NONE),
            EventHandler::Conditional(Box::new(PaletteNavigation {
                palette: palette.clone(),
                direction: -1,
            })),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Down, Modifiers::NONE),
            EventHandler::Conditional(Box::new(PaletteNavigation {
                palette: palette.clone(),
                direction: 1,
            })),
        );
        Ok(Self {
            editor,
            rewind_requested,
            reasoning_requested,
            pending_initial: None,
            reasoning_key,
            multi,
            palette,
        })
    }

    pub fn reasoning_key(&self) -> char {
        self.reasoning_key
    }

    pub fn is_reasoning_toggle(
        &self,
        code: crossterm::event::KeyCode,
        modifiers: crossterm::event::KeyModifiers,
    ) -> bool {
        matches!(code, crossterm::event::KeyCode::Char(character) if character == self.reasoning_key)
            && modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
    }

    pub fn read_action(&mut self) -> io::Result<InputAction> {
        let prompt = ("a> ", "\x1b[1;96ma> \x1b[0m");
        let result = if let Some((left, right)) = self.pending_initial.take() {
            self.editor.readline_with_initial(&prompt, (&left, &right))
        } else {
            self.editor.readline(&prompt)
        };
        match result {
            Ok(line) => {
                let line = self.accept_palette_selection(line);
                if !line.trim().is_empty() {
                    self.editor
                        .add_history_entry(line.as_str())
                        .map_err(io::Error::other)?;
                }
                let mode = if self.multi.load(Ordering::SeqCst) {
                    InputMode::Multi
                } else {
                    InputMode::Once
                };
                Ok(InputAction::Submit(line, mode))
            }
            Err(ReadlineError::Interrupted)
                if self.rewind_requested.swap(false, Ordering::SeqCst) =>
            {
                Ok(InputAction::Rewind)
            }
            Err(ReadlineError::Interrupted) if self.take_reasoning_request() => {
                Ok(InputAction::ToggleReasoning)
            }
            Err(ReadlineError::Interrupted) => Ok(InputAction::Interrupt),
            Err(ReadlineError::Eof) => Ok(InputAction::Eof),
            Err(error) => Err(io::Error::other(error)),
        }
    }

    fn accept_palette_selection(&self, line: String) -> String {
        let Some(prefix) = command_prefix(&line, line.len()) else {
            return line;
        };
        let commands = matching_commands(prefix);
        let selected = self.palette.selected(prefix, commands.len());
        commands
            .get(selected)
            .map_or(line, |command| command.name.into())
    }

    pub fn add_history_entries(&mut self, entries: &[String]) -> io::Result<()> {
        for entry in entries {
            self.editor
                .add_history_entry(entry.as_str())
                .map_err(io::Error::other)?;
        }
        Ok(())
    }

    pub fn select_option(
        &mut self,
        prompt: &str,
        choices: &[String],
        default: usize,
    ) -> io::Result<Option<usize>> {
        if choices.is_empty() {
            return Ok(None);
        }
        Select::new()
            .with_prompt(prompt)
            .items(choices)
            .default(default.min(choices.len() - 1))
            .interact_opt()
            .map_err(io::Error::other)
    }

    fn take_reasoning_request(&mut self) -> bool {
        let Ok(mut requested) = self.reasoning_requested.lock() else {
            return false;
        };
        let Some(initial) = requested.take() else {
            return false;
        };
        self.pending_initial = Some(initial);
        true
    }

    pub fn select_checkpoint(
        &mut self,
        checkpoints: &[(String, String)],
        renderer: &InlineRenderer,
    ) -> io::Result<Option<String>> {
        if checkpoints.is_empty() {
            renderer.render_status("no user messages to rewind to")?;
            return Ok(None);
        }
        renderer.render_rewind_options(checkpoints)?;
        loop {
            match self
                .editor
                .readline(&("rewind> ", "\x1b[1;35mrewind> \x1b[0m"))
            {
                Ok(value) if value.trim().is_empty() => return Ok(None),
                Ok(value) => match value.trim().parse::<usize>() {
                    Ok(index) if (1..=checkpoints.len()).contains(&index) => {
                        return Ok(Some(checkpoints[index - 1].0.clone()));
                    }
                    _ => renderer.render_status("enter a checkpoint number, or blank to cancel")?,
                },
                Err(ReadlineError::Interrupted | ReadlineError::Eof) => return Ok(None),
                Err(error) => return Err(io::Error::other(error)),
            }
        }
    }
}
