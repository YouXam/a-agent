use std::borrow::Cow;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dialoguer::Select;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{
    Cmd, ConditionalEventHandler, Context, Editor, Event, EventContext, EventHandler, Helper,
    KeyCode, KeyEvent, Modifiers, RepeatCount,
};
use unicode_width::UnicodeWidthStr;

use super::InlineRenderer;

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
}

impl Completer for AgentHelper {
    type Candidate = Pair;
}

impl Hinter for AgentHelper {
    type Hint = String;

    fn hint(&self, line: &str, position: usize, _context: &Context<'_>) -> Option<String> {
        if position != line.len() {
            return None;
        }
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
            .then(|| format!("{}{label}", " ".repeat(terminal_width - used - 1)))
    }
}

impl Highlighter for AgentHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
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

struct ModeHandler(Arc<AtomicBool>);

impl ConditionalEventHandler for ModeHandler {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        _context: &EventContext,
    ) -> Option<Cmd> {
        self.0.fetch_xor(true, Ordering::SeqCst);
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
        let editor_config = rustyline::Config::builder()
            .keyseq_timeout(Some(500))
            .build();
        let mut editor = Editor::<AgentHelper, DefaultHistory>::with_config(editor_config)
            .map_err(io::Error::other)?;
        editor.set_helper(Some(AgentHelper {
            multi: multi.clone(),
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
            EventHandler::Conditional(Box::new(ModeHandler(multi.clone()))),
        );
        Ok(Self {
            editor,
            rewind_requested,
            reasoning_requested,
            pending_initial: None,
            reasoning_key,
            multi,
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
