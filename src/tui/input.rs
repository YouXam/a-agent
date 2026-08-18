use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rustyline::error::ReadlineError;
use rustyline::{
    Cmd, ConditionalEventHandler, DefaultEditor, Event, EventContext, EventHandler, KeyCode,
    KeyEvent, Modifiers, RepeatCount,
};

use super::InlineRenderer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputAction {
    Submit(String),
    Rewind,
    ToggleReasoning,
    Interrupt,
    Eof,
}

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

pub struct InputEditor {
    editor: DefaultEditor,
    rewind_requested: Arc<AtomicBool>,
    reasoning_requested: Arc<Mutex<Option<(String, String)>>>,
    pending_initial: Option<(String, String)>,
    reasoning_key: char,
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
        let mut editor = DefaultEditor::new().map_err(io::Error::other)?;
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
        Ok(Self {
            editor,
            rewind_requested,
            reasoning_requested,
            pending_initial: None,
            reasoning_key,
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
        let prompt = ("a> ", "\x1b[1;36ma> \x1b[0m");
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
                Ok(InputAction::Submit(line))
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
