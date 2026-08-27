use std::borrow::Cow;
use std::io;
use std::path::{Path, PathBuf};
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
    KeyCode, KeyEvent, Modifiers, Movement, RepeatCount,
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
    from: Option<String>,
    applied: Option<String>,
    re_anchor: bool,
    index: usize,
}

impl PaletteState {
    /// Returns the token the palette filters on. Navigation writes the
    /// highlighted entry into the input, so the buffer alone cannot be the
    /// filter: once it holds a full entry the list would collapse to that one
    /// row. What the user typed is kept while the buffer holds either side of a
    /// completion this palette requested, and is refreshed as soon as the user
    /// edits the token themselves. Accepting with Tab re-anchors the filter to
    /// the accepted text, which is what lets `@src/` list the directory it just
    /// completed to.
    fn filter_prefix(&self, token: &str) -> String {
        let Ok(mut state) = self.0.lock() else {
            return token.to_owned();
        };
        let ours = state.applied.as_deref() == Some(token) || state.from.as_deref() == Some(token);
        if state.applied.is_some() && ours {
            if state.re_anchor && state.applied.as_deref() == Some(token) {
                state.prefix = token.into();
                state.re_anchor = false;
                state.from = None;
                state.applied = None;
                state.index = 0;
            }
            return state.prefix.clone();
        }
        if state.prefix != token {
            state.prefix = token.into();
            state.index = 0;
        }
        state.from = None;
        state.applied = None;
        state.re_anchor = false;
        state.prefix.clone()
    }

    fn selected(&self, count: usize) -> usize {
        let Ok(mut state) = self.0.lock() else {
            return 0;
        };
        state.index = state.index.min(count.saturating_sub(1));
        state.index
    }

    fn move_selection(&self, count: usize, direction: isize) -> usize {
        if count == 0 {
            return 0;
        }
        let Ok(mut state) = self.0.lock() else {
            return 0;
        };
        state.index = (state.index as isize + direction).rem_euclid(count as isize) as usize;
        state.index
    }

    fn request_completion(&self, from: &str, to: &str, re_anchor: bool) {
        if let Ok(mut state) = self.0.lock() {
            state.from = Some(from.to_owned());
            state.applied = Some(to.to_owned());
            state.re_anchor = re_anchor;
        }
    }

    /// The text a handler asked to complete to. Read only, so the completer
    /// cannot disturb the selection it is being asked to render.
    fn pending_completion(&self) -> Option<String> {
        self.0.lock().ok()?.applied.clone()
    }

    /// The currently highlighted candidate for `token`, resolved against the
    /// filter the user actually typed.
    fn highlighted(&self, token: &PaletteToken<'_>, cwd: &Path) -> Option<Candidate> {
        let filter = self.filter_prefix(token.text);
        let filtered = PaletteToken {
            start: token.start,
            text: &filter,
            kind: token.kind,
        };
        let mut rows = candidates(&filtered, cwd);
        let selected = self.selected(rows.len());
        (selected < rows.len()).then(|| rows.swap_remove(selected))
    }

    /// Moves the selection and returns the newly highlighted candidate.
    fn navigate(
        &self,
        token: &PaletteToken<'_>,
        cwd: &Path,
        direction: isize,
    ) -> Option<Candidate> {
        let filter = self.filter_prefix(token.text);
        let filtered = PaletteToken {
            start: token.start,
            text: &filter,
            kind: token.kind,
        };
        let mut rows = candidates(&filtered, cwd);
        let selected = self.move_selection(rows.len(), direction);
        (selected < rows.len()).then(|| rows.swap_remove(selected))
    }

    fn clear(&self) {
        if let Ok(mut state) = self.0.lock() {
            state.prefix.clear();
            state.from = None;
            state.applied = None;
            state.re_anchor = false;
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

/// Directories that are never worth completing into.
const SKIPPED_DIRECTORIES: &[&str] = &[".git", "node_modules", "target"];
const PATH_ROWS: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaletteKind {
    Slash,
    Path,
}

struct PaletteToken<'a> {
    start: usize,
    text: &'a str,
    kind: PaletteKind,
}

/// Start of the last whitespace-separated token, treating `\ ` as part of the
/// token so a mention of a path containing spaces stays one token.
fn token_start(line: &str) -> usize {
    let mut start = 0;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if character.is_whitespace() && !escaped {
            start = index + character.len_utf8();
        }
        escaped = character == '\\' && !escaped;
    }
    start
}

/// Splits a line the same way, then keeps the `@` mentions with their escapes
/// removed. Shared with the palette so completion and resolution agree on where
/// a mention ends.
pub fn mention_paths(line: &str) -> Vec<String> {
    let mut mentions = Vec::new();
    let mut token = String::new();
    let mut escaped = false;
    let mut push = |token: &mut String| {
        if let Some(mention) = token.strip_prefix('@') {
            let mention = mention.trim_end_matches([',', '.', ';', ':', ')']);
            if !mention.is_empty() {
                mentions.push(mention.to_owned());
            }
        }
        token.clear();
    };
    for character in line.chars() {
        if escaped {
            token.push(character);
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            character if character.is_whitespace() => push(&mut token),
            character => token.push(character),
        }
    }
    push(&mut token);
    mentions
}

fn unescape_mention(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut escaped = false;
    for character in text.chars() {
        if escaped {
            out.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else {
            out.push(character);
        }
    }
    out
}

fn escape_mention(path: &str) -> String {
    path.chars()
        .flat_map(|character| {
            let escape = character.is_whitespace() || character == '\\';
            escape.then_some('\\').into_iter().chain([character])
        })
        .collect()
}

/// The token under the cursor that the palette can complete: a slash command at
/// the start of the line, or an `@path` anywhere in it.
fn palette_token(line: &str, position: usize) -> Option<PaletteToken<'_>> {
    if position != line.len() {
        return None;
    }
    let start = token_start(line);
    let text = &line[start..];
    let kind = match text.chars().next()? {
        '/' if start == 0 => PaletteKind::Slash,
        '@' => PaletteKind::Path,
        _ => return None,
    };
    Some(PaletteToken { start, text, kind })
}

/// A palette row: the text a completion writes, plus how it is displayed.
struct Candidate {
    completion: String,
    label: String,
    detail: String,
}

fn candidates(token: &PaletteToken<'_>, cwd: &Path) -> Vec<Candidate> {
    match token.kind {
        PaletteKind::Slash => SLASH_COMMANDS
            .iter()
            .filter(|command| command.name.starts_with(token.text))
            .map(|command| Candidate {
                completion: command.name.into(),
                label: command.usage.into(),
                detail: command.description.into(),
            })
            .collect(),
        PaletteKind::Path => path_candidates(&unescape_mention(&token.text[1..]), cwd),
    }
}

fn path_candidates(typed: &str, cwd: &Path) -> Vec<Candidate> {
    let (parent, name) = typed
        .rsplit_once('/')
        .map_or(("", typed), |(parent, name)| (parent, name));
    let directory = if parent.is_empty() {
        cwd.to_path_buf()
    } else {
        cwd.join(parent)
    };
    let Ok(entries) = std::fs::read_dir(&directory) else {
        return Vec::new();
    };
    let mut rows = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let file_name = entry.file_name().into_string().ok()?;
            if !file_name.starts_with(name) {
                return None;
            }
            if file_name.starts_with('.') && !name.starts_with('.') {
                return None;
            }
            let is_directory = entry.file_type().ok()?.is_dir();
            if is_directory && SKIPPED_DIRECTORIES.contains(&file_name.as_str()) {
                return None;
            }
            let relative = if parent.is_empty() {
                file_name.clone()
            } else {
                format!("{parent}/{file_name}")
            };
            let suffix = if is_directory { "/" } else { "" };
            Some((
                is_directory,
                file_name,
                Candidate {
                    // Whitespace is escaped so the mention survives the split
                    // that resolves it into a target.
                    completion: format!("@{}{suffix}", escape_mention(&relative)),
                    label: format!("{relative}{suffix}"),
                    detail: if is_directory { "directory" } else { "file" }.into(),
                },
            ))
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    rows.into_iter()
        .take(PATH_ROWS)
        .map(|(_, _, candidate)| candidate)
        .collect()
}

fn candidate_row(candidate: &Candidate, selected: bool, color: bool) -> String {
    let marker = if selected { '›' } else { ' ' };
    let row = format!("{marker} {:<22} {}", candidate.label, candidate.detail);
    if !color {
        return row;
    }
    if selected {
        format!("\x1b[1;36m{row}\x1b[0m")
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
    cwd: PathBuf,
}

impl Completer for AgentHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        position: usize,
        _context: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        let Some(token) = palette_token(line, position) else {
            return Ok((0, Vec::new()));
        };
        let candidates = self
            .palette
            .pending_completion()
            .map(|target| Pair {
                display: target.clone(),
                replacement: target,
            })
            .into_iter()
            .collect();
        Ok((token.start, candidates))
    }
}

impl Hinter for AgentHelper {
    type Hint = AgentHint;

    fn hint(&self, line: &str, position: usize, _context: &Context<'_>) -> Option<AgentHint> {
        if position != line.len() {
            self.palette.clear();
            return None;
        }
        if let Some(token) = palette_token(line, position) {
            let filter = self.palette.filter_prefix(token.text);
            let filtered = PaletteToken {
                start: token.start,
                text: &filter,
                kind: token.kind,
            };
            let rows = candidates(&filtered, &self.cwd);
            let selected = self.palette.selected(rows.len());
            let color = std::env::var_os("NO_COLOR").is_none();
            let rendered = rows
                .iter()
                .enumerate()
                .map(|(index, candidate)| candidate_row(candidate, index == selected, color))
                .collect::<Vec<_>>();
            return Some(AgentHint(if rendered.is_empty() {
                match token.kind {
                    PaletteKind::Slash => "\n  No matching commands".into(),
                    PaletteKind::Path => "\n  No matching paths".into(),
                }
            } else {
                format!("\n{}", rendered.join("\n"))
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
            Cow::Owned(format!("\x1b[1;35m{hint}\x1b[0m"))
        } else {
            Cow::Owned(format!("\x1b[90m{hint}\x1b[0m"))
        }
    }
}

impl Validator for AgentHelper {}
impl Helper for AgentHelper {}

/// How many times a selector redraws before giving up. A resize storm from a
/// dragged window edge produces a burst of signals, not an endless stream.
const RESIZE_RETRIES: usize = 32;

/// Whether a read failed because a signal arrived rather than because the user
/// pressed Ctrl+C, which `console` also reports as `Interrupted` but without an
/// errno.
fn is_signal_interruption(error: &io::Error) -> bool {
    error.raw_os_error() == Some(libc::EINTR)
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

/// Ctrl+C throws away what is typed instead of quitting, matching every other
/// shell prompt. Quitting is still one more Ctrl+C away on an empty line, and the
/// discarded text stays in the kill ring, so a mistaken press is recoverable with
/// Ctrl+Y.
struct AbandonLine;

impl ConditionalEventHandler for AbandonLine {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        context: &EventContext,
    ) -> Option<Cmd> {
        if context.line().is_empty() {
            // Nothing to discard, so let the default interrupt exit.
            return None;
        }
        Some(Cmd::Kill(Movement::WholeBuffer))
    }
}

struct TabHandler {
    multi: Arc<AtomicBool>,
    palette: Arc<PaletteState>,
    cwd: PathBuf,
}

impl ConditionalEventHandler for TabHandler {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        context: &EventContext,
    ) -> Option<Cmd> {
        if let Some(token) = palette_token(context.line(), context.pos()) {
            if let Some(candidate) = self.palette.highlighted(&token, &self.cwd) {
                // Accepting re-anchors the filter, so completing to a directory
                // lists that directory on the next keystroke.
                self.palette
                    .request_completion(token.text, &candidate.completion, true);
            }
            return Some(Cmd::Complete);
        }
        self.multi.fetch_xor(true, Ordering::SeqCst);
        Some(Cmd::Repaint)
    }
}

/// Rustyline maps one key press to one command, so Enter cannot both rewrite the
/// buffer and submit it. When the input still holds the prefix the user typed,
/// Enter completes it to the highlighted entry instead of submitting, so the
/// transcript never records a line that differs from what ran.
struct PaletteSubmit {
    palette: Arc<PaletteState>,
    cwd: PathBuf,
}

impl ConditionalEventHandler for PaletteSubmit {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        context: &EventContext,
    ) -> Option<Cmd> {
        let token = palette_token(context.line(), context.pos())?;
        let candidate = self.palette.highlighted(&token, &self.cwd)?;
        if candidate.completion == token.text {
            return None;
        }
        self.palette
            .request_completion(token.text, &candidate.completion, true);
        Some(Cmd::Complete)
    }
}

struct PaletteNavigation {
    palette: Arc<PaletteState>,
    direction: isize,
    cwd: PathBuf,
}

impl ConditionalEventHandler for PaletteNavigation {
    fn handle(
        &self,
        _event: &Event,
        _repeat: RepeatCount,
        _positive: bool,
        context: &EventContext,
    ) -> Option<Cmd> {
        let token = palette_token(context.line(), context.pos())?;
        let candidate = self.palette.navigate(&token, &self.cwd, self.direction)?;
        // Arrow navigation keeps the typed filter so siblings stay reachable.
        self.palette
            .request_completion(token.text, &candidate.completion, false);
        Some(Cmd::Complete)
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
    pub fn with_reasoning_toggle(value: &str, cwd: impl Into<PathBuf>) -> io::Result<Self> {
        let cwd = cwd.into();
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
            // Circular completion runs its own key loop, which swallows the next
            // arrow press and restores the pre-completion buffer. List applies a
            // single candidate and returns immediately.
            .completion_type(rustyline::CompletionType::List)
            .build();
        let mut editor = Editor::<AgentHelper, DefaultHistory>::with_config(editor_config)
            .map_err(io::Error::other)?;
        editor.set_helper(Some(AgentHelper {
            multi: multi.clone(),
            palette: palette.clone(),
            cwd: cwd.clone(),
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
            KeyEvent::ctrl('c'),
            EventHandler::Conditional(Box::new(AbandonLine)),
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
                cwd: cwd.clone(),
            })),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Enter, Modifiers::NONE),
            EventHandler::Conditional(Box::new(PaletteSubmit {
                palette: palette.clone(),
                cwd: cwd.clone(),
            })),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Up, Modifiers::NONE),
            EventHandler::Conditional(Box::new(PaletteNavigation {
                palette: palette.clone(),
                direction: -1,
                cwd: cwd.clone(),
            })),
        );
        editor.bind_sequence(
            KeyEvent(KeyCode::Down, Modifiers::NONE),
            EventHandler::Conditional(Box::new(PaletteNavigation {
                palette: palette.clone(),
                direction: 1,
                cwd,
            })),
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
        let default = default.min(choices.len() - 1);
        // A selector reads keys through select(2), which returns EINTR whenever a
        // signal arrives — and a window resize sends SIGWINCH, for which a handler
        // is installed for the rest of the process once a turn has run. Losing the
        // menu because the window changed size would be absurd, so the menu is
        // redrawn and the read retried. The old frame is erased first so the
        // resize does not leave a second menu behind.
        for _ in 0..RESIZE_RETRIES {
            match Select::new()
                .with_prompt(prompt)
                .items(choices)
                .default(default)
                .interact_opt()
            {
                Ok(choice) => return Ok(choice),
                Err(dialoguer::Error::IO(error)) if is_signal_interruption(&error) => {
                    let term = dialoguer::console::Term::stderr();
                    let drawn = choices.len() + 1;
                    let height = usize::from(term.size().0).saturating_sub(1);
                    term.clear_last_lines(drawn.min(height.max(1)))?;
                }
                Err(error) => return Err(io::Error::other(error)),
            }
        }
        Err(io::Error::other(
            "the terminal kept interrupting the selection",
        ))
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
        let labels = checkpoints
            .iter()
            .map(|(_, label)| label.clone())
            .collect::<Vec<_>>();
        let Some(index) = self.select_option("Rewind to", &labels, 0)? else {
            return Ok(None);
        };
        Ok(Some(checkpoints[index].0.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_tokens_only_start_a_line_while_mentions_can_appear_anywhere() {
        let slash = palette_token("/mo", 3).expect("slash token");
        assert_eq!(slash.kind, PaletteKind::Slash);
        assert_eq!(slash.text, "/mo");
        assert_eq!(slash.start, 0);

        let mention = palette_token("fix @src/pa", 11).expect("path token");
        assert_eq!(mention.kind, PaletteKind::Path);
        assert_eq!(mention.text, "@src/pa");
        assert_eq!(mention.start, 4);

        // A slash that is not the first token is a path separator, not a command.
        assert!(palette_token("look at /etc", 12).is_none());
        assert!(palette_token("plain words", 11).is_none());
        // Completion only applies at the end of the line.
        assert!(palette_token("@src", 2).is_none());
    }

    #[test]
    fn path_candidates_put_directories_first_and_hide_noise() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("srcfile.rs"), "").unwrap();
        std::fs::write(root.join(".srchidden"), "").unwrap();
        std::fs::write(root.join("other.rs"), "").unwrap();

        let rows = path_candidates("s", root);
        let completions = rows
            .iter()
            .map(|candidate| candidate.completion.as_str())
            .collect::<Vec<_>>();
        assert_eq!(completions, vec!["@src/", "@srcfile.rs"], "{completions:?}");

        // Ignored directories stay out even when they match the prefix.
        assert!(path_candidates("t", root).is_empty());
        assert!(
            path_candidates("", root)
                .iter()
                .all(|c| c.completion != "@.git/")
        );
        // Hidden entries appear once a dot is typed.
        let hidden = path_candidates(".src", root);
        assert_eq!(hidden.len(), 1, "{:?}", hidden[0].completion);
        assert_eq!(hidden[0].completion, "@.srchidden");
    }

    #[test]
    fn mentions_of_paths_with_spaces_survive_completion_and_resolution() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::write(root.join("name with space.txt"), "").unwrap();

        let candidate = path_candidates("name", root).remove(0);
        assert_eq!(candidate.completion, r"@name\ with\ space.txt");
        // The escaped mention stays one token, both for the palette and for the
        // resolver, so completion cannot produce a mention that silently fails.
        let line = format!("review {}", candidate.completion);
        let token = palette_token(&line, line.len()).expect("token");
        assert_eq!(token.text, r"@name\ with\ space.txt");
        assert_eq!(mention_paths(&line), vec!["name with space.txt".to_owned()]);
        assert_eq!(
            path_candidates(&unescape_mention(&token.text[1..]), root).len(),
            1
        );
    }

    #[test]
    fn mentions_are_split_off_trailing_punctuation_and_other_words() {
        assert_eq!(
            mention_paths("compare @src/a.rs and @src/b.rs, please"),
            vec!["src/a.rs".to_owned(), "src/b.rs".to_owned()]
        );
        assert!(mention_paths("no mentions here").is_empty());
        assert!(mention_paths("@").is_empty());
    }

    #[test]
    fn descending_into_a_directory_lists_its_contents() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/parser.rs"), "").unwrap();
        std::fs::write(root.join("src/main.rs"), "").unwrap();

        let rows = path_candidates("src/", root);
        let completions = rows
            .iter()
            .map(|candidate| candidate.completion.as_str())
            .collect::<Vec<_>>();
        assert_eq!(completions, vec!["@src/main.rs", "@src/parser.rs"]);
    }
}
