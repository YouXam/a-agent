use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::cursor::{MoveUp, RestorePosition, SavePosition};
use crossterm::execute;
use crossterm::style::{Attribute, Color, Stylize};
use crossterm::terminal::{Clear, ClearType};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde_json::Value;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::model::{ContentBlock, ConversationItem, Role, StreamEvent, ToolResult};
use crate::provider::EventSink;

const GENERATION_ID: &str = "__a_generation";
const GENERATION_CONTENT_ID: &str = "__a_generation_content";

/// One file in a rewind plan: what will happen to it, and why.
#[derive(Debug, Clone)]
pub struct RevertLine {
    /// Imperative verb, so the list reads as the plan it is rather than as
    /// history: `delete`, `restore`, `recreate`, `keep`.
    pub action: String,
    pub path: String,
    pub detail: String,
    pub blocked: bool,
}

impl RevertLine {
    fn color(&self) -> Color {
        if self.blocked {
            return Color::DarkGrey;
        }
        match self.action.as_str() {
            "delete" => Color::DarkRed,
            "recreate" => Color::DarkGreen,
            _ => Color::DarkYellow,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderLimits {
    pub tool_input_max_bytes: usize,
    pub tool_output_max_bytes: usize,
    pub tool_output_max_lines: usize,
    pub tool_live_output_lines: usize,
    pub patch_diff_max_lines: usize,
}

impl Default for RenderLimits {
    fn default() -> Self {
        Self {
            tool_input_max_bytes: 2048,
            tool_output_max_bytes: 8192,
            tool_output_max_lines: 16,
            tool_live_output_lines: 6,
            patch_diff_max_lines: 24,
        }
    }
}

#[derive(Clone)]
pub struct InlineRenderer {
    inner: Arc<Mutex<State>>,
}

struct State {
    writer: Box<dyn Write + Send>,
    color: bool,
    limits: RenderLimits,
    reasoning_visible: bool,
    reasoning_announced: bool,
    reasoning_buffer: String,
    reasoning_pending: String,
    assistant_buffer: String,
    reasoning_at_line_start: bool,
    assistant_at_line_start: bool,
    tools: HashMap<String, ToolDisplay>,
    live: Option<LiveTools>,
}

struct ToolDisplay {
    name: String,
    arguments: BoundedInput,
    output: TailBuffer,
}

struct BoundedInput {
    text: String,
    max_bytes: usize,
    truncated: bool,
}

struct TailBuffer {
    bytes: Vec<u8>,
    max_bytes: usize,
    total_bytes: usize,
    total_lines: usize,
}

struct LimitedText {
    lines: Vec<String>,
    truncated: bool,
}

struct LiveTools {
    multi: MultiProgress,
    entries: HashMap<String, ProgressBar>,
    max_lines: usize,
    visible_lines: usize,
    terminal_width: usize,
    reserved: bool,
    reserve_stdout_rows: bool,
}

impl InlineRenderer {
    pub fn stdout(show_reasoning: bool) -> io::Result<Self> {
        Self::stdout_with_limits(show_reasoning, RenderLimits::default())
    }

    pub fn stdout_with_limits(show_reasoning: bool, limits: RenderLimits) -> io::Result<Self> {
        let interactive = io::stdout().is_terminal();
        let renderer = Self::new_with_limits(
            io::stdout(),
            show_reasoning,
            interactive && std::env::var_os("NO_COLOR").is_none(),
            limits,
        );
        if interactive {
            renderer.with_state(|state| {
                state.live = Some(LiveTools::new(state.limits.tool_live_output_lines));
                Ok(())
            })?;
        }
        Ok(renderer)
    }

    pub fn new(writer: impl Write + Send + 'static, show_reasoning: bool, color: bool) -> Self {
        Self::new_with_limits(writer, show_reasoning, color, RenderLimits::default())
    }

    pub fn new_with_limits(
        writer: impl Write + Send + 'static,
        show_reasoning: bool,
        color: bool,
        limits: RenderLimits,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(State {
                writer: Box::new(writer),
                color,
                limits,
                reasoning_visible: show_reasoning,
                reasoning_announced: false,
                reasoning_buffer: String::new(),
                reasoning_pending: String::new(),
                assistant_buffer: String::new(),
                reasoning_at_line_start: true,
                assistant_at_line_start: true,
                tools: HashMap::new(),
                live: None,
            })),
        }
    }

    pub fn begin_turn(&self) -> io::Result<()> {
        self.with_state(|state| {
            state.finish_open_lines()?;
            state.reasoning_announced = false;
            state.reasoning_buffer.clear();
            state.reasoning_pending.clear();
            state.assistant_buffer.clear();
            state.tools.clear();
            if let Some(live) = &mut state.live {
                live.clear();
            }
            Ok(())
        })
    }

    pub fn render_user(&self, message: &str) -> io::Result<()> {
        self.begin_turn()?;
        self.with_state(|state| {
            write_prefixed_block(
                &mut state.writer,
                state.color,
                "› ",
                message,
                Color::DarkCyan,
                true,
                Color::Reset,
            )?;
            state.writer.flush()
        })
    }

    /// Lists what a rewind would do to each file, one line per path, using the
    /// same colors as a patch block so the plan reads at a glance.
    pub fn render_revert_plan(&self, lines: &[RevertLine]) -> io::Result<()> {
        let width = lines
            .iter()
            .map(|line| line.action.len())
            .max()
            .unwrap_or_default();
        self.with_state(|state| {
            state.flush_generation_pending()?;
            state.finish_generation();
            state.finish_open_lines()?;
            for line in lines {
                write_styled(&mut state.writer, state.color, "    ", Color::Reset, false)?;
                write_styled(
                    &mut state.writer,
                    state.color,
                    &line.action,
                    line.color(),
                    true,
                )?;
                write_styled(
                    &mut state.writer,
                    state.color,
                    &" ".repeat(width - line.action.len() + 1),
                    Color::Reset,
                    false,
                )?;
                write_styled(
                    &mut state.writer,
                    state.color,
                    &line.path,
                    if line.blocked {
                        Color::DarkGrey
                    } else {
                        Color::Reset
                    },
                    false,
                )?;
                write_styled(
                    &mut state.writer,
                    state.color,
                    &format!("  {}\n", line.detail),
                    Color::DarkGrey,
                    false,
                )?;
            }
            state.writer.flush()
        })
    }

    pub fn render_status(&self, message: &str) -> io::Result<()> {
        self.with_state(|state| {
            state.flush_generation_pending()?;
            state.finish_generation();
            state.finish_open_lines()?;
            write_styled(
                &mut state.writer,
                state.color,
                &format!("· {message}\n"),
                Color::DarkGrey,
                false,
            )?;
            state.writer.flush()
        })
    }

    /// Shows a labelled spinner in the transient region while something is being
    /// awaited. Reuses the generation spinner, so nothing is drawn when stdout is
    /// not a terminal and nothing reaches the scrollback either way.
    pub fn begin_transient(&self, label: &str) -> io::Result<()> {
        self.with_state(|state| {
            state.start_generation()?;
            state.update_generation_message(label);
            Ok(())
        })
    }

    pub fn end_transient(&self) -> io::Result<()> {
        self.with_state(|state| {
            state.finish_generation();
            Ok(())
        })
    }

    pub fn render_resumed_history(&self, items: &[ConversationItem]) -> io::Result<()> {
        self.begin_turn()?;
        self.with_state(|state| {
            write_styled(
                &mut state.writer,
                state.color,
                "──────── Resumed conversation ────────\n",
                Color::DarkGrey,
                true,
            )?;
            state.writer.flush()
        })?;
        for item in items {
            match item.role {
                Role::User => {
                    let text = item
                        .blocks
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(text) => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let text = without_request_scaffolding(&text);
                    if !text.is_empty() {
                        self.render_user(text)?;
                    }
                }
                Role::Assistant => {
                    self.begin_turn()?;
                    for block in &item.blocks {
                        match block {
                            ContentBlock::Reasoning(delta) => {
                                self.render_event(StreamEvent::ReasoningDelta {
                                    delta: delta.clone(),
                                })?
                            }
                            ContentBlock::Text(delta) => {
                                self.render_event(StreamEvent::TextDelta {
                                    delta: delta.clone(),
                                })?
                            }
                            ContentBlock::ToolCall(call) => {
                                self.render_event(StreamEvent::ToolCallStart {
                                    id: call.id.clone(),
                                    name: call.name.clone(),
                                })?;
                                self.render_event(StreamEvent::ToolCallArgsDelta {
                                    id: call.id.clone(),
                                    delta: call.arguments.clone(),
                                })?;
                                self.render_event(StreamEvent::ToolCallEnd {
                                    id: call.id.clone(),
                                })?;
                            }
                            ContentBlock::ToolResult(_) => {}
                        }
                    }
                    self.render_event(StreamEvent::Done)?;
                }
                Role::Tool => {
                    for block in &item.blocks {
                        if let ContentBlock::ToolResult(result) = block {
                            self.render_event(StreamEvent::ToolExecutionEnd {
                                id: result.call_id.clone(),
                                result: result.clone(),
                            })?;
                        }
                    }
                }
                Role::System => {}
            }
        }
        Ok(())
    }

    pub fn event_sink(&self) -> EventSink {
        let renderer = self.clone();
        EventSink::new(move |event| {
            let _ = renderer.render_event(event);
        })
    }

    pub fn toggle_reasoning(&self) -> io::Result<bool> {
        self.with_state(|state| {
            state.finish_open_lines()?;
            state.reasoning_visible = !state.reasoning_visible;
            if state.reasoning_visible && !state.reasoning_buffer.is_empty() {
                write_styled(
                    &mut state.writer,
                    state.color,
                    "▾ Reasoning\n",
                    Color::DarkGrey,
                    true,
                )?;
                let buffered = state.reasoning_buffer.clone();
                append_stream(
                    &mut state.writer,
                    state.color,
                    "  ",
                    &buffered,
                    Color::DarkGrey,
                    Color::DarkGrey,
                    &mut state.reasoning_at_line_start,
                )?;
                state.finish_open_lines()?;
            }
            state.writer.flush()?;
            Ok(state.reasoning_visible)
        })
    }

    fn render_event(&self, event: StreamEvent) -> io::Result<()> {
        self.with_state(|state| {
            match event {
                StreamEvent::GenerationStart => state.start_generation()?,
                StreamEvent::ReasoningDelta { delta } => state.reasoning_delta(&delta)?,
                StreamEvent::TextDelta { delta } => state.assistant_delta(&delta)?,
                StreamEvent::ToolCallStart { id, name } => {
                    state.flush_generation_pending()?;
                    state.update_generation_message(&format!("generating  {name}"));
                    state.tools.insert(
                        id,
                        ToolDisplay::new(
                            name,
                            state.limits.tool_input_max_bytes,
                            state.limits.tool_output_max_bytes,
                        ),
                    );
                }
                StreamEvent::ToolCallArgsDelta { id, delta } => {
                    if let Some(tool) = state.tools.get_mut(&id) {
                        tool.arguments.push(&delta);
                    }
                }
                StreamEvent::ToolCallEnd { .. } => {}
                StreamEvent::ToolExecutionStart { id } => state.start_live_tool(&id),
                StreamEvent::ToolExecutionOutput { id, delta } => {
                    if let Some(tool) = state.tools.get_mut(&id) {
                        tool.output.push(delta.as_bytes());
                    }
                    state.update_live_tool(&id);
                }
                StreamEvent::ToolExecutionEnd { id, result } => {
                    state.render_completed_tool(&id, &result)?;
                }
                StreamEvent::Error { message } => {
                    state.flush_generation_pending()?;
                    state.finish_generation();
                    state.finish_open_lines()?;
                    write_styled(
                        &mut state.writer,
                        state.color,
                        &format!("× {message}\n"),
                        Color::DarkRed,
                        true,
                    )?;
                }
                StreamEvent::Done => {
                    state.flush_generation_pending()?;
                    state.finish_generation();
                    state.finish_open_lines()?;
                    state.reasoning_announced = false;
                    state.reasoning_buffer.clear();
                    state.reasoning_pending.clear();
                    state.assistant_buffer.clear();
                }
                StreamEvent::Usage(_) => {}
            }
            state.writer.flush()
        })
    }

    fn with_state<T>(&self, action: impl FnOnce(&mut State) -> io::Result<T>) -> io::Result<T> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("renderer lock poisoned"))?;
        action(&mut state)
    }
}

impl State {
    fn start_generation(&mut self) -> io::Result<()> {
        self.finish_open_lines()?;
        if let Some(live) = &mut self.live {
            live.reserve_generation();
            live.start(GENERATION_ID, "thinking".into());
        }
        Ok(())
    }

    fn finish_generation(&mut self) {
        if let Some(live) = &mut self.live
            && live.entries.contains_key(GENERATION_ID)
        {
            live.finish(GENERATION_CONTENT_ID);
            live.finish(GENERATION_ID);
            live.restore_for_commit();
        }
    }

    fn generation_active(&self) -> bool {
        self.live
            .as_ref()
            .is_some_and(|live| live.entries.contains_key(GENERATION_ID))
    }

    fn update_generation_message(&mut self, message: &str) {
        if let Some(live) = &mut self.live
            && let Some(progress) = live.entries.get(GENERATION_ID)
        {
            progress.set_message(message.to_owned());
            progress.tick();
        }
    }

    fn reasoning_delta(&mut self, delta: &str) -> io::Result<()> {
        self.reasoning_buffer.push_str(delta);
        if self.generation_active() {
            self.announce_generation_reasoning()?;
            self.update_generation_message("thinking");
            if self.reasoning_visible {
                self.reasoning_pending.push_str(delta);
                let complete = take_complete_lines(&mut self.reasoning_pending);
                let pending = self.reasoning_pending.clone();
                self.set_generation_partial("  ", &pending, Color::DarkGrey, Color::DarkGrey);
                if !complete.is_empty() {
                    self.commit_generation_text("  ", &complete, Color::DarkGrey, Color::DarkGrey)?;
                }
            }
            return Ok(());
        }
        if !self.reasoning_announced {
            self.finish_open_lines()?;
            write_styled(
                &mut self.writer,
                self.color,
                if self.reasoning_visible {
                    "▾ Reasoning\n"
                } else {
                    "▸ Reasoning\n"
                },
                Color::DarkGrey,
                true,
            )?;
            self.reasoning_announced = true;
        }
        if self.reasoning_visible {
            append_stream(
                &mut self.writer,
                self.color,
                "  ",
                delta,
                Color::DarkGrey,
                Color::DarkGrey,
                &mut self.reasoning_at_line_start,
            )?;
        }
        Ok(())
    }

    fn assistant_delta(&mut self, delta: &str) -> io::Result<()> {
        if self.generation_active() {
            self.flush_generation_reasoning()?;
            self.update_generation_message("generating");
            self.assistant_buffer.push_str(delta);
            let complete = take_complete_lines(&mut self.assistant_buffer);
            let pending = self.assistant_buffer.clone();
            self.set_generation_partial("│ ", &pending, Color::DarkGreen, Color::Reset);
            if !complete.is_empty() {
                self.commit_generation_text("│ ", &complete, Color::DarkGreen, Color::Reset)?;
            }
            return Ok(());
        }
        if !self.reasoning_at_line_start {
            writeln!(self.writer)?;
            self.reasoning_at_line_start = true;
        }
        append_stream(
            &mut self.writer,
            self.color,
            "│ ",
            delta,
            Color::DarkGreen,
            Color::Reset,
            &mut self.assistant_at_line_start,
        )
    }

    fn announce_generation_reasoning(&mut self) -> io::Result<()> {
        if self.reasoning_announced {
            return Ok(());
        }
        self.reasoning_announced = true;
        let active = self.generation_active();
        if active && let Some(live) = &mut self.live {
            live.restore_for_commit();
        }
        write_styled(
            &mut self.writer,
            self.color,
            if self.reasoning_visible {
                "▾ Reasoning\n"
            } else {
                "▸ Reasoning\n"
            },
            Color::DarkGrey,
            true,
        )?;
        if active && let Some(live) = &mut self.live {
            live.resume_after_commit();
        }
        Ok(())
    }

    fn set_generation_partial(
        &mut self,
        prefix: &str,
        pending: &str,
        prefix_color: Color,
        content_color: Color,
    ) {
        let Some(live) = &mut self.live else {
            return;
        };
        if pending.is_empty() {
            live.set_generation_content(None);
            return;
        }
        let content = clean_terminal_line(pending);
        let available = live
            .terminal_width
            .saturating_sub(UnicodeWidthStr::width(prefix));
        let content = truncate_display_width(&content, available);
        let message = if self.color {
            format!(
                "{}{}",
                prefix.with(prefix_color),
                content.with(content_color)
            )
        } else {
            format!("{prefix}{content}")
        };
        live.set_generation_content(Some(message));
    }

    fn commit_generation_text(
        &mut self,
        prefix: &str,
        text: &str,
        prefix_color: Color,
        content_color: Color,
    ) -> io::Result<()> {
        let active = self.generation_active();
        if active && let Some(live) = &mut self.live {
            live.restore_for_commit();
        }
        let mut at_line_start = true;
        append_stream(
            &mut self.writer,
            self.color,
            prefix,
            text,
            prefix_color,
            content_color,
            &mut at_line_start,
        )?;
        if !at_line_start {
            writeln!(self.writer)?;
        }
        if active && let Some(live) = &mut self.live {
            live.resume_after_commit();
        }
        Ok(())
    }

    fn flush_generation_reasoning(&mut self) -> io::Result<()> {
        let pending = std::mem::take(&mut self.reasoning_pending);
        if pending.is_empty() {
            return Ok(());
        }
        if let Some(live) = &mut self.live {
            live.set_generation_content(None);
        }
        self.commit_generation_text("  ", &pending, Color::DarkGrey, Color::DarkGrey)
    }

    fn flush_generation_pending(&mut self) -> io::Result<()> {
        if !self.generation_active() {
            return Ok(());
        }
        self.flush_generation_reasoning()?;
        let pending = std::mem::take(&mut self.assistant_buffer);
        if pending.is_empty() {
            return Ok(());
        }
        if let Some(live) = &mut self.live {
            live.set_generation_content(None);
        }
        self.commit_generation_text("│ ", &pending, Color::DarkGreen, Color::Reset)
    }

    fn finish_open_lines(&mut self) -> io::Result<()> {
        for at_line_start in [
            &mut self.reasoning_at_line_start,
            &mut self.assistant_at_line_start,
        ] {
            if !*at_line_start {
                writeln!(self.writer)?;
                *at_line_start = true;
            }
        }
        Ok(())
    }

    fn start_live_tool(&mut self, id: &str) {
        let Some(tool) = self.tools.get(id) else {
            return;
        };
        if let Some(live) = &mut self.live {
            live.reserve(self.tools.len());
            let message =
                live_tool_message(tool, live.visible_lines, live.terminal_width, self.color);
            live.start(id, message);
        }
    }

    fn update_live_tool(&mut self, id: &str) {
        let Some(tool) = self.tools.get(id) else {
            return;
        };
        if let Some(live) = &mut self.live {
            let message =
                live_tool_message(tool, live.visible_lines, live.terminal_width, self.color);
            live.update(id, message);
        }
    }

    fn finish_live_tool(&mut self, id: &str) {
        if let Some(live) = &mut self.live {
            live.finish(id);
        }
    }

    fn render_completed_tool(&mut self, id: &str, result: &ToolResult) -> io::Result<()> {
        let has_live = self.live.is_some();
        self.finish_live_tool(id);
        if has_live && let Some(live) = &mut self.live {
            live.restore_for_commit();
        }
        self.render_tool_result(id, result)?;
        if has_live && let Some(live) = &mut self.live {
            live.resume_after_commit();
        }
        Ok(())
    }

    fn render_tool_input(
        &mut self,
        id: &str,
        symbol: &str,
        color: Color,
        summary: &str,
    ) -> io::Result<()> {
        let Some(tool) = self.tools.get_mut(id) else {
            return Ok(());
        };
        let name = tool.name.clone();
        let raw_arguments = tool.arguments.text.clone();
        let truncated = tool.arguments.truncated;
        self.finish_open_lines()?;
        match name.as_str() {
            "read" => {
                let detail = read_detail(&raw_arguments);
                write_tool_completion(
                    &mut self.writer,
                    self.color,
                    symbol,
                    color,
                    &name,
                    &format!("{detail} · {summary}"),
                )
            }
            "apply_patch" => {
                write_tool_completion(&mut self.writer, self.color, symbol, color, &name, summary)?;
                render_patch_operations(
                    &mut self.writer,
                    self.color,
                    &raw_arguments,
                    self.limits.tool_input_max_bytes,
                    truncated,
                    self.limits.patch_diff_max_lines,
                )
            }
            "bash" => {
                let summary = match bash_timeout_label(&raw_arguments) {
                    Some(timeout) => format!("{summary} · {timeout}"),
                    None => summary.to_owned(),
                };
                write_tool_completion(
                    &mut self.writer,
                    self.color,
                    symbol,
                    color,
                    &name,
                    &summary,
                )?;
                let input = limited_text(
                    &format_tool_input(&name, &raw_arguments),
                    self.limits.tool_input_max_bytes,
                    8,
                    false,
                );
                render_bash_command(&mut self.writer, self.color, &input, truncated)
            }
            _ => {
                write_tool_header(&mut self.writer, self.color, &name, None)?;
                let input = limited_text(
                    &format_tool_input(&name, &raw_arguments),
                    self.limits.tool_input_max_bytes,
                    8,
                    false,
                );
                render_section(&mut self.writer, self.color, "input", &input, truncated)
            }
        }
    }

    fn render_tool_result(&mut self, id: &str, result: &ToolResult) -> io::Result<()> {
        let name = self
            .tools
            .get(id)
            .map(|tool| tool.name.clone())
            .unwrap_or_else(|| "tool".to_owned());
        let exit_code = parse_exit_code(&result.output);
        let bash_interrupted = name == "bash"
            && (result.output.contains("[bash cancelled]")
                || result.output.contains("[bash timed out after "));
        let failed = result.is_error || bash_interrupted || exit_code.is_some_and(|code| code != 0);
        let symbol = if failed { "×" } else { "✓" };
        let color = if failed {
            Color::DarkRed
        } else {
            Color::DarkGreen
        };
        let summary = tool_summary(&name, result, exit_code);
        self.render_tool_input(id, symbol, color, &summary)?;
        let output = match self.tools.get(id) {
            Some(tool) => {
                if tool.output.total_bytes > 0 {
                    tool.output.limited(self.limits.tool_output_max_lines)
                } else if tool.name == "bash" {
                    let output = bash_visible_output(&result.output);
                    limited_text(
                        &output,
                        self.limits.tool_output_max_bytes,
                        self.limits.tool_output_max_lines,
                        true,
                    )
                } else {
                    limited_text(
                        &result.output,
                        self.limits.tool_output_max_bytes,
                        self.limits.tool_output_max_lines,
                        tool.name == "bash",
                    )
                }
            }
            None => limited_text(
                &result.output,
                self.limits.tool_output_max_bytes,
                self.limits.tool_output_max_lines,
                false,
            ),
        };
        match name.as_str() {
            "bash" => render_bash_output(&mut self.writer, self.color, &output)?,
            "read" => render_direct_output(&mut self.writer, self.color, &output, Color::Reset)?,
            "apply_patch" if result.is_error => {
                render_direct_output(&mut self.writer, self.color, &output, Color::DarkRed)?
            }
            "apply_patch" => {}
            _ => render_section(
                &mut self.writer,
                self.color,
                "output",
                &output,
                output.truncated,
            )?,
        }
        if !matches!(name.as_str(), "bash" | "read" | "apply_patch") {
            write_tool_completion(&mut self.writer, self.color, symbol, color, &name, &summary)?;
        }
        self.tools.remove(id);
        Ok(())
    }
}

impl ToolDisplay {
    fn new(name: String, input_max_bytes: usize, output_max_bytes: usize) -> Self {
        Self {
            name,
            arguments: BoundedInput::new(input_max_bytes.max(64 * 1024)),
            output: TailBuffer::new(output_max_bytes),
        }
    }
}

impl LiveTools {
    fn new(max_lines: usize) -> Self {
        let terminal_width = crossterm::terminal::size()
            .map(|(width, _)| usize::from(width))
            .unwrap_or(80);
        let mut live = Self::with_draw_target(
            ProgressDrawTarget::stdout_with_hz(20),
            max_lines,
            terminal_width,
        );
        live.reserve_stdout_rows = true;
        live
    }

    fn with_draw_target(
        draw_target: ProgressDrawTarget,
        max_lines: usize,
        terminal_width: usize,
    ) -> Self {
        Self {
            multi: MultiProgress::with_draw_target(draw_target),
            entries: HashMap::new(),
            max_lines,
            visible_lines: max_lines,
            terminal_width: terminal_width.max(1),
            reserved: false,
            reserve_stdout_rows: false,
        }
    }

    fn reserve(&mut self, tool_count: usize) {
        if self.reserved {
            return;
        }
        let (terminal_width, terminal_height) =
            crossterm::terminal::size().unwrap_or((self.terminal_width as u16, 24));
        self.terminal_width = usize::from(terminal_width).max(1);
        let available_rows = usize::from(terminal_height.saturating_sub(1));
        let rows_per_tool = available_rows / tool_count.max(1);
        self.visible_lines = self.max_lines.min(rows_per_tool.saturating_sub(2));
        let desired = tool_count.saturating_mul(self.visible_lines.saturating_add(2));
        let height = desired.min(available_rows) as u16;
        self.reserve_rows(height);
    }

    fn reserve_generation(&mut self) {
        self.reserve_rows(1);
    }

    fn reserve_rows(&mut self, height: u16) {
        if self.reserved {
            return;
        }
        if self.reserve_stdout_rows && height > 0 {
            let mut stdout = io::stdout();
            for _ in 0..height {
                let _ = writeln!(stdout);
            }
            let _ = execute!(stdout, MoveUp(height));
            let _ = execute!(stdout, SavePosition);
            let _ = stdout.flush();
        }
        self.reserved = true;
    }

    fn start(&mut self, id: &str, message: String) {
        if self.entries.contains_key(id) {
            self.update(id, message);
            return;
        }
        let progress = self.multi.add(ProgressBar::new_spinner());
        progress.set_style(
            ProgressStyle::with_template("{spinner:.yellow} {msg}")
                .expect("static progress template is valid")
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
        );
        progress.set_message(message);
        progress.enable_steady_tick(Duration::from_millis(80));
        progress.force_draw();
        self.entries.insert(id.to_owned(), progress);
    }

    fn update(&mut self, id: &str, message: String) {
        if let Some(progress) = self.entries.get(id) {
            progress.set_message(message);
            progress.tick();
        }
    }

    fn set_generation_content(&mut self, message: Option<String>) {
        match message {
            Some(message) => {
                if self.entries.contains_key(GENERATION_CONTENT_ID) {
                    self.update(GENERATION_CONTENT_ID, message);
                    return;
                }
                self.restore_for_commit();
                let progress = ProgressBar::new_spinner();
                progress.set_style(
                    ProgressStyle::with_template("{msg}")
                        .expect("static progress template is valid"),
                );
                progress.set_message(message);
                let progress = if let Some(spinner) = self.entries.get(GENERATION_ID) {
                    self.multi.insert_before(spinner, progress)
                } else {
                    self.multi.add(progress)
                };
                self.entries
                    .insert(GENERATION_CONTENT_ID.to_owned(), progress);
                self.resume_after_commit();
            }
            None if self.entries.contains_key(GENERATION_CONTENT_ID) => {
                self.finish(GENERATION_CONTENT_ID);
                self.restore_for_commit();
                self.resume_after_commit();
            }
            None => {}
        }
    }

    fn finish(&mut self, id: &str) {
        if let Some(progress) = self.entries.remove(id) {
            progress.disable_steady_tick();
            self.multi.remove(&progress);
        }
    }

    fn restore_for_commit(&mut self) {
        let _ = self.multi.clear();
        if self.reserved {
            if self.reserve_stdout_rows {
                let mut stdout = io::stdout();
                let _ = execute!(stdout, RestorePosition, Clear(ClearType::FromCursorDown));
                let _ = stdout.flush();
            }
            self.reserved = false;
        }
    }

    fn resume_after_commit(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        if self.entries.contains_key(GENERATION_ID) {
            let rows = 1 + usize::from(self.entries.contains_key(GENERATION_CONTENT_ID));
            self.reserve_rows(rows as u16);
        } else {
            self.reserve(self.entries.len());
        }
        for progress in self.entries.values() {
            progress.force_draw();
        }
    }

    fn clear(&mut self) {
        for (_, progress) in self.entries.drain() {
            progress.disable_steady_tick();
            self.multi.remove(&progress);
        }
        self.restore_for_commit();
    }
}

impl Drop for LiveTools {
    fn drop(&mut self) {
        self.clear();
    }
}

impl BoundedInput {
    fn new(max_bytes: usize) -> Self {
        Self {
            text: String::new(),
            max_bytes,
            truncated: false,
        }
    }

    fn push(&mut self, delta: &str) {
        let remaining = self.max_bytes.saturating_sub(self.text.len());
        let mut end = remaining.min(delta.len());
        while end > 0 && !delta.is_char_boundary(end) {
            end -= 1;
        }
        self.text.push_str(&delta[..end]);
        self.truncated |= end < delta.len();
    }
}

impl TailBuffer {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
            total_bytes: 0,
            total_lines: 0,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.total_bytes += bytes.len();
        self.total_lines += bytes.iter().filter(|byte| **byte == b'\n').count();
        self.bytes.extend_from_slice(bytes);
        if self.bytes.len() > self.max_bytes {
            self.bytes.drain(..self.bytes.len() - self.max_bytes);
        }
    }

    fn limited(&self, max_lines: usize) -> LimitedText {
        let text = String::from_utf8_lossy(&self.bytes);
        let mut lines = text.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
        if lines.len() > max_lines {
            lines = lines.split_off(lines.len() - max_lines);
        }
        LimitedText {
            lines,
            truncated: self.total_bytes > self.bytes.len() || self.total_lines > max_lines,
        }
    }
}

fn format_tool_input(name: &str, raw: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return raw.to_owned();
    };
    match name {
        "read" => {
            let mut lines = Vec::new();
            if let Some(path) = value.get("path").and_then(Value::as_str) {
                lines.push(format!("path: {path}"));
            }
            if let Some(offset) = value.get("offset").and_then(Value::as_u64) {
                lines.push(format!("offset: {offset}"));
            }
            if let Some(limit) = value.get("limit").and_then(Value::as_u64) {
                lines.push(format!("limit: {limit}"));
            }
            lines.join("\n")
        }
        "bash" => value
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or(raw)
            .to_owned(),
        "apply_patch" => value
            .get("patch")
            .and_then(Value::as_str)
            .unwrap_or(raw)
            .to_owned(),
        _ => serde_json::to_string_pretty(&value).unwrap_or_else(|_| raw.to_owned()),
    }
}

/// The timeout a bash call asked for, when it raised its own limit. A command
/// that sits there for minutes should not look like a hang.
fn bash_timeout_label(raw: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(raw).ok()?;
    let seconds = value.get("timeout_seconds").and_then(Value::as_u64)?;
    Some(format!("timeout {seconds}s"))
}

fn live_tool_message(
    tool: &ToolDisplay,
    max_lines: usize,
    terminal_width: usize,
    color_enabled: bool,
) -> String {
    let mut bash_command = None;
    let label = match tool.name.as_str() {
        "bash" => {
            let command = format_tool_input("bash", &tool.arguments.text);
            let mut lines = command.lines();
            let first_line = lines.next().unwrap_or_default();
            bash_command = Some(format!(
                "{first_line}{}",
                if lines.next().is_some() { "…" } else { "" }
            ));
            match bash_timeout_label(&tool.arguments.text) {
                Some(timeout) => format!("bash  {timeout}"),
                None => "bash".into(),
            }
        }
        "read" => format!("read  {}", read_detail(&tool.arguments.text)),
        "apply_patch" => {
            let count = patch_operations(&tool.arguments.text).len();
            format!(
                "apply_patch  {count} file{}",
                if count == 1 { "" } else { "s" }
            )
        }
        name => name.to_owned(),
    };
    let label = clean_terminal_line(&label);
    let label = truncate_display_width(&label, terminal_width.saturating_sub(2));
    let mut message = label;
    if let Some(command) = bash_command {
        let command = clean_terminal_line(&command);
        let command = truncate_display_width(&command, terminal_width.saturating_sub(4));
        message.push('\n');
        if color_enabled {
            message.push_str("  \x1b[1;36m$\x1b[0m ");
        } else {
            message.push_str("  $ ");
        }
        message.push_str(&command);
    }
    let output = tool.output.limited(max_lines);
    if max_lines == 0 {
        return message;
    }
    for (index, line) in output.lines.into_iter().enumerate() {
        let prefix = if output.truncated && index == 0 {
            "  … "
        } else {
            "  │ "
        };
        let line = clean_terminal_line(&line);
        let line = truncate_display_width(&line, terminal_width.saturating_sub(4));
        message.push('\n');
        message.push_str(prefix);
        message.push_str(&line);
    }
    message
}

fn bash_visible_output(raw: &str) -> String {
    let mut lines = raw.lines().collect::<Vec<_>>();
    if lines
        .last()
        .is_some_and(|line| line.starts_with("[exit code: ") && line.ends_with(']'))
    {
        lines.pop();
    }
    if lines.last().is_some_and(|line| {
        *line == "[bash cancelled]"
            || (line.starts_with("[bash timed out after ") && line.ends_with(']'))
    }) {
        lines.pop();
    }
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

fn clean_terminal_line(value: &str) -> String {
    String::from_utf8_lossy(&strip_ansi_escapes::strip(value.as_bytes()))
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

fn truncate_display_width(value: &str, max_width: usize) -> String {
    if UnicodeWidthStr::width(value) <= max_width {
        return value.to_owned();
    }
    if max_width == 0 {
        return String::new();
    }
    let content_width = max_width.saturating_sub(1);
    let mut result = String::new();
    let mut width = 0;
    for character in value.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if width + character_width > content_width {
            break;
        }
        result.push(character);
        width += character_width;
    }
    result.push('…');
    result
}

fn write_tool_header(
    writer: &mut dyn Write,
    color_enabled: bool,
    name: &str,
    detail: Option<&str>,
) -> io::Result<()> {
    write_styled(
        writer,
        color_enabled,
        &format!("● {name}"),
        Color::DarkYellow,
        true,
    )?;
    if let Some(detail) = detail.filter(|detail| !detail.is_empty()) {
        write_styled(
            writer,
            color_enabled,
            &format!("  {detail}"),
            Color::Reset,
            false,
        )?;
    }
    writeln!(writer)
}

fn write_tool_completion(
    writer: &mut dyn Write,
    color_enabled: bool,
    symbol: &str,
    color: Color,
    name: &str,
    summary: &str,
) -> io::Result<()> {
    write_styled(
        writer,
        color_enabled,
        &format!("{symbol} {name}  {summary}\n"),
        color,
        true,
    )
}

fn read_detail(raw_arguments: &str) -> String {
    let value = serde_json::from_str::<Value>(raw_arguments).ok();
    let path = value
        .as_ref()
        .and_then(|value| value.get("path"))
        .and_then(Value::as_str)
        .unwrap_or("?");
    let offset = value
        .as_ref()
        .and_then(|value| value.get("offset"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let limit = value
        .as_ref()
        .and_then(|value| value.get("limit"))
        .and_then(Value::as_u64);
    match (offset, limit) {
        (0, None) => path.to_owned(),
        (offset, None) => format!("{path}  from line {}", offset + 1),
        (offset, Some(limit)) => format!("{path}  lines {}-{}", offset + 1, offset + limit),
    }
}

fn render_patch_operations(
    writer: &mut dyn Write,
    color_enabled: bool,
    raw_arguments: &str,
    max_bytes: usize,
    capture_truncated: bool,
    diff_max_lines: usize,
) -> io::Result<()> {
    let operations = patch_operations(raw_arguments).join("\n");
    let limited = limited_text(&operations, max_bytes, 12, false);
    for line in &limited.lines {
        let (operation, path) = line.split_once(' ').unwrap_or(("?", line));
        let color = match operation {
            "A" => Color::DarkGreen,
            "M" => Color::DarkYellow,
            "D" => Color::DarkRed,
            _ => Color::Reset,
        };
        write_styled(
            writer,
            color_enabled,
            &format!("  {operation} "),
            color,
            true,
        )?;
        write_styled(
            writer,
            color_enabled,
            &format!("{path}\n"),
            Color::Reset,
            false,
        )?;
    }
    if capture_truncated || limited.truncated {
        write_styled(
            writer,
            color_enabled,
            "  … patch file list truncated\n",
            Color::DarkYellow,
            false,
        )?;
    }
    render_patch_diff(writer, color_enabled, raw_arguments, diff_max_lines)
}

/// Shows the hunks that were applied. The patch text the model sent *is* the
/// diff, so no diffing is needed and nothing can drift between what is shown and
/// what was written.
fn render_patch_diff(
    writer: &mut dyn Write,
    color_enabled: bool,
    raw_arguments: &str,
    max_lines: usize,
) -> io::Result<()> {
    if max_lines == 0 {
        return Ok(());
    }
    let patch = patch_text(raw_arguments);
    let mut shown = 0;
    let mut truncated = false;
    for line in patch.lines() {
        if line.starts_with("*** ") {
            continue;
        }
        if shown == max_lines {
            truncated = true;
            break;
        }
        let color = match line.as_bytes().first() {
            Some(b'+') => Color::DarkGreen,
            Some(b'-') => Color::DarkRed,
            Some(b'@') => Color::DarkGrey,
            _ => Color::Reset,
        };
        write_styled(
            writer,
            color_enabled,
            &format!("    {line}\n"),
            color,
            false,
        )?;
        shown += 1;
    }
    if truncated {
        write_styled(
            writer,
            color_enabled,
            "    … diff truncated\n",
            Color::DarkYellow,
            false,
        )?;
    }
    Ok(())
}

fn patch_text(raw_arguments: &str) -> String {
    serde_json::from_str::<Value>(raw_arguments)
        .ok()
        .and_then(|value| {
            value
                .get("patch")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_default()
}

fn patch_operations(raw_arguments: &str) -> Vec<String> {
    patch_text(raw_arguments)
        .lines()
        .filter_map(|line| {
            line.strip_prefix("*** Update File: ")
                .map(|path| ('M', path))
                .or_else(|| line.strip_prefix("*** Add File: ").map(|path| ('A', path)))
                .or_else(|| {
                    line.strip_prefix("*** Delete File: ")
                        .map(|path| ('D', path))
                })
        })
        .map(|(operation, path)| format!("{operation} {path}"))
        .collect()
}

fn limited_text(raw: &str, max_bytes: usize, max_lines: usize, tail: bool) -> LimitedText {
    let mut truncated = raw.len() > max_bytes;
    let bounded = if raw.len() <= max_bytes {
        raw
    } else if tail {
        let mut start = raw.len() - max_bytes;
        while start < raw.len() && !raw.is_char_boundary(start) {
            start += 1;
        }
        &raw[start..]
    } else {
        let mut end = max_bytes;
        while end > 0 && !raw.is_char_boundary(end) {
            end -= 1;
        }
        &raw[..end]
    };
    let all_lines = bounded.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    truncated |= all_lines.len() > max_lines;
    let lines = if all_lines.len() <= max_lines {
        all_lines
    } else if tail {
        all_lines[all_lines.len() - max_lines..].to_vec()
    } else {
        all_lines[..max_lines].to_vec()
    };
    LimitedText { lines, truncated }
}

fn render_section(
    writer: &mut dyn Write,
    color_enabled: bool,
    label: &str,
    content: &LimitedText,
    truncated: bool,
) -> io::Result<()> {
    write_styled(
        writer,
        color_enabled,
        &format!("  {label}\n"),
        Color::DarkGrey,
        true,
    )?;
    for line in &content.lines {
        write_styled(writer, color_enabled, "  │ ", Color::DarkGrey, false)?;
        write_styled(
            writer,
            color_enabled,
            &format!("{line}\n"),
            Color::Reset,
            false,
        )?;
    }
    if truncated || content.truncated {
        write_styled(
            writer,
            color_enabled,
            &format!("  … {label} truncated\n"),
            Color::DarkYellow,
            false,
        )?;
    }
    Ok(())
}

fn render_bash_command(
    writer: &mut dyn Write,
    color_enabled: bool,
    content: &LimitedText,
    truncated: bool,
) -> io::Result<()> {
    for (index, line) in content.lines.iter().enumerate() {
        let prompt = if index == 0 { "  $ " } else { "  > " };
        write_styled(writer, color_enabled, prompt, Color::DarkCyan, true)?;
        write_styled(
            writer,
            color_enabled,
            &format!("{line}\n"),
            Color::Reset,
            false,
        )?;
    }
    if truncated || content.truncated {
        write_styled(
            writer,
            color_enabled,
            "  … command truncated\n",
            Color::DarkYellow,
            false,
        )?;
    }
    Ok(())
}

fn render_bash_output(
    writer: &mut dyn Write,
    color_enabled: bool,
    content: &LimitedText,
) -> io::Result<()> {
    if content.lines.is_empty() {
        write_styled(writer, color_enabled, "  ", Color::DarkGrey, false)?;
        if color_enabled {
            writeln!(
                writer,
                "{}",
                "(no output)"
                    .with(Color::DarkGrey)
                    .attribute(Attribute::Italic)
            )?;
        } else {
            writeln!(writer, "(no output)")?;
        }
    }
    for line in &content.lines {
        write_styled(writer, color_enabled, "  │ ", Color::DarkGrey, false)?;
        write_styled(
            writer,
            color_enabled,
            &format!("{line}\n"),
            Color::Reset,
            false,
        )?;
    }
    if content.truncated {
        write_styled(
            writer,
            color_enabled,
            "  … output truncated\n",
            Color::DarkYellow,
            false,
        )?;
    }
    Ok(())
}

fn render_direct_output(
    writer: &mut dyn Write,
    color_enabled: bool,
    content: &LimitedText,
    content_color: Color,
) -> io::Result<()> {
    for line in &content.lines {
        write_styled(writer, color_enabled, "  │ ", Color::DarkGrey, false)?;
        write_styled(
            writer,
            color_enabled,
            &format!("{line}\n"),
            content_color,
            false,
        )?;
    }
    if content.truncated {
        write_styled(
            writer,
            color_enabled,
            "  … output truncated\n",
            Color::DarkYellow,
            false,
        )?;
    }
    Ok(())
}

fn parse_exit_code(output: &str) -> Option<i32> {
    output.lines().rev().find_map(|line| {
        line.trim()
            .strip_prefix("[exit code: ")?
            .strip_suffix(']')?
            .parse()
            .ok()
    })
}

fn tool_summary(name: &str, result: &ToolResult, exit_code: Option<i32>) -> String {
    if name == "bash" && result.output.contains("[bash cancelled]") {
        return "cancelled".into();
    }
    if name == "bash" && result.output.contains("[bash timed out after ") {
        return "timed out".into();
    }
    if let Some(exit_code) = exit_code {
        return format!("exit {exit_code}");
    }
    if result.is_error {
        return "failed".into();
    }
    match name {
        "read" => {
            let lines = result
                .output
                .lines()
                .filter(|line| {
                    line.split_once(": ")
                        .is_some_and(|(number, _)| number.parse::<usize>().is_ok())
                })
                .count();
            format!("{lines} line{}", if lines == 1 { "" } else { "s" })
        }
        "apply_patch" => patch_result_summary(&result.output),
        _ => "done".into(),
    }
}

fn patch_result_summary(output: &str) -> String {
    let mut files = 0usize;
    let mut added = 0usize;
    let mut removed = 0usize;
    for line in output.lines() {
        let Some((_, counts)) = line.rsplit_once(" (+") else {
            continue;
        };
        let Some((added_text, removed_text)) = counts
            .strip_suffix(')')
            .and_then(|counts| counts.split_once(" -"))
        else {
            continue;
        };
        let (Ok(line_added), Ok(line_removed)) =
            (added_text.parse::<usize>(), removed_text.parse::<usize>())
        else {
            continue;
        };
        files += 1;
        added += line_added;
        removed += line_removed;
    }
    if files == 0 {
        format!("{} files", output.lines().count())
    } else {
        format!("{files} files  +{added} -{removed}")
    }
}

fn append_stream(
    writer: &mut dyn Write,
    color_enabled: bool,
    prefix: &str,
    delta: &str,
    prefix_color: Color,
    content_color: Color,
    at_line_start: &mut bool,
) -> io::Result<()> {
    for segment in delta.split_inclusive('\n') {
        if *at_line_start {
            write_styled(writer, color_enabled, prefix, prefix_color, false)?;
        }
        write_styled(writer, color_enabled, segment, content_color, false)?;
        *at_line_start = segment.ends_with('\n');
    }
    Ok(())
}

fn take_complete_lines(pending: &mut String) -> String {
    let Some(last_newline) = pending.rfind('\n') else {
        return String::new();
    };
    pending.drain(..=last_newline).collect()
}

fn write_prefixed_block(
    writer: &mut dyn Write,
    color_enabled: bool,
    prefix: &str,
    content: &str,
    prefix_color: Color,
    prefix_bold: bool,
    content_color: Color,
) -> io::Result<()> {
    for line in content.lines() {
        write_styled(writer, color_enabled, prefix, prefix_color, prefix_bold)?;
        write_styled(writer, color_enabled, line, content_color, false)?;
        writeln!(writer)?;
    }
    if content.is_empty() {
        writeln!(writer)?;
    }
    Ok(())
}

/// Strips the target list the runtime prepends to a user message. It is part of
/// the stored item because the model needs it again on resume, but replaying it
/// would show the user machine scaffolding as their own words.
///
/// Only this block is stripped. A piped-stdin block contains blank lines of its
/// own, so where it ends cannot be determined without guessing, and a wrong
/// guess would hide part of the message instead.
fn without_request_scaffolding(text: &str) -> &str {
    const HEADER: &str = "Files provided with this request:\n";
    let Some(rest) = text.strip_prefix(HEADER) else {
        return text;
    };
    let mut cursor = rest;
    while let Some((line, tail)) = cursor.split_once('\n') {
        if line.starts_with("- ") {
            cursor = tail;
            continue;
        }
        // The blank line that separated this block from the next section.
        return if line.is_empty() { tail } else { cursor };
    }
    cursor
}

fn write_styled(
    writer: &mut dyn Write,
    color_enabled: bool,
    text: &str,
    color: Color,
    bold: bool,
) -> io::Result<()> {
    if !color_enabled {
        return write!(writer, "{text}");
    }
    let styled = text.with(color);
    if bold {
        write!(writer, "{}", styled.attribute(Attribute::Bold))
    } else {
        write!(writer, "{styled}")
    }
}

#[cfg(test)]
mod tests {
    use indicatif::{InMemoryTerm, ProgressDrawTarget};
    use unicode_width::UnicodeWidthStr;

    use super::{GENERATION_ID, LiveTools, ToolDisplay, live_tool_message};

    #[test]
    fn live_parallel_tools_keep_only_the_latest_output_lines() {
        let terminal = InMemoryTerm::new(12, 100);
        let target = ProgressDrawTarget::term_like(Box::new(terminal.clone()));
        let mut live = LiveTools::with_draw_target(target, 2, 100);
        let mut bash = ToolDisplay::new("bash".into(), 1024, 4096);
        bash.arguments.push(r#"{"command":"cargo test"}"#);
        bash.output.push(
            (0..6)
                .map(|index| format!("line {index}\n"))
                .collect::<String>()
                .as_bytes(),
        );
        live.start("bash", live_tool_message(&bash, 2, 100, false));

        let mut read = ToolDisplay::new("read".into(), 1024, 4096);
        read.arguments.push(r#"{"path":"src/lib.rs"}"#);
        live.start("read", live_tool_message(&read, 2, 100, false));
        let contents = terminal.contents();
        assert!(contents.contains("  $ cargo test"), "{contents}");
        assert!(contents.contains("read  src/lib.rs"), "{contents}");
        assert!(contents.contains("line 4"), "{contents}");
        assert!(contents.contains("line 5"), "{contents}");
        assert!(!contents.contains("line 0"), "{contents}");

        live.finish("bash");
        live.update("read", live_tool_message(&read, 2, 100, false));
        let contents = terminal.contents();
        assert!(!contents.contains("  $ cargo test"), "{contents}");
        assert!(contents.contains("read  src/lib.rs"), "{contents}");
    }

    #[test]
    fn live_parallel_tools_without_output_are_adjacent() {
        let terminal = InMemoryTerm::new(30, 100);
        let target = ProgressDrawTarget::term_like(Box::new(terminal.clone()));
        let mut live = LiveTools::with_draw_target(target, 6, 100);
        for index in 0..3 {
            let mut bash = ToolDisplay::new("bash".into(), 1024, 4096);
            bash.arguments.push(r#"{"command":"sleep 20"}"#);
            live.start(
                &format!("bash-{index}"),
                live_tool_message(&bash, 6, 100, false),
            );
        }

        let contents = terminal.contents();
        let positions = contents
            .lines()
            .enumerate()
            .filter_map(|(index, line)| line.contains("  $ sleep 20").then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(positions.len(), 3, "{contents}");
        assert_eq!(positions[1] - positions[0], 2, "{contents}");
        assert_eq!(positions[2] - positions[1], 2, "{contents}");
    }

    #[test]
    fn live_tool_lines_fit_the_terminal_and_strip_control_sequences() {
        let mut bash = ToolDisplay::new("bash".into(), 4096, 4096);
        bash.arguments
            .push(r#"{"command":"cd /a/very/long/directory && grep -rn pyright .gitlab-ci.yml"}"#);
        bash.output
            .push(b"old\n\x1b[31ma very long output line that must be shortened safely\x1b[0m\nlatest\rvalue\n");

        let message = live_tool_message(&bash, 2, 40, false);
        let lines = message.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 4, "{message:?}");
        assert!(UnicodeWidthStr::width(lines[0]) <= 38, "{message:?}");
        assert!(
            lines[1..]
                .iter()
                .all(|line| UnicodeWidthStr::width(*line) <= 40),
            "{message:?}"
        );
        assert!(!message.contains('\x1b'), "{message:?}");
        assert!(!message.contains('\r'), "{message:?}");
    }

    #[test]
    fn live_bash_dollar_uses_the_completed_command_color() {
        let mut bash = ToolDisplay::new("bash".into(), 1024, 4096);
        bash.arguments.push(r#"{"command":"cargo test"}"#);
        let message = live_tool_message(&bash, 2, 100, true);
        let plain = String::from_utf8(strip_ansi_escapes::strip(message.as_bytes())).unwrap();
        assert_eq!(
            plain.lines().collect::<Vec<_>>(),
            ["bash", "  $ cargo test"]
        );
        let dollar = message.find('$').unwrap();
        assert!(message[..dollar].contains("\x1b[1;36m"), "{message:?}");
    }

    #[test]
    fn generation_content_is_visible_above_the_spinner_before_done() {
        let terminal = InMemoryTerm::new(8, 100);
        let target = ProgressDrawTarget::term_like(Box::new(terminal.clone()));
        let mut live = LiveTools::with_draw_target(target, 2, 100);
        live.reserve_generation();
        live.start(GENERATION_ID, "generating".into());
        live.set_generation_content(Some("│ streamed before done".into()));

        let contents = terminal.contents();
        let text = contents.find("│ streamed before done").unwrap();
        let spinner = contents.find("generating").unwrap();
        assert!(text < spinner, "{contents:?}");
    }
}
