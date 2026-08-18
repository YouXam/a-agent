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

use crate::model::{StreamEvent, ToolResult};
use crate::provider::EventSink;

const GENERATION_ID: &str = "__a_generation";

#[derive(Debug, Clone, Copy)]
pub struct RenderLimits {
    pub tool_input_max_bytes: usize,
    pub tool_output_max_bytes: usize,
    pub tool_output_max_lines: usize,
    pub tool_live_output_lines: usize,
}

impl Default for RenderLimits {
    fn default() -> Self {
        Self {
            tool_input_max_bytes: 2048,
            tool_output_max_bytes: 8192,
            tool_output_max_lines: 16,
            tool_live_output_lines: 6,
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
    reasoning_at_line_start: bool,
    assistant_at_line_start: bool,
    tools: HashMap<String, ToolDisplay>,
    live: Option<LiveTools>,
}

struct ToolDisplay {
    name: String,
    arguments: BoundedInput,
    output: TailBuffer,
    input_rendered: bool,
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
                Color::Cyan,
                true,
                Color::White,
            )?;
            state.writer.flush()
        })
    }

    pub fn render_status(&self, message: &str) -> io::Result<()> {
        self.with_state(|state| {
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

    pub fn render_rewind_options(&self, choices: &[(String, String)]) -> io::Result<()> {
        self.with_state(|state| {
            state.finish_open_lines()?;
            write_styled(
                &mut state.writer,
                state.color,
                "Rewind to:\n",
                Color::Magenta,
                true,
            )?;
            for (index, (_, label)) in choices.iter().enumerate() {
                write_styled(
                    &mut state.writer,
                    state.color,
                    &format!("  {}  {label}\n", index + 1),
                    Color::Grey,
                    false,
                )?;
            }
            state.writer.flush()
        })
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
            if !matches!(&event, StreamEvent::GenerationStart) {
                state.finish_generation();
            }
            match event {
                StreamEvent::GenerationStart => state.start_generation()?,
                StreamEvent::ReasoningDelta { delta } => state.reasoning_delta(&delta)?,
                StreamEvent::TextDelta { delta } => state.assistant_delta(&delta)?,
                StreamEvent::ToolCallStart { id, name } => {
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
                    state.finish_open_lines()?;
                    write_styled(
                        &mut state.writer,
                        state.color,
                        &format!("× {message}\n"),
                        Color::Red,
                        true,
                    )?;
                }
                StreamEvent::Done => {
                    state.finish_open_lines()?;
                    state.reasoning_announced = false;
                    state.reasoning_buffer.clear();
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
            live.finish(GENERATION_ID);
            live.restore_for_commit();
        }
    }

    fn reasoning_delta(&mut self, delta: &str) -> io::Result<()> {
        self.reasoning_buffer.push_str(delta);
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
        if !self.reasoning_at_line_start {
            writeln!(self.writer)?;
            self.reasoning_at_line_start = true;
        }
        append_stream(
            &mut self.writer,
            self.color,
            "│ ",
            delta,
            Color::Green,
            Color::White,
            &mut self.assistant_at_line_start,
        )
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
            let message = live_tool_message(tool, live.visible_lines, live.terminal_width);
            live.start(id, message);
        }
    }

    fn update_live_tool(&mut self, id: &str) {
        let Some(tool) = self.tools.get(id) else {
            return;
        };
        if let Some(live) = &mut self.live {
            let message = live_tool_message(tool, live.visible_lines, live.terminal_width);
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
        bash_completion: Option<(&str, Color, &str)>,
    ) -> io::Result<()> {
        let Some(tool) = self.tools.get_mut(id) else {
            return Ok(());
        };
        if tool.input_rendered {
            return Ok(());
        }
        tool.input_rendered = true;
        let name = tool.name.clone();
        let raw_arguments = tool.arguments.text.clone();
        let truncated = tool.arguments.truncated;
        self.finish_open_lines()?;
        match name.as_str() {
            "read" => render_read_call(&mut self.writer, self.color, &raw_arguments),
            "apply_patch" => {
                write_tool_header(&mut self.writer, self.color, &name, None)?;
                render_patch_operations(
                    &mut self.writer,
                    self.color,
                    &raw_arguments,
                    self.limits.tool_input_max_bytes,
                    truncated,
                )
            }
            "bash" => {
                if let Some((symbol, color, summary)) = bash_completion {
                    write_tool_completion(
                        &mut self.writer,
                        self.color,
                        symbol,
                        color,
                        &name,
                        summary,
                    )?;
                } else {
                    write_tool_header(&mut self.writer, self.color, &name, None)?;
                }
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
        let color = if failed { Color::Red } else { Color::Green };
        let summary = tool_summary(&name, result, exit_code);
        let bash_completion = (name == "bash").then_some((symbol, color, summary.as_str()));
        self.render_tool_input(id, bash_completion)?;
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
            "read" => render_direct_output(&mut self.writer, self.color, &output, Color::Grey)?,
            "apply_patch" if result.is_error => {
                render_direct_output(&mut self.writer, self.color, &output, Color::Red)?
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
        if name != "bash" {
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
            input_rendered: false,
        }
    }
}

impl LiveTools {
    fn new(max_lines: usize) -> Self {
        let terminal_width = crossterm::terminal::size()
            .map(|(width, _)| usize::from(width))
            .unwrap_or(80);
        Self::with_draw_target(
            ProgressDrawTarget::stdout_with_hz(20),
            max_lines,
            terminal_width,
        )
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
        self.visible_lines = self.max_lines.min(rows_per_tool.saturating_sub(1));
        let desired = tool_count.saturating_mul(self.visible_lines.saturating_add(1));
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
        if height > 0 {
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

    fn finish(&mut self, id: &str) {
        if let Some(progress) = self.entries.remove(id) {
            progress.disable_steady_tick();
            self.multi.remove(&progress);
        }
    }

    fn restore_for_commit(&mut self) {
        let _ = self.multi.clear();
        if self.reserved {
            let mut stdout = io::stdout();
            let _ = execute!(stdout, RestorePosition, Clear(ClearType::FromCursorDown));
            let _ = stdout.flush();
            self.reserved = false;
        }
    }

    fn resume_after_commit(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        self.reserve(self.entries.len());
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

fn live_tool_message(tool: &ToolDisplay, max_lines: usize, terminal_width: usize) -> String {
    let label = match tool.name.as_str() {
        "bash" => {
            let command = format_tool_input("bash", &tool.arguments.text);
            let mut lines = command.lines();
            let first_line = lines.next().unwrap_or_default();
            format!(
                "bash  $ {first_line}{}",
                if lines.next().is_some() { "…" } else { "" }
            )
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
    let output = tool.output.limited(max_lines);
    if max_lines == 0 {
        return label;
    }
    let mut message = label;
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
        Color::Yellow,
        true,
    )?;
    if let Some(detail) = detail.filter(|detail| !detail.is_empty()) {
        write_styled(
            writer,
            color_enabled,
            &format!("  {detail}"),
            Color::Grey,
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

fn render_read_call(
    writer: &mut dyn Write,
    color_enabled: bool,
    raw_arguments: &str,
) -> io::Result<()> {
    let detail = read_detail(raw_arguments);
    write_tool_header(writer, color_enabled, "read", Some(&detail))
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
) -> io::Result<()> {
    let operations = patch_operations(raw_arguments).join("\n");
    let limited = limited_text(&operations, max_bytes, 12, false);
    for line in &limited.lines {
        let (operation, path) = line.split_once(' ').unwrap_or(("?", line));
        let color = match operation {
            "A" => Color::Green,
            "M" => Color::Yellow,
            "D" => Color::Red,
            _ => Color::Grey,
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
            Color::Grey,
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
    Ok(())
}

fn patch_operations(raw_arguments: &str) -> Vec<String> {
    let patch = serde_json::from_str::<Value>(raw_arguments)
        .ok()
        .and_then(|value| {
            value
                .get("patch")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_default();
    patch
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
            Color::Grey,
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
    for line in &content.lines {
        write_styled(writer, color_enabled, "  $ ", Color::Cyan, true)?;
        write_styled(
            writer,
            color_enabled,
            &format!("{line}\n"),
            Color::White,
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
            Color::Grey,
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
        "read" => format!("{} lines", result.output.lines().count()),
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

    use super::{LiveTools, ToolDisplay, live_tool_message};

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
        live.start("bash", live_tool_message(&bash, 2, 100));

        let mut read = ToolDisplay::new("read".into(), 1024, 4096);
        read.arguments.push(r#"{"path":"src/lib.rs"}"#);
        live.start("read", live_tool_message(&read, 2, 100));
        let contents = terminal.contents();
        assert!(contents.contains("bash  $ cargo test"), "{contents}");
        assert!(contents.contains("read  src/lib.rs"), "{contents}");
        assert!(contents.contains("line 4"), "{contents}");
        assert!(contents.contains("line 5"), "{contents}");
        assert!(!contents.contains("line 0"), "{contents}");

        live.finish("bash");
        live.update("read", live_tool_message(&read, 2, 100));
        let contents = terminal.contents();
        assert!(!contents.contains("bash  $ cargo test"), "{contents}");
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
            live.start(&format!("bash-{index}"), live_tool_message(&bash, 6, 100));
        }

        let contents = terminal.contents();
        let positions = contents
            .lines()
            .enumerate()
            .filter_map(|(index, line)| line.contains("bash  $ sleep 20").then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(positions.len(), 3, "{contents}");
        assert_eq!(positions[1] - positions[0], 1, "{contents}");
        assert_eq!(positions[2] - positions[1], 1, "{contents}");
    }

    #[test]
    fn live_tool_lines_fit_the_terminal_and_strip_control_sequences() {
        let mut bash = ToolDisplay::new("bash".into(), 4096, 4096);
        bash.arguments
            .push(r#"{"command":"cd /a/very/long/directory && grep -rn pyright .gitlab-ci.yml"}"#);
        bash.output
            .push(b"old\n\x1b[31ma very long output line that must be shortened safely\x1b[0m\nlatest\rvalue\n");

        let message = live_tool_message(&bash, 2, 40);
        let lines = message.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3, "{message:?}");
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
}
