# a-agent

`a` is a fast, single-process terminal coding agent written in Rust.

Its core stays intentionally small:

- three tools: `read`, `apply_patch`, and `bash`
- no repository indexing, daemon, PTY wrapper, LSP, or background watcher
- progressive loading for file targets and Skills
- an append-only colored transcript with a Rustyline prompt
- resumable, branchable SQLite conversations
- Anthropic Messages, OpenAI Responses, and OpenAI-compatible Chat Completions

## Build

Rust 1.95 or newer is required.

```bash
cargo build --release
cargo install --path .
```

The installed binary is named `a`.

## Configure

On the first agent run, `a` creates `~/.config/a/config.toml` from
[`config.example.toml`](config.example.toml) and prints its path. The default
Responses settings are active; optional endpoints, provider alternatives,
headers, and request fields remain commented out until needed.

OpenAI Responses:

```toml
[provider]
type = "responses"
base_url = "https://api.openai.com/v1"
model = "gpt-5.6"
api_key_env = "OPENAI_API_KEY"
```

An API key can also be stored directly. A non-empty `api_key` takes precedence
over `api_key_env`:

```toml
[provider]
api_key = "sk-..."
```

This is convenient but stores the secret as plaintext. Keep the file private
and out of version control; environment variables remain the safer default.

Anthropic Messages:

```toml
[provider]
type = "anthropic"
base_url = "https://api.anthropic.com"
model = "your-claude-model"
api_key_env = "ANTHROPIC_API_KEY"
```

Anthropic's Rust SDK appends `/v1`; configure an origin rather than a `/v1`
endpoint. OpenAI protocol base URLs should normally include `/v1`.

OpenAI-compatible Chat Completions:

```toml
[provider]
type = "chatcompletion"
base_url = "https://gateway.example/v1"
model = "provider-model-id"
api_key_env = "GATEWAY_API_KEY"

[provider.headers]
X-Tenant = "acme"

[provider.request]
service_tier = "priority"
```

`base_url`, model, API-key environment variable, custom headers, and extra
request fields are configurable. No provider discovery or capability probe is
performed. Only the configured provider is initialized.

Project-local `.a/config.toml` values override the global config.

## Use

```bash
a
a "fix the parser test"
a -1 "update the version and run tests"
a src/parser.rs "simplify this"
a src/a.rs src/b.rs "remove duplication"
cargo test 2>&1 | a "fix this failure"
a -r
a -r -1 "continue"
a --session a_SESSION_ID "continue"
```

`-1` exits after one complete logical user turn, including all model/tool
cycles. Target files are sent as paths; their contents are read only if the
model calls `read`. Piped stdin is bounded and keeps the tail, which is useful
for compiler and test logs.

## Context

Active `AGENTS.md` files are loaded along the current/target path ancestry,
from broad scope to specific scope. The optional global file is:

```text
~/.config/a/AGENTS.md
```

Skills are discovered only in direct child directories:

```text
~/.config/a/skills/<name>/SKILL.md
<project>/.a/skills/<name>/SKILL.md
```

Only Skill `name`, `description`, and path are loaded during startup. The model
must use `read` to load a relevant Skill body.

## Sessions And Rewind

Sessions are stored in `$XDG_STATE_HOME/a/sessions.db`, or
`~/.local/state/a/sessions.db` when `XDG_STATE_HOME` is unset. SQLite uses WAL,
normal synchronous mode, foreign keys, and semantic write boundaries.

In interactive mode, press `Esc` twice to select a previous user checkpoint.
Rewind moves the session HEAD; it does not delete the old branch.

Default controls:

- `Esc Esc`: rewind picker
- `Ctrl+O`: toggle reasoning visibility
- `Esc` or `Ctrl+C` during a turn: cancel the active turn
- `Ctrl+C` at the prompt: exit
- `Up` / `Down`: input or picker history

The reasoning key is configurable with `ui.reasoning_toggle = "ctrl-r"`.
User, assistant, reasoning, running tools, successful tools, and errors use
distinct semantic colors. Every line is written once to normal terminal
scrollback: there is no viewport, content redraw, history overwrite, or
alternate screen.

Tool calls use domain-specific, bounded output: Bash shows `$` commands and a
live output tail, `read` shows the path/range and a numbered preview, and
`apply_patch` shows A/M/D file operations plus diff statistics. Unknown tools
fall back to labeled input/output blocks. Configure display limits with
`ui.tool_input_max_bytes`, `ui.tool_output_max_bytes`, and
`ui.tool_output_max_lines`.
The transient parallel-tool panel keeps the latest
`ui.tool_live_output_lines` lines per running tool.

## Fish

Install Fish hooks and the AI-mode binding:

```bash
a --install-fish
```

Restart Fish, or source `~/.config/fish/conf.d/a.fish`. `Ctrl+G` opens a
dedicated `[AI]` input prompt. It deliberately does not enable Fish `--shell`
mode, so shell syntax highlighting, autosuggestions, and tab completion do not
apply. Enter runs `a --resume --one-turn` and then returns to the Fish prompt.

The Fish hooks record command, cwd, exit status, pipe status, start time, and
duration. They never capture stdout or stderr.

## Security

`read` and `apply_patch` are restricted to the startup cwd, including symlink
resolution checks. `bash` is intentionally a normal shell running as the
current user; it is not sandboxed. API keys are read from the configured
environment variable and are not persisted in SQLite.

## Validate

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo build --release
```

Set `A_DEBUG_TIMING=1` to print startup phase timings before the first model
request.

Resume startup can be benchmarked against a local mock provider. The final
argument is the number of turns preloaded into the baseline session:

```bash
node scripts/bench-resume.mjs target/release/a 100 1
node scripts/bench-resume.mjs target/release/a 50 100
```
