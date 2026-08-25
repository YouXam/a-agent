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
cargo install a-agent
# Or install the current checkout:
cargo build --release
cargo install --path .
```

The installed binary is named `a`.

## Configure

On the first agent run, `a` creates `~/.config/a/config.toml` from
[`config.example.toml`](config.example.toml) and prints its path. The default
Responses settings are active. Configuration uses named provider and model
profiles; the legacy singular `[provider]` table is not supported.

One provider can serve multiple independently configured models:

```toml
default_model = "codex"

[providers.openai]
type = "responses"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[models.codex]
provider = "openai"
model = "gpt-5.6"
effort = "medium"
efforts = ["low", "medium", "high", "xhigh", "max"]
context_window = 1050000

[models.fast]
provider = "openai"
model = "gpt-5.6"
effort = "low"
efforts = ["none", "low", "medium"]
```

An API key can also be stored directly. A non-empty `api_key` takes precedence
over `api_key_env`:

```toml
[providers.openai]
api_key = "sk-..."
```

This is convenient but stores the secret as plaintext. Keep the file private
and out of version control; environment variables remain the safer default.

Anthropic Messages:

```toml
[providers.anthropic]
type = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"

[models.claude]
provider = "anthropic"
model = "your-claude-model"
effort = "high"
efforts = ["low", "medium", "high", "max"]
```

Anthropic's Rust SDK appends `/v1`; configure an origin rather than a `/v1`
endpoint. OpenAI protocol base URLs should normally include `/v1`.

OpenAI-compatible Chat Completions:

```toml
[providers.gateway]
type = "chatcompletion"
base_url = "https://gateway.example/v1"
api_key_env = "GATEWAY_API_KEY"

[providers.gateway.headers]
X-Tenant = "acme"

[providers.gateway.request]
service_tier = "priority"

[models.gateway]
provider = "gateway"
model = "provider-model-id"
effort = "medium"
efforts = ["low", "medium", "high"]
```

Provider profiles own endpoints and authentication. Model profiles own the
model ID, effort choices, context window, token limit, and optional
header/request overrides.
When set, `context_window` must be greater than the model's effective
`max_tokens` value.
No provider discovery or capability probe is performed. Only the selected
provider is initialized.

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

Interactive `a` defaults to multi-turn mode. `Tab` switches the right prompt
between `multi · tab` and `once · tab`; once mode exits after the current
response. Input history is stored in SQLite and shared across interactive
a-agent sessions. Fish keeps its own existing history behavior.

Interactive commands:

```text
/model [profile]
/effort [level]
/thinking
/status
/clear
/compact
/resume [session-id]
/help
```

Without an argument, `/model` and `/effort` open an arrow-key selector. Typing
`/` opens a live-filtered command palette below the input. Use `Up`/`Down` to
select a command, then `Tab` to complete it or `Enter` to run it immediately.
Each row shows the command's parameters and purpose. `/thinking` toggles
reasoning visibility, like `Ctrl+O`, and
reports the new state. Model profile and effort changes are persisted with the
session and restored by resume. `/resume` opens a current-directory session
selector when no ID is given; entries are labeled with their first user prompt.
Resuming a session prints a divider and replays its active history before new
input. Sessions from another cwd are rejected. In Fish, selecting a session
also rebinds that Fish process so later `Ctrl+G` turns continue the selected
conversation.
`/compact` immediately summarizes the active branch; automatic compaction uses
the same path.

`/status` reports the active model and effort plus context usage. It
distinguishes the latest provider-reported token anchor from locally estimated
trailing messages, and shows the context window and automatic-compaction
threshold when configured.

When a model profile sets `context_window`, automatic compaction triggers near
`context_window - max_tokens`. Context usage is anchored to the latest valid
token count returned by the provider. OpenAI cached input is normalized into
separate cache-read/cache-write fields; `total_tokens` is preferred when the
provider returns it. Only messages added after the usage anchor use a small
characters-per-token estimate. Compaction does not add a token-count API
request to each turn, and compaction summary requests do not expose tools.

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
An interrupted turn records an internal notice so a later resume does not
silently continue the cancelled task.

In interactive mode, press `Esc` to select a previous user checkpoint. A
second `Esc` triggers it immediately instead of waiting for the terminal's
500 ms escape-sequence timeout. Rewind moves the session HEAD; it does not
delete the old branch. The picker uses `Up`/`Down`, `Enter`, and `Esc`, and
includes persisted user checkpoints from before compaction.

Default controls:

- `Esc` / `Esc Esc`: rewind picker
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
`apply_patch` updates existing files in place so permissions, ownership, hard
links, ACLs, and extended attributes remain attached to the same inode. New
files use the process's normal umask.
The transient parallel-tool panel keeps the latest
`ui.tool_live_output_lines` lines per running tool.
A transient spinner remains below the currently streamed reasoning or assistant
line throughout model generation. Completed lines are appended to scrollback
immediately; only the unfinished line is redrawn. The spinner disappears when
generation completes and never enters scrollback.

## Fish

Install Fish hooks and the AI-mode binding:

```bash
a --install-fish
```

Restart Fish, or source `~/.config/fish/conf.d/a.fish`. `Ctrl+G` opens the
dedicated `a> ` input prompt. It deliberately does not enable Fish `--shell`
mode, so shell syntax highlighting and autosuggestions do not apply. The right
prompt shows `once · tab` by default; press `Tab` to switch to `multi · tab`.
Once mode returns to Fish after one response. Multi mode keeps showing `a> `
after each response until it is switched back to once or cancelled. The mode
choice is retained for the lifetime of that Fish process, including across
`Ctrl+G` and `Ctrl+C` exits from AI input.
Press `Ctrl+G` again to restore the current text to the normal Fish editor on
the same line. Press `Ctrl+C` to cancel the AI input line and open a fresh Fish
prompt.

Each Fish process has an isolated conversation for each cwd. Opening another
Fish process in the same directory starts a separate conversation. Use
`a --resume` explicitly to resume the latest conversation for a cwd regardless
of which Fish process created it.

The Fish hooks record command, cwd, exit status, pipe status, start time, and
duration. They never capture stdout or stderr. The Rust runtime injects recent
command records from that Fish process and cwd into the request's system
context.

## Security

`read`, `apply_patch`, and `bash` can access paths outside the cwd and run with
the current user's permissions; none is sandboxed. API keys are read from the
selected provider profile or its configured environment variable and are never
persisted in SQLite.

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

## Release

Releases are published to crates.io by
[`.github/workflows/publish.yml`](.github/workflows/publish.yml). Authentication
uses crates.io Trusted Publishing over GitHub OIDC, so no registry token is
stored in the repository.

Pushing a `v<version>` tag runs the release job. The tag must match the version
in `Cargo.toml`, and formatting, Clippy, the full test suite, and
`cargo publish --dry-run` must pass before the upload. If the version is already
on crates.io, the upload is skipped instead of failing, so a run can be safely
repeated. The workflow can also be started manually with `workflow_dispatch`,
which skips only the tag check.

```bash
git tag -a v0.1.0 -m 'a-agent 0.1.0'
git push origin v0.1.0
```
