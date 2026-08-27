# a-agent

`a` is a fast, single-process terminal coding agent written in Rust, for people
who already live in a shell.

It stays at your prompt, leaves your scrollback intact, and already carries the
context your shell just produced. In Fish, `Ctrl+G` turns the current prompt
into an `a> ` prompt:

```text
❯ cargo test
test drops_punctuation ... FAILED
…
error: test failed, to rerun pass `--test slug`

a> fix this                                                         once · tab
▸ Reasoning
✓ read  /tmp/demo/src/lib.rs · 3 lines
  │ 1: pub fn slug(title: &str) -> String {
  │ 2:     title.to_lowercase().replace(' ', "-")
  │ 3: }
│ The test expects punctuation to be dropped. The current `slug` only replaces
 spaces with hyphens, so `"Hello, World"` becomes `"hello,-world"` instead of
`"hello-world"`.
✓ apply_patch  1 files  +6 -5
  M /tmp/demo/src/lib.rs
✓ bash  exit 0
  $ cd /tmp/demo && cargo test
  │ running 1 test
  … output truncated
│ Fixed. The test now passes.
❯
```

Excerpt of a real session; some tool calls are omitted. `a` inherited the
failing command, its exit status, and the directory from the shell — nothing
was pasted in.

It is also a plain CLI, usable from any shell:

```bash
a                                          # interactive
a "fix the parser test"
a -1 "update the version and run tests"    # exit after one turn
a src/parser.rs "simplify this"            # targets are paths, not contents
cargo test 2>&1 | a "fix this failure"     # pipe logs in
a -r "continue"                            # resume this directory's session
```

## Install

```bash
cargo install a-agent
```

Rust 1.95 or newer is required. The installed binary is named `a`. Then install
the shell integration:

```bash
a --install-fish
```

To build from a checkout instead:

```bash
cargo build --release
cargo install --path .
```

The agent runs in any terminal and any shell. The `Ctrl+G` prompt, the
command-history context, and the per-shell conversations target Fish.

## Why a-agent

- **Shell-native.** `Ctrl+G` enters and leaves AI input on the same Fish prompt
  line, and each Fish process gets its own conversation per directory.
- **It knows what you just ran.** Fish hooks record command, cwd, exit status,
  and duration — never output — and the runtime injects the recent ones.
- **Append-only output.** No viewport, alternate screen, or repainted history;
  your scrollback stays scrollable and copyable.
- **No indexing, no daemon.** Startup does not scale with repository size;
  resuming a 100-turn conversation reaches the first request in about 15 ms.
- **Three tools.** `read`, `apply_patch`, `bash`. Independent calls run in
  parallel; patches touching the same file are serialized. A `bash` call can
  raise its own timeout for a release build or a full test run.
- **Resumable and reversible.** SQLite sessions, `Esc` to rewind to an earlier
  message, compaction anchored on the token usage the provider reported.
- **Any provider.** Anthropic Messages, OpenAI Responses, and OpenAI-compatible
  Chat Completions, with third-party base URLs and headers as configuration.

## Configure

The first agent run creates `~/.config/a/config.toml` from
[`config.example.toml`](config.example.toml) and prints its path. Configuration
uses named provider and model profiles; one provider can serve several
independently configured models.

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

`type` is `responses`, `anthropic`, or `chatcompletion`. A provider may set
`api_key` directly, which takes precedence over `api_key_env` at the cost of
storing the secret in plaintext. Anthropic's SDK appends `/v1`, so configure an
origin for it; OpenAI-protocol base URLs normally include `/v1`.

Third-party gateways can extend requests:

```toml
[providers.gateway]
type = "chatcompletion"
base_url = "https://gateway.example/v1"
api_key_env = "GATEWAY_API_KEY"

[providers.gateway.headers]
X-Tenant = "acme"

[providers.gateway.request]
service_tier = "priority"
```

Provider profiles own endpoints and authentication; model profiles own the model
ID, effort choices, `context_window`, `max_tokens`, and optional header or
request overrides. `context_window` must exceed `max_tokens`. Nothing is probed
at startup, and only the selected provider is initialized. Terminal display
limits live under `[ui]`. Project-local `.a/config.toml` overrides the global
file.

## Use

```bash
a src/a.rs src/b.rs "remove duplication"
a -r
a -r -1 "continue"
a --session a_SESSION_ID "continue"
```

`-1` exits after one complete user turn, including all model and tool cycles.
Target files are sent as paths and read only if the model calls `read`.
Interactive `a` defaults to multi-turn mode; `Tab` switches between
`multi · tab` and `once · tab`.

Piped stdin is bounded and keeps the tail, so `cargo test 2>&1 | a "fix this"`
sends the useful end of a long log. Only a pipe or a redirected file counts as
input: a supervisor that hands its child a socket is not piping anything, and
reading that would block forever. A real pipe is read to end of file however
long the producer takes. Like any program that reads stdin, `a` inside a
`while read` loop would consume the rest of that loop's input; redirect it there
with `a ... < /dev/null`.

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

Typing `/` lists them. `/model`, `/effort`, and `/resume` open a selector when
given no argument. Model, effort, and reasoning visibility persist with the
session and are restored on resume. `/resume` only offers sessions from the
current directory.

Typing `@` anywhere in the prompt completes a path from the current directory:
`Up`/`Down` cycle the matches, `Tab` accepts one, and accepting a directory
lists it so the next `Tab` descends. Only one directory is read per keystroke,
and `.git`, `node_modules`, and `target` are skipped. A mentioned path is sent
as a target for that turn, exactly like a path given on the command line, so the
file is still only read if the model calls `read`.

When a model profile sets `context_window`, `a` compacts automatically near
`context_window - max_tokens`. Usage comes from the token counts the provider
reports, so no extra API call is made; only messages newer than the last report
are estimated locally.

`/status` also reports what the session has cost. Prices come from
[models.dev](https://models.dev), fetched only when `/status` runs and then
cached for a day, so startup never pays for it. A model id that several
providers price differently is not guessed at: `/status` names the candidates
and the line to add, because a plausible but wrong number is worse than none.

```toml
[models.deepseek]
pricing = "deepseek/deepseek-v4-flash"   # a models.dev provider/model key

[models.deepseek.cost]                   # or state the prices yourself,
input = 0.14                             # in USD per million tokens
output = 0.28
cache_read = 0.0028
```

Explicit `cost` wins and needs no network. `A_PRICING_URL` points the lookup at
a mirror. The total covers every request the session made, including ones on
branches a rewind left behind.

## Context

Active `AGENTS.md` files are loaded along the current/target path ancestry, from
broad scope to specific scope. The optional global file is:

```text
~/.config/a/AGENTS.md
```

Skills use the [Agent Skills](https://agentskills.io) format: a directory whose
`SKILL.md` carries `name` and `description` in YAML frontmatter, plus any
bundled `scripts/`, `references/`, or `assets/`. Only the shared convention is
scanned, so skills installed by any compliant client are visible here and vice
versa:

```text
~/.agents/skills/<name>/SKILL.md
<project>/.agents/skills/<name>/SKILL.md
```

Direct child directories of those two locations are scanned. A project skill
takes precedence over a user skill of the same name.

Only `name`, `description`, and the path are loaded at startup; the model
`read`s the body when a task matches. A skill missing a description is skipped
with a warning, and a name that disagrees with its directory is warned about but
still loaded.

## Sessions

Sessions live in `$XDG_STATE_HOME/a/sessions.db`, or
`~/.local/state/a/sessions.db` when `XDG_STATE_HOME` is unset. Resuming replays
the history it is about to continue. Cancelling a turn records an explicit
notice, so a later resume does not silently continue the cancelled task.

- `Esc`: rewind to an earlier message of yours; the old branch is kept
- `Esc` or `Ctrl+C` during a turn: cancel it
- `Ctrl+O`: toggle reasoning visibility, configurable as
  `ui.reasoning_toggle = "ctrl-r"`
- `Ctrl+C` at the prompt: discard the typed line, or exit when it is already
  empty. `Ctrl+Y` puts a discarded line back

Every `apply_patch` prints its own hunks, so what was written is visible without
opening the file. Long patches are cut at `ui.patch_diff_max_lines` (24).

`bash` runs under `tools.bash_timeout_seconds` (120) unless the call asks for
longer, which it should for release builds, dependency installs, and full test
runs. Requests are capped at `tools.bash_max_timeout_seconds` (1800), and a
raised limit is shown next to the command so a long-running step does not look
like a hang. A timeout reports both the limit that applied and the ceiling, so
the retry can ask for enough time.

Rewinding can put the files back too. It first lists what it would do to each
file — `delete` for one created after that point, `restore` with line counts for
one that was edited, `recreate` for one that was deleted — and then offers three
choices: rewind the conversation only, rewind and revert the files, or cancel.

A file the agent touched several times counts once, reverting to its state at the
rewind point. A file that changed since the agent last wrote it is marked `keep`
and left alone, so a rewind never discards an edit of yours; so is one whose
previous contents exceeded `session.snapshot_max_bytes` (1 MiB), which is
reported rather than half-restored. There is no separate `/undo`: rewinding to
the most recent message is the same operation.

## Fish

`a --install-fish` writes `~/.config/fish/conf.d/a.fish`. Restart Fish, or
source that file in an already-running shell.

`Ctrl+G` opens the `a> ` prompt. `Ctrl+G` again returns your text to the normal
Fish editor on the same line; `Ctrl+C` clears the line and cancels on an empty
one. `Tab` switches between
`once · tab` and `multi · tab`, and the choice lasts for the life of that Fish
process.

Each Fish process has its own conversation per directory, so two panes in one
repository do not collide. `a --resume` picks up a directory's latest
conversation regardless of which shell created it.

The hooks record command, cwd, exit status, pipe status, and duration — never
stdout or stderr — and the runtime injects the recent ones from that shell and
directory into the request's system context.

## Security

`read`, `apply_patch`, and `bash` run with your permissions and are not
sandboxed; they can reach paths outside the cwd. `apply_patch` updates existing
files in place, so permissions, ownership, hard links, ACLs, and extended
attributes stay attached to the same inode; new files use your normal umask. API
keys come from the selected provider profile or its environment variable and are
never written to SQLite.
