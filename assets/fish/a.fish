# a-agent Fish integration. Records command metadata only; no output is captured.

if not set -q __a_fish_session_key
    set -g __a_fish_session_key (string join '' a_fish_ $fish_pid _ (date +%s) _ (random) _ (random))
end

function __a_preexec --on-event fish_preexec
    set -g __a_command $argv[1]
    set -g __a_cwd $PWD
    set -g __a_started_at (date +%s%3N)
end

function __a_postexec --on-event fish_postexec
    set -l execution_snapshot $status $pipestatus
    if not set -q __a_started_at
        return
    end
    set -l exit_code $execution_snapshot[1]
    set -l pipe_status (string join ' ' $execution_snapshot[2..])
    set -l finished_at (date +%s%3N)
    set -l duration_ms (math "$finished_at - $__a_started_at")
    command a __record-shell \
        --cwd "$__a_cwd" \
        --command "$__a_command" \
        --exit-code "$exit_code" \
        --pipe-status "$pipe_status" \
        --fish-session-key "$__a_fish_session_key" \
        --started-at "$__a_started_at" \
        --duration-ms "$duration_ms" >/dev/null 2>&1
end

function __a_render_ai_prompt
    set_color --bold brmagenta
    printf '[AI] '
    set_color normal
end

function __a_ai_prompt
    set -l initial (commandline)
    commandline -r ''
    commandline -f repaint

    set -l prompt
    if read --local --line --command "$initial" --prompt __a_render_ai_prompt prompt
        if test -n (string trim -- "$prompt")
            set -lx A_FISH_AI_PROMPT "$prompt"
            command a --fish-ai --fish-session-key "$__a_fish_session_key" --one-turn
            set -l agent_status $status
            return $agent_status
        end
    end
    commandline -f repaint
end

function __a_bind_keys
    bind -M default \cg __a_ai_prompt
    bind -M insert \cg __a_ai_prompt
end

function __a_refresh_bindings --on-event fish_prompt
    __a_bind_keys
end

__a_bind_keys
