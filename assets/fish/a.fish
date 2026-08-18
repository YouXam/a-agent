# a-agent Fish integration. Records command metadata only; no output is captured.
# The Rust runtime injects recent records from this Fish session into agent requests.

if not set -q __a_fish_session_key
    set -g __a_fish_session_key (string join '' a_fish_ $fish_pid _ (date +%s) _ (random) _ (random))
end
if not set -q __a_ai_turn_mode
    set -g __a_ai_turn_mode once
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
    set_color --bold brcyan
    printf 'a> '
    set_color normal
end

function __a_render_ai_right_prompt
    if test "$__a_ai_turn_mode" = multi
        set_color --bold brmagenta
        printf 'multi · tab'
    else
        set_color brblack
        printf 'once · tab'
    end
    set_color normal
end

function __a_toggle_turn_mode
    if test "$__a_ai_turn_mode" = multi
        set -g __a_ai_turn_mode once
    else
        set -g __a_ai_turn_mode multi
    end
    commandline -f repaint
end

function __a_handle_tab
    if set -q __a_ai_prompt_active
        __a_toggle_turn_mode
    else
        commandline -f complete
    end
end

function __a_ai_prompt
    if set -q __a_ai_prompt_active
        set -g __a_ai_toggle_text (commandline)
        set -g __a_ai_toggle_requested 1
        commandline -f execute
        return
    end

    set -l initial (commandline)
    set -l shell_prompt_rows (count (fish_prompt))
    commandline -r ''
    commandline -f repaint

    while true
        set -g __a_ai_prompt_active 1
        set -l prompt
        read --local --line --command "$initial" --prompt __a_render_ai_prompt --right-prompt __a_render_ai_right_prompt prompt
        set -l read_status $status
        set -e __a_ai_prompt_active

        if set -q __a_ai_toggle_requested
            set -l shell_text "$__a_ai_toggle_text"
            set -e __a_ai_toggle_requested
            set -e __a_ai_toggle_text
            set -l columns (tput cols 2>/dev/null)
            if not string match --quiet --regex '^[1-9][0-9]*$' -- "$columns"
                set columns 80
            end
            set -l text_width (string length --visible -- "$shell_text")
            set -l ai_rows (math "max(1, ceil((3 + $text_width) / $columns))")
            set -l clear_rows (math "max(1, $shell_prompt_rows + $ai_rows - 1)")
            printf '\e[%dA\r\e[J' $clear_rows
            commandline -r "$shell_text"
            commandline -f repaint
            return
        end

        if test $read_status -ne 0
            commandline -f repaint
            return
        end

        if test -n (string trim -- "$prompt")
            set -lx A_FISH_AI_PROMPT "$prompt"
            command a --fish-ai --fish-session-key "$__a_fish_session_key" --one-turn
            set -l agent_status $status
            if test "$__a_ai_turn_mode" = once
                return $agent_status
            end
            set shell_prompt_rows 1
        else if test "$__a_ai_turn_mode" = once
            commandline -f repaint
            return
        end

        set initial ''
    end
end

function __a_bind_keys
    bind -M default \cg __a_ai_prompt
    bind -M insert \cg __a_ai_prompt
    bind -M default \t __a_handle_tab
    bind -M insert \t __a_handle_tab
end

function __a_refresh_bindings --on-event fish_prompt
    __a_bind_keys
end

__a_bind_keys
