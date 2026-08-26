use std::fmt::Write;
use std::path::PathBuf;

use super::{AgentsFile, SkillMetadata};

#[derive(Debug, Clone)]
pub struct ContextInput {
    pub cwd: PathBuf,
    pub agents: Vec<AgentsFile>,
    pub skills: Vec<SkillMetadata>,
    pub platform: String,
    pub shell: String,
}

pub fn build_system_prompt(input: &ContextInput) -> String {
    let mut prompt = format!(
        "You are a coding agent operating in {}.\n\n\
         Inspect only what you need. Prefer minimal, targeted changes.\n\
         Use tools only when the request requires workspace facts or actions. Answer simple conversation, formatting, and direct-answer requests without tools. Never inspect the workspace merely because it is available.\n\
         Use read to inspect files before editing. Use apply_patch for file edits.\n\
         Relative read and apply_patch paths are resolved from the cwd; absolute paths and parent traversal are allowed.\n\
         Use bash for search, git, build, tests, and other shell operations.\n\
         Before any git command, first confirm that .git exists; otherwise do not run git.\n\n\
         When independent tool calls can run in parallel, issue them together.\n\
         Do not parallelize dependent or conflicting mutations.\n\n\
         Follow all active AGENTS.md instructions.\n\
         After modifying code, run relevant validation when practical.\n",
        input.cwd.display()
    );
    prompt.push_str(
        "The Fish script records command metadata in SQLite. The Rust runtime injects recent records from the current Fish session and cwd into each request under `Runtime shell context`. Commands listed there are directly visible to you. Do not claim that the integration lacks context injection or that you cannot see listed commands. Bash tool calls in the conversation are also visible to you.\n",
    );

    if !input.agents.is_empty() {
        prompt.push_str("\nActive AGENTS.md instructions:\n");
        for file in &input.agents {
            let _ = write!(prompt, "\n# {}\n{}\n", file.path.display(), file.content);
        }
    }
    if !input.skills.is_empty() {
        prompt.push_str(
            "\nAvailable Skills. Each one lists only its name and description; when a task matches a description, read the SKILL.md at the listed path before proceeding. Resolve relative paths inside a skill against that skill's own directory, which is the parent of its SKILL.md, and pass absolute paths to tools.\n",
        );
        for skill in &input.skills {
            let _ = write!(
                prompt,
                "- {}\n  {}\n  Path: {}\n",
                skill.name,
                skill.description,
                skill.path.display()
            );
        }
    }
    let _ = write!(
        prompt,
        "\nEnvironment:\n- platform: {}\n- shell: {}\n",
        input.platform, input.shell
    );
    prompt
}
