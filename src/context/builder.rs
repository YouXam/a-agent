use std::fmt::Write;
use std::path::PathBuf;

use super::{AgentsFile, SkillMetadata};

#[derive(Debug, Clone)]
pub struct ContextInput {
    pub cwd: PathBuf,
    pub agents: Vec<AgentsFile>,
    pub skills: Vec<SkillMetadata>,
    pub targeted_files: Vec<PathBuf>,
    pub platform: String,
    pub shell: String,
}

pub fn build_system_prompt(input: &ContextInput) -> String {
    let mut prompt = format!(
        "You are a coding agent operating in {}.\n\n\
         Inspect only what you need. Prefer minimal, targeted changes.\n\
         Use read to inspect files before editing. Use apply_patch for file edits.\n\
         apply_patch file paths must be relative to the cwd.\n\
         Use bash for search, git, build, tests, and other shell operations.\n\
         Before any git command, first confirm that .git exists; otherwise do not run git.\n\n\
         When independent tool calls can run in parallel, issue them together.\n\
         Do not parallelize dependent or conflicting mutations.\n\n\
         Follow all active AGENTS.md instructions.\n\
         Read a relevant SKILL.md with read only when needed.\n\
         After modifying code, run relevant validation when practical.\n",
        input.cwd.display()
    );
    prompt.push_str(
        "When a user message includes recent shell commands recorded by the Fish integration, those commands are directly available context. Do not claim that you cannot see commands listed there.\n",
    );

    if !input.agents.is_empty() {
        prompt.push_str("\nActive AGENTS.md instructions:\n");
        for file in &input.agents {
            let _ = write!(prompt, "\n# {}\n{}\n", file.path.display(), file.content);
        }
    }
    if !input.skills.is_empty() {
        prompt.push_str("\nAvailable Skills:\n");
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
    if !input.targeted_files.is_empty() {
        prompt.push_str("\nTargeted files:\n");
        for path in &input.targeted_files {
            let _ = writeln!(prompt, "- {}", path.display());
        }
    }
    prompt
}
