mod agents;
mod builder;
mod skills;
mod stdin;

pub use agents::{AgentsFile, discover_agents, discover_agents_for_targets};
pub use builder::{ContextInput, build_system_prompt};
pub use skills::{SkillMetadata, discover_skills, parse_skill_metadata, skill_roots};
pub use stdin::bound_stdin;
