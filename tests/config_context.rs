use std::fs;
use std::path::Path;

use a_agent::config::{Config, ProviderKind};
use a_agent::context::{
    ContextInput, SkillMetadata, bound_stdin, build_system_prompt, discover_agents,
    discover_skills, parse_skill_metadata,
};
use tempfile::tempdir;

#[test]
fn rejects_legacy_single_provider_configuration() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
[provider]
type = "responses"
model = "old-model"
"#,
    )
    .unwrap();
    let error = Config::load_from(&cwd, &home).unwrap_err().to_string();
    assert!(error.contains("legacy [provider]"), "{error}");
}

#[test]
fn resolves_multiple_provider_and_model_profiles_with_effort() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(cwd.join(".a")).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
default_model = "fast"

[providers.openai]
type = "responses"
base_url = "https://openai.example/v1"
api_key = "openai-secret"

[providers.claude]
type = "anthropic"
base_url = "https://anthropic.example"
api_key = "claude-secret"

[models.fast]
provider = "openai"
model = "gpt-fast"
effort = "low"
efforts = ["low", "medium", "high"]
context_window = 128000
max_tokens = 4096

[models.deep]
provider = "openai"
model = "gpt-deep"
effort = "high"
efforts = ["medium", "high", "max"]

[models.deep.request]
service_tier = "priority"

[models.claude]
provider = "claude"
model = "claude-deep"
effort = "high"
efforts = ["low", "medium", "high", "max"]
"#,
    )
    .unwrap();
    fs::write(
        cwd.join(".a/config.toml"),
        "[models.deep]\neffort = \"medium\"\n",
    )
    .unwrap();

    let config = Config::load_from(&cwd, &home).unwrap();
    assert_eq!(config.model_names(), ["claude", "deep", "fast"]);

    let fast = config.resolve_model(None, None).unwrap();
    assert_eq!(fast.name, "fast");
    assert_eq!(fast.provider_name, "openai");
    assert_eq!(fast.provider.kind, ProviderKind::Responses);
    assert_eq!(fast.provider.model, "gpt-fast");
    assert_eq!(fast.provider.max_tokens, 4096);
    assert_eq!(fast.context_window, Some(128000));
    assert_eq!(fast.effort.as_deref(), Some("low"));
    assert_eq!(fast.provider.request["reasoning"]["effort"], "low");

    let deep = config.resolve_model(Some("deep"), Some("high")).unwrap();
    assert_eq!(deep.provider.model, "gpt-deep");
    assert_eq!(deep.effort.as_deref(), Some("high"));
    assert_eq!(deep.provider.request["reasoning"]["effort"], "high");
    assert_eq!(deep.provider.request["service_tier"], "priority");

    let claude = config.resolve_model(Some("claude"), Some("max")).unwrap();
    assert_eq!(claude.provider.kind, ProviderKind::Anthropic);
    assert_eq!(claude.provider.request["output_config"]["effort"], "max");
}

#[test]
fn model_without_effort_does_not_inject_reasoning_configuration() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
default_model = "plain"
[providers.internal]
type = "responses"
api_key = "secret"
[models.plain]
provider = "internal"
model = "provider-model"
"#,
    )
    .unwrap();
    let config = Config::load_from(&cwd, &home).unwrap();
    let model = config.resolve_model(None, None).unwrap();
    assert_eq!(model.effort, None);
    assert!(model.efforts.is_empty());
    assert!(!model.provider.request.contains_key("reasoning"));
    assert!(config.resolve_model(None, Some("high")).is_err());
}

#[test]
fn context_window_must_leave_room_for_model_output() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
default_model = "invalid"
[providers.test]
type = "responses"
api_key = "secret"
max_tokens = 4096
[models.invalid]
provider = "test"
model = "test-model"
context_window = 4096
"#,
    )
    .unwrap();

    let error = Config::load_from(&cwd, &home).unwrap_err().to_string();
    assert!(error.contains("context_window"), "{error}");
    assert!(error.contains("max_tokens"), "{error}");
}

#[test]
fn first_run_creates_a_documented_global_config_without_overwriting_it() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let path = Config::ensure_user_config(&home).unwrap().unwrap();
    assert_eq!(path, home.join(".config/a/config.toml"));
    let source = fs::read_to_string(&path).unwrap();
    assert!(source.contains("default_model = \"codex\""));
    assert!(source.contains("[providers.openai]"));
    assert!(source.contains("[models.codex]"));
    assert!(source.contains("type = \"responses\""));
    assert!(source.contains("api_key_env = \"OPENAI_API_KEY\""));
    assert!(source.contains("# api_key = \"sk-...\""));
    assert!(source.contains("# base_url = \"https://gateway.example/v1\""));
    assert!(source.contains("# [providers.anthropic]"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    fs::write(&path, "# user config\n").unwrap();
    assert!(Config::ensure_user_config(&home).unwrap().is_none());
    assert_eq!(fs::read_to_string(path).unwrap(), "# user config\n");
}

#[test]
fn stdin_is_bounded_by_bytes_and_keeps_the_tail() {
    let result = bound_stdin(b"header\nline one\nimportant failure\n", 20);
    assert!(result.starts_with("[stdin truncated; showing last 20 bytes]\n"));
    assert!(result.ends_with("important failure\n"));
}

#[test]
fn skill_frontmatter_is_parsed_without_loading_the_body() {
    let source = "---\nname: rust-review\ndescription: Review Rust safely.\n---\nSECRET BODY";
    let parsed = parse_skill_metadata(source, Path::new("/skills/rust-review/SKILL.md")).unwrap();
    assert_eq!(parsed.name, "rust-review");
    assert_eq!(parsed.description, "Review Rust safely.");
    assert!(!parsed.description.contains("SECRET"));
}

#[test]
fn project_skill_overrides_global_skill_metadata() {
    let temp = tempdir().unwrap();
    let global = temp.path().join("global");
    let project = temp.path().join("repo/.a/skills");
    fs::create_dir_all(global.join("review")).unwrap();
    fs::create_dir_all(project.join("review")).unwrap();
    fs::write(
        global.join("review/SKILL.md"),
        "---\nname: review\ndescription: global\n---\nbody",
    )
    .unwrap();
    fs::write(
        project.join("review/SKILL.md"),
        "---\nname: review\ndescription: project\n---\nbody",
    )
    .unwrap();

    let skills = discover_skills(&global, &project).unwrap();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].description, "project");
}

#[test]
fn agents_discovery_is_ancestry_only_and_broad_to_specific() {
    let temp = tempdir().unwrap();
    let repo = temp.path().join("repo");
    let cwd = repo.join("src/parser");
    fs::create_dir_all(&cwd).unwrap();
    fs::create_dir(repo.join(".git")).unwrap();
    fs::write(repo.join("AGENTS.md"), "repo rules").unwrap();
    fs::write(repo.join("src/AGENTS.md"), "src rules").unwrap();

    let files = discover_agents(&cwd, None).unwrap();
    assert_eq!(
        files
            .iter()
            .map(|item| item.content.as_str())
            .collect::<Vec<_>>(),
        ["repo rules", "src rules"]
    );
}

#[test]
fn global_agents_stays_first_even_when_its_path_is_deeper() {
    let temp = tempdir().unwrap();
    let repo = temp.path().join("repo");
    let cwd = repo.join("src");
    let global = temp.path().join("very/deep/home/.config/a/AGENTS.md");
    fs::create_dir_all(&cwd).unwrap();
    fs::create_dir(repo.join(".git")).unwrap();
    fs::create_dir_all(global.parent().unwrap()).unwrap();
    fs::write(&global, "global rules").unwrap();
    fs::write(repo.join("AGENTS.md"), "repo rules").unwrap();
    let files = a_agent::context::discover_agents_for_targets(&cwd, Some(&global), &[]).unwrap();
    assert_eq!(
        files
            .iter()
            .map(|item| item.content.as_str())
            .collect::<Vec<_>>(),
        ["global rules", "repo rules"]
    );
}

#[test]
fn system_prompt_has_active_rules_and_skill_references_only() {
    let prompt = build_system_prompt(&ContextInput {
        cwd: Path::new("/repo").to_path_buf(),
        agents: vec![a_agent::context::AgentsFile {
            path: Path::new("/repo/AGENTS.md").to_path_buf(),
            content: "Run tests.".into(),
        }],
        skills: vec![SkillMetadata {
            name: "review".into(),
            description: "Review code.".into(),
            path: Path::new("/repo/.a/skills/review/SKILL.md").to_path_buf(),
        }],
        targeted_files: vec![Path::new("/repo/src/a.rs").to_path_buf()],
        platform: "linux".into(),
        shell: "/bin/fish".into(),
    });
    assert!(prompt.contains("Run tests."));
    assert!(prompt.contains("Review code."));
    assert!(prompt.contains("/repo/.a/skills/review/SKILL.md"));
    assert!(prompt.contains("Targeted files:\n- /repo/src/a.rs"));
    assert!(prompt.contains("Relative read and apply_patch paths are resolved from the cwd"));
    assert!(prompt.contains("Before any git command, first confirm that .git exists"));
    assert!(prompt.contains("Never inspect the workspace merely because it is available"));
    assert!(prompt.contains("Rust runtime injects recent records"));
    assert!(prompt.contains("integration lacks context injection"));
    assert!(!prompt.contains("SECRET BODY"));
}
