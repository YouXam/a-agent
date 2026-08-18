use std::fs;
use std::path::Path;

use a_agent::config::{Config, ProviderKind};
use a_agent::context::{
    ContextInput, SkillMetadata, bound_stdin, build_system_prompt, discover_agents,
    discover_skills, parse_skill_metadata,
};
use tempfile::tempdir;

#[test]
fn loads_layered_config_with_third_party_provider_settings() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(cwd.join(".a")).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
[provider]
type = "responses"
model = "global-model"
api_key_env = "CUSTOM_KEY"
api_key = "direct-secret"
base_url = "https://gateway.example/v1"

[provider.headers]
X-Tenant = "acme"

[provider.request]
service_tier = "priority"

[tools]
max_parallel = 3

[ui]
tool_input_max_bytes = 1200
tool_output_max_bytes = 2400
tool_output_max_lines = 9
tool_live_output_lines = 4
"#,
    )
    .unwrap();
    fs::write(
        cwd.join(".a/config.toml"),
        "[provider]\nmodel = \"project-model\"\n",
    )
    .unwrap();

    let config = Config::load_from(&cwd, &home).unwrap();
    assert_eq!(config.provider.kind, ProviderKind::Responses);
    assert_eq!(config.provider.model, "project-model");
    assert_eq!(
        config.provider.base_url.as_deref(),
        Some("https://gateway.example/v1")
    );
    assert_eq!(config.provider.api_key_env, "CUSTOM_KEY");
    assert_eq!(
        config
            .provider
            .resolve_api_key_with(|_| panic!("environment must not be read"))
            .unwrap(),
        "direct-secret"
    );
    assert!(!format!("{config:?}").contains("direct-secret"));
    assert_eq!(
        config.provider.headers.get("X-Tenant").map(String::as_str),
        Some("acme")
    );
    assert_eq!(config.tools.max_parallel, 3);
    assert_eq!(config.ui.tool_input_max_bytes, 1200);
    assert_eq!(config.ui.tool_output_max_bytes, 2400);
    assert_eq!(config.ui.tool_output_max_lines, 9);
    assert_eq!(config.ui.tool_live_output_lines, 4);
    assert_eq!(config.provider.request["service_tier"], "priority");
}

#[test]
fn first_run_creates_a_documented_global_config_without_overwriting_it() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let path = Config::ensure_user_config(&home).unwrap().unwrap();
    assert_eq!(path, home.join(".config/a/config.toml"));
    let source = fs::read_to_string(&path).unwrap();
    assert!(source.contains("type = \"responses\""));
    assert!(source.contains("api_key_env = \"OPENAI_API_KEY\""));
    assert!(source.contains("# api_key = \"sk-...\""));
    assert!(source.contains("# base_url = \"https://gateway.example/v1\""));
    assert!(source.contains("# type = \"anthropic\""));
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
    assert!(prompt.contains("relative to the cwd"));
    assert!(prompt.contains("Before any git command, first confirm that .git exists"));
    assert!(prompt.contains("commands are directly available context"));
    assert!(!prompt.contains("SECRET BODY"));
}
