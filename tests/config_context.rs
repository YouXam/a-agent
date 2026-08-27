use std::fs;
use std::path::Path;

use a_agent::config::{Config, ProviderKind, Rates};
use a_agent::context::{
    ContextInput, SkillMetadata, bound_stdin, build_system_prompt, discover_agents,
    discover_skills, parse_skill_metadata, skill_roots,
};
use a_agent::pricing;
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
fn the_bash_timeout_ceiling_defaults_high_and_cannot_sit_below_the_default() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    let base = r#"
default_model = "test"
[providers.test]
type = "responses"
api_key = "secret"
[models.test]
provider = "test"
model = "test-model"
"#;
    fs::write(home.join(".config/a/config.toml"), base).unwrap();
    let config = Config::load_from(&cwd, &home).unwrap();
    assert_eq!(config.tools.bash_timeout_seconds, 120);
    assert!(
        config.tools.bash_max_timeout_seconds >= config.tools.bash_timeout_seconds,
        "a call must be able to ask for at least the default"
    );

    fs::write(
        home.join(".config/a/config.toml"),
        format!("{base}\n[tools]\nbash_timeout_seconds = 300\nbash_max_timeout_seconds = 60\n"),
    )
    .unwrap();
    let error = Config::load_from(&cwd, &home).unwrap_err().to_string();
    assert!(error.contains("bash_max_timeout_seconds"), "{error}");
    assert!(error.contains("bash_timeout_seconds"), "{error}");
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
fn skill_roots_follow_the_agent_skills_conventions() {
    let home = Path::new("/home/user");
    let project = Path::new("/repo");
    let roots = skill_roots(home, project);
    let shown = roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        shown,
        vec!["/home/user/.agents/skills", "/repo/.agents/skills"],
        "only the shared convention is scanned, user scope first so project wins"
    );
}

#[test]
fn project_skill_overrides_user_skill_metadata() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let project = temp.path().join("repo");
    let user_root = home.join(".agents/skills/review");
    let project_root = project.join(".agents/skills/review");
    fs::create_dir_all(&user_root).unwrap();
    fs::create_dir_all(&project_root).unwrap();
    fs::write(
        user_root.join("SKILL.md"),
        "---\nname: review\ndescription: user\n---\nbody",
    )
    .unwrap();
    fs::write(
        project_root.join("SKILL.md"),
        "---\nname: review\ndescription: project\n---\nbody",
    )
    .unwrap();

    let skills = discover_skills(&skill_roots(&home, &project)).unwrap();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].description, "project");
}

#[test]
fn user_and_project_skills_are_merged_and_client_directories_are_ignored() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let project = temp.path().join("repo");
    for (root, name) in [
        (home.join(".agents/skills/shared-user"), "shared-user"),
        (
            project.join(".agents/skills/shared-project"),
            "shared-project",
        ),
        (home.join(".claude/skills/claude-user"), "claude-user"),
        (project.join(".a/skills/native-project"), "native-project"),
    ] {
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {name} description\n---\nbody"),
        )
        .unwrap();
    }

    let skills = discover_skills(&skill_roots(&home, &project)).unwrap();
    let names = skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["shared-project", "shared-user"], "{skills:?}");
}

#[test]
fn skill_metadata_follows_the_specification() {
    // A nested metadata map must not be mistaken for the top-level name, and a
    // block scalar description has to survive as one line of text.
    let source = concat!(
        "---\n",
        "name: pdf-processing\n",
        "description: >-\n",
        "  Extract PDF text, fill forms, merge files.\n",
        "  Use when handling PDFs.\n",
        "license: Apache-2.0\n",
        "metadata:\n",
        "  name: not-the-skill-name\n",
        "  version: \"1.0\"\n",
        "---\n",
        "SECRET BODY"
    );
    let parsed =
        parse_skill_metadata(source, Path::new("/skills/pdf-processing/SKILL.md")).unwrap();
    assert_eq!(parsed.name, "pdf-processing");
    assert_eq!(
        parsed.description,
        "Extract PDF text, fill forms, merge files. Use when handling PDFs."
    );
    assert!(!parsed.description.contains("SECRET"));
}

#[test]
fn unquoted_colons_in_a_description_are_tolerated() {
    let source = "---\nname: review\ndescription: Use this skill when: the user asks\n---\nbody";
    let parsed = parse_skill_metadata(source, Path::new("/skills/review/SKILL.md")).unwrap();
    assert_eq!(parsed.description, "Use this skill when: the user asks");
}

#[test]
fn a_skill_without_a_description_is_skipped() {
    let temp = tempdir().unwrap();
    let root = temp.path().join(".agents/skills");
    fs::create_dir_all(root.join("broken")).unwrap();
    fs::create_dir_all(root.join("valid")).unwrap();
    fs::write(root.join("broken/SKILL.md"), "---\nname: broken\n---\nbody").unwrap();
    fs::write(
        root.join("valid/SKILL.md"),
        "---\nname: valid\ndescription: usable\n---\nbody",
    )
    .unwrap();

    let skills = discover_skills(std::slice::from_ref(&root)).unwrap();
    assert_eq!(skills.len(), 1, "{skills:?}");
    assert_eq!(skills[0].name, "valid");
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
            path: Path::new("/repo/.agents/skills/review/SKILL.md").to_path_buf(),
        }],
        platform: "linux".into(),
        shell: "/bin/fish".into(),
    });
    assert!(prompt.contains("Run tests."));
    assert!(prompt.contains("Review code."));
    assert!(prompt.contains("/repo/.agents/skills/review/SKILL.md"));
    assert!(!prompt.contains("Targeted files"));
    assert!(prompt.contains("Relative read and apply_patch paths are resolved from the cwd"));
    assert!(prompt.contains("Before any git command, first confirm that .git exists"));
    assert!(prompt.contains("Never inspect the workspace merely because it is available"));
    assert!(prompt.contains("Rust runtime injects recent records"));
    assert!(prompt.contains("integration lacks context injection"));
    assert!(!prompt.contains("SECRET BODY"));
}

fn pricing_catalog() -> String {
    serde_json::json!({
        "deepseek": {
            "models": {
                "shared-model": { "cost": { "input": 0.14, "output": 0.28, "cache_read": 0.0028 } },
                "only-here": { "cost": { "input": 1.0, "output": 2.0 } },
            "tiered": {
                "cost": {
                    "input": 5.0,
                    "output": 30.0,
                    "cache_read": 0.5,
                    "tiers": [
                        {
                            "input": 10.0,
                            "output": 45.0,
                            "cache_read": 1.0,
                            "tier": { "type": "context", "size": 200000 }
                        }
                    ]
                }
            },
                "no-prices": {}
            }
        },
        "mirror": {
            "models": {
                "shared-model": { "cost": { "input": 0.0, "output": 0.0 } }
            }
        }
    })
    .to_string()
}

#[test]
fn a_unique_model_id_resolves_but_an_ambiguous_one_is_reported() {
    let catalog = pricing_catalog();

    let unique = pricing::resolve_from_catalog(&catalog, "only-here", None);
    assert_eq!(
        unique,
        pricing::Resolution::Known {
            schedule: pricing::Schedule::flat(Rates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.0,
                cache_write: 0.0
            }),
            source: "deepseek/only-here".into()
        }
    );

    // Mirrors price the same id differently, so guessing one would show a
    // plausible but wrong number.
    match pricing::resolve_from_catalog(&catalog, "shared-model", None) {
        pricing::Resolution::Ambiguous { model, providers } => {
            assert_eq!(model, "shared-model");
            assert_eq!(providers, vec!["deepseek".to_owned(), "mirror".to_owned()]);
        }
        other => panic!("expected ambiguity, got {other:?}"),
    }

    // An explicit key resolves the ambiguity.
    let keyed =
        pricing::resolve_from_catalog(&catalog, "shared-model", Some("mirror/shared-model"));
    assert!(
        matches!(&keyed, pricing::Resolution::Known { schedule, .. } if schedule.base.input == 0.0),
        "{keyed:?}"
    );

    for (model, key, reason) in [
        ("no-prices", None, "lists no prices"),
        ("missing", None, "was not found"),
        (
            "shared-model",
            Some("deepseek"),
            "must look like provider/model",
        ),
    ] {
        match pricing::resolve_from_catalog(&catalog, model, key) {
            pricing::Resolution::Unknown(message) => {
                assert!(message.contains(reason), "{message}");
            }
            other => panic!("expected Unknown for {model}, got {other:?}"),
        }
    }
}

#[test]
fn cost_uses_each_token_class_at_its_own_rate() {
    let rates = Rates {
        input: 1.0,
        output: 10.0,
        cache_read: 0.1,
        cache_write: 2.0,
    };
    let usage = a_agent::model::Usage {
        input_tokens: Some(1_000_000),
        output_tokens: Some(1_000_000),
        cached_tokens: Some(1_000_000),
        cache_write_tokens: Some(1_000_000),
        total_tokens: Some(4_000_000),
    };
    // total_tokens must not be double counted on top of the parts.
    assert!((pricing::request_cost(usage, rates) - 13.1).abs() < 1e-9);
    assert_eq!(pricing::format_cost(0.0005), "$0.0005");
    assert_eq!(pricing::format_cost(1.5), "$1.50");
}

#[test]
fn a_session_the_provider_never_metered_is_not_priced_at_zero() {
    let schedule = pricing::Schedule::flat(Rates {
        input: 0.14,
        output: 0.28,
        cache_read: 0.0028,
        cache_write: 0.0,
    });
    let blank = a_agent::model::Usage {
        input_tokens: Some(0),
        output_tokens: Some(0),
        cached_tokens: None,
        cache_write_tokens: None,
        total_tokens: Some(0),
    };
    // A gateway that omits usage leaves nothing to price, and claiming a
    // measured near-zero spend would be wrong.
    assert!(!pricing::has_measured_usage(&[]));
    assert!(!pricing::has_measured_usage(&[blank]));
    assert!(pricing::has_measured_usage(&[a_agent::model::Usage {
        input_tokens: Some(1),
        ..blank
    }]));

    // f64's Sum identity is -0.0, so an unpriced session used to render as a
    // negative amount.
    assert_eq!(
        pricing::format_cost(pricing::session_cost(&[], &schedule)),
        "$0.00"
    );
    assert_eq!(pricing::format_cost(-0.0), "$0.00");
}

#[test]
fn explicit_costs_and_a_pricing_key_are_read_from_the_model_profile() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("repo");
    fs::create_dir_all(home.join(".config/a")).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    fs::write(
        home.join(".config/a/config.toml"),
        r#"
default_model = "priced"
[providers.gateway]
type = "responses"
api_key = "secret"
[models.priced]
provider = "gateway"
model = "shared-model"
pricing = "deepseek/shared-model"
[models.priced.cost]
input = 0.5
output = 1.5
"#,
    )
    .unwrap();

    let config = Config::load_from(&cwd, &home).unwrap();
    let selection = config.resolve_model(None, None).unwrap();
    assert_eq!(selection.pricing.as_deref(), Some("deepseek/shared-model"));
    let cost = selection.cost.expect("explicit rates");
    assert_eq!(cost.input, 0.5);
    assert_eq!(cost.output, 1.5);
    assert_eq!(cost.cache_read, 0.0);
}

#[test]
fn a_tiered_schedule_prices_each_request_by_its_own_size() {
    let catalog = pricing_catalog();
    let pricing::Resolution::Known { schedule, .. } =
        pricing::resolve_from_catalog(&catalog, "tiered", None)
    else {
        panic!("expected tiered rates");
    };
    assert!(schedule.is_tiered());
    assert_eq!(schedule.rates_for(199_999).input, 5.0);
    assert_eq!(schedule.rates_for(200_000).input, 10.0);

    let small = a_agent::model::Usage {
        input_tokens: Some(100_000),
        output_tokens: Some(1_000),
        ..Default::default()
    };
    let large = a_agent::model::Usage {
        input_tokens: Some(150_000),
        cached_tokens: Some(60_000),
        output_tokens: Some(1_000),
        ..Default::default()
    };
    // The large request crosses the threshold once cache reads are counted, so
    // summing the session's tokens first would have priced it at the low tier.
    let expected = (100_000.0 * 5.0 + 1_000.0 * 30.0) / 1e6
        + (150_000.0 * 10.0 + 60_000.0 * 1.0 + 1_000.0 * 45.0) / 1e6;
    let actual = pricing::session_cost(&[small, large], &schedule);
    assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");

    let flat = pricing::Schedule::flat(schedule.base);
    assert!(!flat.is_tiered());
    assert!(pricing::session_cost(&[small, large], &flat) < actual);
}

#[test]
fn a_long_skill_with_multibyte_text_does_not_break_discovery() {
    let temp = tempdir().unwrap();
    let root = temp.path().join(".agents/skills");
    fs::create_dir_all(root.join("long")).unwrap();
    fs::create_dir_all(root.join("usable")).unwrap();
    // Frontmatter, then a body long enough that the metadata read stops
    // mid-character. Slicing bytes there must not fail the whole startup.
    let mut long = String::from("---\nname: long\ndescription: A long skill.\n---\n");
    while long.len() < 9000 {
        long.push_str("中文说明");
    }
    fs::write(root.join("long/SKILL.md"), &long).unwrap();
    fs::write(
        root.join("usable/SKILL.md"),
        "---\nname: usable\ndescription: still discovered\n---\nbody",
    )
    .unwrap();

    let skills = discover_skills(std::slice::from_ref(&root)).unwrap();
    let names = skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["long", "usable"], "{skills:?}");
    assert_eq!(skills[0].description, "A long skill.");
}
