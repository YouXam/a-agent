use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    Responses,
    Chatcompletion,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Responses => "responses",
            Self::Chatcompletion => "chatcompletion",
        }
    }

    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "anthropic" => Ok(Self::Anthropic),
            "responses" => Ok(Self::Responses),
            "chatcompletion" => Ok(Self::Chatcompletion),
            _ => anyhow::bail!("unknown provider type in session: {value}"),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub kind: ProviderKind,
    pub base_url: Option<String>,
    pub model: String,
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub max_tokens: u32,
    pub request: BTreeMap<String, serde_json::Value>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            kind: ProviderKind::Responses,
            base_url: None,
            model: "gpt-5.6".into(),
            api_key_env: "OPENAI_API_KEY".into(),
            api_key: None,
            headers: BTreeMap::new(),
            max_tokens: 8192,
            request: BTreeMap::new(),
        }
    }
}

impl ProviderConfig {
    pub fn resolve_api_key(&self) -> Result<String> {
        self.resolve_api_key_with(|name| std::env::var(name).ok())
    }

    pub fn resolve_api_key_with(
        &self,
        get_env: impl FnOnce(&str) -> Option<String>,
    ) -> Result<String> {
        if let Some(api_key) = self.api_key.as_ref().filter(|key| !key.is_empty()) {
            return Ok(api_key.clone());
        }
        get_env(&self.api_key_env).ok_or_else(|| {
            anyhow::anyhow!(
                "provider authentication is not configured; set provider.api_key, set {}, or update ~/.config/a/config.toml",
                self.api_key_env
            )
        })
    }
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key_env", &self.api_key_env)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("headers", &self.headers)
            .field("max_tokens", &self.max_tokens)
            .field("request", &self.request)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub show_reasoning: bool,
    pub reasoning_toggle: String,
    pub tool_input_max_bytes: usize,
    pub tool_output_max_bytes: usize,
    pub tool_output_max_lines: usize,
    pub tool_live_output_lines: usize,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            show_reasoning: false,
            reasoning_toggle: "ctrl-o".into(),
            tool_input_max_bytes: 2048,
            tool_output_max_bytes: 8192,
            tool_output_max_lines: 16,
            tool_live_output_lines: 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    pub bash_timeout_seconds: u64,
    pub max_parallel: usize,
    pub max_output_bytes: usize,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            bash_timeout_seconds: 120,
            max_parallel: 8,
            max_output_bytes: 65_536,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    pub shell_history_count: usize,
    pub stdin_max_bytes: usize,
    pub read_max_lines: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            shell_history_count: 5,
            stdin_max_bytes: 131_072,
            read_max_lines: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionConfig {
    pub max_agent_cycles: usize,
    pub shell_history_limit: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_agent_cycles: 50,
            shell_history_limit: 5000,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub provider: ProviderConfig,
    pub ui: UiConfig,
    pub tools: ToolsConfig,
    pub context: ContextConfig,
    pub session: SessionConfig,
}

impl Config {
    pub fn ensure_user_config(home: &Path) -> Result<Option<PathBuf>> {
        let path = home.join(".config/a/config.toml");
        if path.exists() {
            return Ok(None);
        }
        let directory = path.parent().context("config path has no parent")?;
        fs::create_dir_all(directory)
            .with_context(|| format!("create config directory {}", directory.display()))?;
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        temporary.write_all(include_bytes!("../config.example.toml"))?;
        temporary.as_file().sync_all()?;
        match temporary.persist_noclobber(&path) {
            Ok(_) => Ok(Some(path)),
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
            Err(error) => Err(error.error)
                .with_context(|| format!("create initial config {}", path.display())),
        }
    }

    pub fn load_from(cwd: &Path, home: &Path) -> Result<Self> {
        let global = home.join(".config/a/config.toml");
        let project = cwd.join(".a/config.toml");
        let mut merged = toml::Value::Table(Default::default());
        for path in [global, project] {
            if path.is_file() {
                let source = fs::read_to_string(&path)
                    .with_context(|| format!("read config {}", path.display()))?;
                let value = toml::from_str::<toml::Value>(&source)
                    .with_context(|| format!("parse config {}", path.display()))?;
                merge_toml(&mut merged, value);
            }
        }

        let api_key_explicit = merged
            .get("provider")
            .and_then(|value| value.get("api_key_env"))
            .is_some();
        let mut config: Self = merged.try_into().context("decode merged configuration")?;
        if !api_key_explicit && config.provider.kind == ProviderKind::Anthropic {
            config.provider.api_key_env = "ANTHROPIC_API_KEY".into();
        }
        if config.tools.max_parallel == 0 {
            anyhow::bail!("tools.max_parallel must be greater than zero");
        }
        Ok(config)
    }

    pub fn load(cwd: &Path) -> Result<Self> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .context("HOME is not set")?;
        Self::load_from(cwd, &home)
    }
}

fn merge_toml(base: &mut toml::Value, overlay: toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(base), toml::Value::Table(overlay)) => {
            for (key, value) in overlay {
                match base.get_mut(&key) {
                    Some(current) => merge_toml(current, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}
