use std::collections::{BTreeMap, BTreeSet};
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
                "provider authentication is not configured; set api_key in the selected provider, set {}, or update ~/.config/a/config.toml",
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelProfile {
    pub provider: String,
    pub model: String,
    pub effort: Option<String>,
    pub efforts: Vec<String>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u32>,
    /// A models.dev `provider/model` key, needed when the model id alone is
    /// ambiguous across providers that price it differently.
    pub pricing: Option<String>,
    pub cost: Option<Rates>,
    pub headers: BTreeMap<String, String>,
    pub request: BTreeMap<String, serde_json::Value>,
}

/// Token prices in USD per million tokens, matching how models.dev states them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Debug, Clone)]
pub struct ModelSelection {
    pub name: String,
    pub provider_name: String,
    pub provider: ProviderConfig,
    pub effort: Option<String>,
    pub efforts: Vec<String>,
    pub context_window: Option<u64>,
    pub pricing: Option<String>,
    pub cost: Option<Rates>,
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
    pub patch_diff_max_lines: usize,
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
            patch_diff_max_lines: 24,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    /// Applied when a bash call does not ask for a timeout of its own.
    pub bash_timeout_seconds: u64,
    /// Most a bash call may ask for. Commands the model expects to be slow can
    /// raise their own limit up to this, so the default stays short without
    /// making long builds impossible.
    pub bash_max_timeout_seconds: u64,
    pub max_parallel: usize,
    pub max_output_bytes: usize,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            bash_timeout_seconds: 120,
            bash_max_timeout_seconds: 1800,
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
    pub input_history_limit: usize,
    /// How much of a file's previous contents to keep so a rewind can restore
    /// it. Larger files are recorded as unrestorable rather than truncated.
    pub snapshot_max_bytes: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_agent_cycles: 50,
            shell_history_limit: 5000,
            input_history_limit: 1000,
            snapshot_max_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub default_model: String,
    pub providers: BTreeMap<String, ProviderConfig>,
    pub models: BTreeMap<String, ModelProfile>,
    pub ui: UiConfig,
    pub tools: ToolsConfig,
    pub context: ContextConfig,
    pub session: SessionConfig,
}

impl Default for Config {
    fn default() -> Self {
        let default_model = ModelProfile {
            provider: "openai".into(),
            model: "gpt-5.6".into(),
            effort: Some("medium".into()),
            efforts: canonical_efforts(),
            context_window: Some(1_050_000),
            ..ModelProfile::default()
        };
        Self {
            default_model: "default".into(),
            providers: BTreeMap::from([("openai".into(), ProviderConfig::default())]),
            models: BTreeMap::from([("default".into(), default_model)]),
            ui: UiConfig::default(),
            tools: ToolsConfig::default(),
            context: ContextConfig::default(),
            session: SessionConfig::default(),
        }
    }
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

        if merged.get("provider").is_some() {
            anyhow::bail!(
                "legacy [provider] configuration is no longer supported; define [providers.<name>], [models.<name>], and default_model"
            );
        }
        let explicit_api_key_envs = merged
            .get("providers")
            .and_then(toml::Value::as_table)
            .into_iter()
            .flat_map(|providers| providers.iter())
            .filter_map(|(name, value)| value.get("api_key_env").map(|_| name.clone()))
            .collect::<BTreeSet<_>>();
        let mut config: Self = merged.try_into().context("decode merged configuration")?;
        for (name, provider) in &mut config.providers {
            if !explicit_api_key_envs.contains(name) && provider.kind == ProviderKind::Anthropic {
                provider.api_key_env = "ANTHROPIC_API_KEY".into();
            }
        }
        if config.tools.max_parallel == 0 {
            anyhow::bail!("tools.max_parallel must be greater than zero");
        }
        if config.tools.bash_max_timeout_seconds < config.tools.bash_timeout_seconds {
            anyhow::bail!(
                "tools.bash_max_timeout_seconds ({}) must be at least tools.bash_timeout_seconds ({})",
                config.tools.bash_max_timeout_seconds,
                config.tools.bash_timeout_seconds
            );
        }
        config.validate_models()?;
        Ok(config)
    }

    pub fn load(cwd: &Path) -> Result<Self> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .context("HOME is not set")?;
        Self::load_from(cwd, &home)
    }

    pub fn model_names(&self) -> Vec<&str> {
        self.models.keys().map(String::as_str).collect()
    }

    pub fn resolve_model(
        &self,
        name: Option<&str>,
        effort_override: Option<&str>,
    ) -> Result<ModelSelection> {
        let name = name.unwrap_or(&self.default_model);
        let profile = self
            .models
            .get(name)
            .with_context(|| format!("model profile not found: {name}"))?;
        let mut provider = self
            .providers
            .get(&profile.provider)
            .cloned()
            .with_context(|| {
                format!(
                    "provider '{}' referenced by model '{name}' was not found",
                    profile.provider
                )
            })?;
        provider.model = profile.model.clone();
        if let Some(max_tokens) = profile.max_tokens {
            provider.max_tokens = max_tokens;
        }
        provider.headers.extend(profile.headers.clone());
        provider.request.extend(profile.request.clone());

        let effort = effort_override.or(profile.effort.as_deref());
        if let Some(effort) = effort {
            validate_effort(effort)?;
            if !profile.efforts.iter().any(|candidate| candidate == effort) {
                anyhow::bail!("effort '{effort}' is not configured for model '{name}'");
            }
            apply_effort(&mut provider, effort)?;
        }
        Ok(ModelSelection {
            name: name.into(),
            provider_name: profile.provider.clone(),
            provider,
            effort: effort.map(str::to_owned),
            efforts: profile.efforts.clone(),
            context_window: profile.context_window,
            pricing: profile.pricing.clone(),
            cost: profile.cost,
        })
    }

    pub fn resolve_session_model(
        &self,
        profile: Option<&str>,
        provider_type: &str,
        model: &str,
        effort: Option<&str>,
    ) -> Result<ModelSelection> {
        if let Some(profile) = profile {
            return self.resolve_model(Some(profile), effort);
        }
        let kind = ProviderKind::parse(provider_type)?;
        for name in self.models.keys() {
            let selection = self.resolve_model(Some(name), None)?;
            if selection.provider.kind == kind && selection.provider.model == model {
                return self.resolve_model(Some(name), effort);
            }
        }
        anyhow::bail!(
            "session model {provider_type}/{model} does not match a configured model profile"
        )
    }

    fn validate_models(&self) -> Result<()> {
        if self.models.is_empty() {
            anyhow::bail!("at least one [models.<name>] profile is required");
        }
        if !self.models.contains_key(&self.default_model) {
            anyhow::bail!("default_model '{}' was not found", self.default_model);
        }
        for (name, profile) in &self.models {
            if profile.provider.is_empty() || profile.model.is_empty() {
                anyhow::bail!("model '{name}' requires provider and model");
            }
            let provider = self.providers.get(&profile.provider).with_context(|| {
                format!(
                    "provider '{}' referenced by model '{name}' was not found",
                    profile.provider
                )
            })?;
            let max_tokens = profile.max_tokens.unwrap_or(provider.max_tokens);
            if max_tokens == 0 {
                anyhow::bail!("max_tokens must be greater than zero for model '{name}'");
            }
            if let Some(context_window) = profile.context_window
                && context_window <= u64::from(max_tokens)
            {
                anyhow::bail!(
                    "context_window ({context_window}) must be greater than max_tokens ({max_tokens}) for model '{name}'"
                );
            }
            for effort in profile.efforts.iter().chain(profile.effort.iter()) {
                validate_effort(effort)?;
            }
            if let Some(effort) = &profile.effort
                && !profile.efforts.iter().any(|candidate| candidate == effort)
            {
                anyhow::bail!(
                    "default effort '{effort}' is not listed in efforts for model '{name}'"
                );
            }
        }
        Ok(())
    }
}

fn canonical_efforts() -> Vec<String> {
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

fn validate_effort(effort: &str) -> Result<()> {
    if canonical_efforts()
        .iter()
        .any(|candidate| candidate == effort)
    {
        Ok(())
    } else {
        anyhow::bail!("unknown effort '{effort}'")
    }
}

fn apply_effort(provider: &mut ProviderConfig, effort: &str) -> Result<()> {
    match provider.kind {
        ProviderKind::Responses => {
            insert_nested_request_value(&mut provider.request, "reasoning", "effort", effort)
        }
        ProviderKind::Chatcompletion => {
            provider.request.insert(
                "reasoning_effort".into(),
                serde_json::Value::String(effort.into()),
            );
            Ok(())
        }
        ProviderKind::Anthropic => {
            insert_nested_request_value(&mut provider.request, "output_config", "effort", effort)
        }
    }
}

fn insert_nested_request_value(
    request: &mut BTreeMap<String, serde_json::Value>,
    object_key: &str,
    field: &str,
    value: &str,
) -> Result<()> {
    let object = request
        .entry(object_key.into())
        .or_insert_with(|| serde_json::json!({}));
    let object = object
        .as_object_mut()
        .with_context(|| format!("request.{object_key} must be an object"))?;
    object.insert(field.into(), serde_json::Value::String(value.into()));
    Ok(())
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
