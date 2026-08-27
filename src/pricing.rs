use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::Rates;
use crate::model::Usage;

pub const SOURCE_URL: &str = "https://models.dev/api.json";
pub const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// The catalog endpoint, overridable with `A_PRICING_URL` for mirrors, proxies,
/// and tests that must not reach a third-party service.
pub fn source_url() -> String {
    std::env::var("A_PRICING_URL").unwrap_or_else(|_| SOURCE_URL.to_owned())
}

/// What is known about a model's prices. Ambiguity is reported rather than
/// resolved by guessing: mirrors of the same model id charge different rates, so
/// picking one would produce a plausible but wrong number.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    Known {
        schedule: Schedule,
        source: String,
    },
    Ambiguous {
        model: String,
        providers: Vec<String>,
    },
    Unknown(String),
}

#[derive(Deserialize)]
struct Provider {
    #[serde(default)]
    models: BTreeMap<String, Model>,
}

#[derive(Deserialize)]
struct Model {
    #[serde(default)]
    cost: Option<Cost>,
}

#[derive(Deserialize)]
struct Cost {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    output: f64,
    #[serde(default)]
    cache_read: f64,
    #[serde(default)]
    cache_write: f64,
    /// Rates that replace the base ones once a request is large enough.
    #[serde(default)]
    tiers: Vec<Tier>,
}

#[derive(Deserialize)]
struct Tier {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    output: f64,
    #[serde(default)]
    cache_read: f64,
    #[serde(default)]
    cache_write: f64,
    tier: TierBound,
}

#[derive(Deserialize)]
struct TierBound {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    size: u64,
}

impl From<&Cost> for Rates {
    fn from(cost: &Cost) -> Self {
        Self {
            input: cost.input,
            output: cost.output,
            cache_read: cost.cache_read,
            cache_write: cost.cache_write,
        }
    }
}

impl From<&Tier> for Rates {
    fn from(tier: &Tier) -> Self {
        Self {
            input: tier.input,
            output: tier.output,
            cache_read: tier.cache_read,
            cache_write: tier.cache_write,
        }
    }
}

/// Base rates plus any context-size tiers, so a request can be priced by how
/// large it actually was.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Schedule {
    pub base: Rates,
    /// Ascending by threshold; the highest threshold at or below the request's
    /// context size wins.
    pub tiers: Vec<(u64, Rates)>,
}

impl Schedule {
    pub fn flat(base: Rates) -> Self {
        Self {
            base,
            tiers: Vec::new(),
        }
    }

    fn from_cost(cost: &Cost) -> Self {
        let mut tiers = cost
            .tiers
            .iter()
            .filter(|tier| tier.tier.kind == "context" && tier.tier.size > 0)
            .map(|tier| (tier.tier.size, Rates::from(tier)))
            .collect::<Vec<_>>();
        tiers.sort_by_key(|(size, _)| *size);
        Self {
            base: Rates::from(cost),
            tiers,
        }
    }

    /// The rates that apply to a request whose context was `context_tokens`.
    pub fn rates_for(&self, context_tokens: u64) -> Rates {
        self.tiers
            .iter()
            .rev()
            .find(|(size, _)| context_tokens >= *size)
            .map_or(self.base, |(_, rates)| *rates)
    }

    pub fn is_tiered(&self) -> bool {
        !self.tiers.is_empty()
    }
}

/// Looks up `model` in a models.dev catalog. `key` is an explicit
/// `provider/model` selector; without one the model id must be unique.
pub fn resolve_from_catalog(catalog: &str, model: &str, key: Option<&str>) -> Resolution {
    let providers = match serde_json::from_str::<BTreeMap<String, Provider>>(catalog) {
        Ok(providers) => providers,
        Err(error) => return Resolution::Unknown(format!("catalog is not valid JSON: {error}")),
    };
    if let Some(key) = key {
        let Some((provider, model)) = key.split_once('/') else {
            return Resolution::Unknown(format!(
                "pricing key {key:?} must look like provider/model"
            ));
        };
        return match providers
            .get(provider)
            .and_then(|entry| entry.models.get(model))
        {
            Some(entry) => match &entry.cost {
                Some(cost) => Resolution::Known {
                    schedule: Schedule::from_cost(cost),
                    source: key.to_owned(),
                },
                None => Resolution::Unknown(format!("{key} lists no prices")),
            },
            None => Resolution::Unknown(format!("{key} was not found on models.dev")),
        };
    }
    let matches = providers
        .iter()
        .filter_map(|(name, provider)| {
            provider
                .models
                .get(model)
                .map(|entry| (name.clone(), entry.cost.as_ref().map(Schedule::from_cost)))
        })
        .collect::<Vec<_>>();
    match matches.len() {
        0 => Resolution::Unknown(format!("{model} was not found on models.dev")),
        1 => {
            let (provider, schedule) = matches.into_iter().next().expect("one match");
            match schedule {
                Some(schedule) => Resolution::Known {
                    schedule,
                    source: format!("{provider}/{model}"),
                },
                None => Resolution::Unknown(format!("{provider}/{model} lists no prices")),
            }
        }
        _ => Resolution::Ambiguous {
            model: model.to_owned(),
            providers: matches.into_iter().map(|(name, _)| name).collect(),
        },
    }
}

pub async fn fetch_catalog(url: &str, timeout: Duration) -> Result<String> {
    let client = reqwest::Client::builder().timeout(timeout).build()?;
    Ok(client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?)
}

/// Cost in USD of one request.
pub fn request_cost(usage: Usage, rates: Rates) -> f64 {
    let scale = |tokens: Option<u64>, rate: f64| tokens.unwrap_or(0) as f64 * rate;
    (scale(usage.input_tokens, rates.input)
        + scale(usage.output_tokens, rates.output)
        + scale(usage.cached_tokens, rates.cache_read)
        + scale(usage.cache_write_tokens, rates.cache_write))
        / 1_000_000.0
}

/// Cost of a whole session. Each request is priced on its own, because a tiered
/// schedule charges by how large that individual request was.
pub fn session_cost(requests: &[Usage], schedule: &Schedule) -> f64 {
    requests
        .iter()
        .map(|usage| request_cost(*usage, schedule.rates_for(request_context(*usage))))
        .sum()
}

/// Tokens the provider had to read for a request, which is what a context tier
/// is measured against.
fn request_context(usage: Usage) -> u64 {
    usage.input_tokens.unwrap_or(0)
        + usage.cached_tokens.unwrap_or(0)
        + usage.cache_write_tokens.unwrap_or(0)
}

/// The candidate most likely to be the model's own vendor, used only to make the
/// suggested `pricing` line useful. A provider whose id prefixes the model id,
/// such as `deepseek` for `deepseek-v4-flash`, beats an alphabetical first pick.
pub fn likely_provider<'a>(model: &str, providers: &'a [String]) -> Option<&'a String> {
    providers
        .iter()
        .find(|provider| model.starts_with(provider.as_str()))
        .or_else(|| providers.first())
}

pub fn format_cost(cost: f64) -> String {
    // `f64`'s Sum identity is -0.0, so an empty or all-zero session would
    // otherwise print "$-0.00".
    let cost = if cost == 0.0 { 0.0 } else { cost };
    if cost > 0.0 && cost < 0.01 {
        format!("${cost:.4}")
    } else {
        format!("${cost:.2}")
    }
}

/// Whether the provider reported any tokens for this session. Without that,
/// there is nothing to price, and a dollar figure would claim a measurement that
/// was never made.
pub fn has_measured_usage(requests: &[Usage]) -> bool {
    requests.iter().any(|usage| {
        [
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_tokens,
            usage.cache_write_tokens,
        ]
        .iter()
        .any(|tokens| tokens.unwrap_or(0) > 0)
    })
}
