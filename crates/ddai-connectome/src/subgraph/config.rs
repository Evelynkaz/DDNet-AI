//! The `build-subgraph` selection config (TOML): everything the algorithm in
//! `subgraph::select` needs that isn't frozen topology/signs from the connectome itself. See
//! `configs/fly/S.toml` and `configs/fly/M.toml` for the actual committed configs (with
//! per-choice comments citing `docs/FLY.md`), and this crate's README for the selection algorithm
//! itself.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SubgraphConfig {
    pub selection: SelectionParams,
    #[serde(default)]
    pub nt: NtSignParams,
    #[serde(default)]
    pub rf: RfParams,
    pub inputs: InputsConfig,
    pub outputs: OutputsConfig,
    #[serde(default)]
    pub input_channels: Vec<InputChannelConfig>,
    #[serde(default)]
    pub output_groups: Vec<OutputGroupConfig>,
}

/// Selection-algorithm parameters (acceptance criterion 3): all of `k`, `θ_path`, `θ_edge`,
/// `max_hidden`, the weak-pair threshold, and how many ascending (AN) types the data-driven rule
/// keeps, come from here — not hardcoded. `γ` is deliberately absent: per FLY.md §4/the task spec,
/// the subgraph builder never applies it, it only stores the raw `N_i^in` the *model* later
/// computes `Z_i = max(1, N_i^in)^γ` from.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectionParams {
    /// Max path length (hops) from an input seed to an output seed a hidden neuron may lie on.
    pub k: u32,
    /// Minimum synapse weight an edge must have to be used during path search/ranking.
    pub theta_path: u32,
    /// Minimum synapse weight an edge must have to be kept in the final induced subgraph.
    pub theta_edge: u32,
    /// Soft cap on the number of top-ranked hidden-neuron *candidates* kept before bilateral
    /// completion — completion can push the final hidden count slightly above this (documented,
    /// reported).
    pub max_hidden: u32,
    /// Type pairs with total subgraph synapses below this share one learnable parameter per
    /// (pre sign, pre superclass, post superclass) instead of getting their own.
    pub weak_pair_threshold: u64,
    /// How many ascending (AN) types the data-driven rule (`subgraph::an_pick`) keeps.
    pub an_top_n_types: u32,
}

/// Sign rule parameters (acceptance criterion 4). Defaults match FLY.md §4/§7.3 and D-013.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct NtSignParams {
    pub sign_confidence_threshold: f32,
    pub use_best_guess_when_uncertain: bool,
    pub ach_sign: i8,
    pub gaba_sign: i8,
    pub glu_sign: i8,
    pub his_sign: i8,
    /// Sign used for dopamine/serotonin/octopamine (and tyramine, which MaleCNS v1.0's NT file
    /// does not actually predict as a distinct class — see this crate's README) — "excluded from
    /// fast transmission by default", per the spec.
    pub modulatory_sign: i8,
}

impl Default for NtSignParams {
    fn default() -> Self {
        Self {
            sign_confidence_threshold: 0.5,
            use_best_guess_when_uncertain: false,
            ach_sign: 1,
            gaba_sign: -1,
            glu_sign: -1,
            his_sign: -1,
            modulatory_sign: 0,
        }
    }
}

/// Receptive-field computation parameters (acceptance criterion 5; review round 1, F2).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RfParams {
    /// Minimum synapse-weighted hex support (direct + one-hop-derived, see `subgraph::rf`'s
    /// module docs) a visual-input neuron needs before its RF center is trusted; below this, the
    /// documented fallback is used instead, even if the raw support is nonzero. Review round 1
    /// found several visual-input types (HSE/HSN/H2/VS, and a good fraction of LC9) resting on
    /// 0-2 real synapses, which is not enough signal to trust as a real retinotopic position.
    pub min_hex_support: u64,
}

impl Default for RfParams {
    fn default() -> Self {
        Self { min_hex_support: 10 }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputsConfig {
    /// Visual projection neuron (VPN) type names — exact spelling, verified to exist in MaleCNS
    /// v1.0 (see this crate's README and the build report for the verification results).
    pub visual_types: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputsConfig {
    /// Descending neuron (DN) type names.
    pub dn_types: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputChannelConfig {
    /// Field name is `type` in the TOML; renamed here since `type` is a Rust keyword.
    #[serde(rename = "type")]
    pub type_name: String,
    pub channels: Vec<String>,
}

/// Which side(s) of a type's neurons belong to one output group (e.g. `direction_left` only
/// wants one side of the steering DNs — see this crate's README for the exact convention used).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SideFilter {
    #[default]
    Both,
    L,
    R,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputGroupConfig {
    pub action: String,
    /// Type names (must all also be listed in `outputs.dn_types`).
    pub types: Vec<String>,
    #[serde(default)]
    pub side: SideFilter,
}

pub fn load_config(path: &Path) -> Result<SubgraphConfig> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let config: SubgraphConfig =
        toml::from_str(&text).with_context(|| format!("parsing {} as TOML", path.display()))?;
    validate_config(&config).with_context(|| format!("validating {}", path.display()))?;
    Ok(config)
}

fn validate_config(config: &SubgraphConfig) -> Result<()> {
    let s = &config.selection;
    if s.k == 0 {
        bail!("selection.k must be >= 1 (0 would make hidden-neuron selection vacuous)");
    }
    if s.max_hidden == 0 {
        bail!("selection.max_hidden must be >= 1");
    }
    if !(0.0..=1.0).contains(&config.nt.sign_confidence_threshold) {
        bail!(
            "nt.sign_confidence_threshold {} must be in [0, 1]",
            config.nt.sign_confidence_threshold
        );
    }
    for sign_name @ (name, value) in [
        ("ach_sign", config.nt.ach_sign),
        ("gaba_sign", config.nt.gaba_sign),
        ("glu_sign", config.nt.glu_sign),
        ("his_sign", config.nt.his_sign),
        ("modulatory_sign", config.nt.modulatory_sign),
    ]
    .iter()
    {
        let _ = sign_name;
        if !(-1..=1).contains(value) {
            bail!("nt.{name} must be -1, 0 or 1, got {value}");
        }
    }
    if config.inputs.visual_types.is_empty() {
        bail!("inputs.visual_types must not be empty");
    }
    if config.outputs.dn_types.is_empty() {
        bail!("outputs.dn_types must not be empty");
    }
    for group in &config.output_groups {
        for ty in &group.types {
            if !config.outputs.dn_types.contains(ty) {
                bail!(
                    "output_groups: action {:?} references type {ty:?}, which is not in outputs.dn_types",
                    group.action
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_toml() -> &'static str {
        r#"
[selection]
k = 2
theta_path = 5
theta_edge = 3
max_hidden = 100
weak_pair_threshold = 20
an_top_n_types = 10

[inputs]
visual_types = ["LC10a"]

[outputs]
dn_types = ["DNa02"]
"#
    }

    #[test]
    fn parses_minimal_config_with_nt_defaults() {
        let config: SubgraphConfig = toml::from_str(minimal_toml()).unwrap();
        assert_eq!(config.selection.k, 2);
        assert_eq!(config.nt.ach_sign, 1);
        assert_eq!(config.nt.sign_confidence_threshold, 0.5);
        assert!(!config.nt.use_best_guess_when_uncertain);
        assert!(config.input_channels.is_empty());
        assert!(config.output_groups.is_empty());
        validate_config(&config).unwrap();
    }

    #[test]
    fn parses_full_config_with_overrides() {
        let toml = format!(
            "{}\n{}",
            minimal_toml(),
            r#"
[nt]
sign_confidence_threshold = 0.7
use_best_guess_when_uncertain = true
glu_sign = 1

[[input_channels]]
type = "LC10a"
channels = ["opponent_position"]

[[output_groups]]
action = "aim"
types = ["DNa02"]
side = "both"
"#
        );
        let config: SubgraphConfig = toml::from_str(&toml).unwrap();
        assert_eq!(config.nt.sign_confidence_threshold, 0.7);
        assert!(config.nt.use_best_guess_when_uncertain);
        assert_eq!(config.nt.glu_sign, 1);
        assert_eq!(config.input_channels.len(), 1);
        assert_eq!(config.input_channels[0].type_name, "LC10a");
        assert_eq!(config.output_groups.len(), 1);
        assert_eq!(config.output_groups[0].side, SideFilter::Both);
        validate_config(&config).unwrap();
    }

    #[test]
    fn rejects_k_zero() {
        let mut config: SubgraphConfig = toml::from_str(minimal_toml()).unwrap();
        config.selection.k = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn rejects_out_of_range_sign() {
        let mut config: SubgraphConfig = toml::from_str(minimal_toml()).unwrap();
        config.nt.ach_sign = 5;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn rejects_output_group_referencing_an_unlisted_type() {
        let toml = format!(
            "{}\n{}",
            minimal_toml(),
            r#"
[[output_groups]]
action = "jump"
types = ["DNp01"]
"#
        );
        let config: SubgraphConfig = toml::from_str(&toml).unwrap();
        let err = validate_config(&config).unwrap_err();
        assert!(format!("{err:#}").contains("DNp01"));
    }

    #[test]
    fn load_config_reads_and_parses_a_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.toml");
        std::fs::write(&path, minimal_toml()).unwrap();
        let config = load_config(&path).unwrap();
        assert_eq!(config.selection.theta_path, 5);
    }
}
