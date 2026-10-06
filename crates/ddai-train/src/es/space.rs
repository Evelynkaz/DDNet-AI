//! The parameter space of the ES: the fly's flat parameter vector (the layout of `FlyLearner::params`), its named groups, which of
//! them are searched and by how much each is perturbed.

use std::ops::Range;

use ddai_fly::bundle::FlyBundle;
use ddai_fly::decoder::DecoderParams;
use ddai_fly::encoder::EncoderParams;
use ddai_fly::params::FlyParams;
use serde::{Deserialize, Serialize};

use crate::learner::{Layout, decoder_from_flat, push_decoder, take};

/// The groups of the flat vector, in order. `a` is the type-pair strength, `b`/`theta` the per-type bias and time constant, `enc_g`/`enc_c`
/// the encoder's gain and offset per `(type, channel)`, `enc_bin` the per-distance-bin gains (usually empty) and `dec` the whole decoder.
pub const GROUPS: [&str; 7] = ["a", "b", "theta", "enc_g", "enc_c", "enc_bin", "dec"];

/// The perturbation scale of each group (absolute, in the parameter's own units) and which groups are searched at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpaceConfig {
    /// Names from [`GROUPS`]; empty = every group.
    pub groups: Vec<String>,
    pub sigma_a: f32,
    pub sigma_b: f32,
    pub sigma_theta: f32,
    pub sigma_enc: f32,
    pub sigma_dec: f32,
}

impl Default for SpaceConfig {
    fn default() -> Self {
        SpaceConfig {
            groups: Vec::new(),
            sigma_a: 0.02,
            sigma_b: 0.02,
            sigma_theta: 0.05,
            sigma_enc: 0.05,
            sigma_dec: 0.05,
        }
    }
}

impl SpaceConfig {
    fn sigma_of(&self, group: &str) -> f32 {
        match group {
            "a" => self.sigma_a,
            "b" => self.sigma_b,
            "theta" => self.sigma_theta,
            "dec" => self.sigma_dec,
            _ => self.sigma_enc,
        }
    }
}

/// Where each group lives in the flat vector, and the per-parameter perturbation scale (`0` = not searched).
#[derive(Debug, Clone)]
pub struct ParamSpace {
    pub ranges: Vec<(&'static str, Range<usize>)>,
    pub sigma: Vec<f32>,
    pub total: usize,
}

impl ParamSpace {
    pub fn new(base: &FlyBundle, cfg: &SpaceConfig) -> Result<ParamSpace, String> {
        for g in &cfg.groups {
            if !GROUPS.contains(&g.as_str()) {
                return Err(format!("unknown parameter group {g:?} (known: {GROUPS:?})"));
            }
        }
        let l = Layout::of(&base.fly_params, &base.encoder_params, &base.decoder_params);
        let total = l.total();
        let lens = [l.a, l.b, l.theta, l.g, l.c, l.bin, total - l.decoder_start()];
        let mut at = 0;
        let mut ranges = Vec::new();
        let mut sigma = vec![0.0f32; total];
        for (name, len) in GROUPS.iter().zip(lens) {
            let r = at..at + len;
            at += len;
            if cfg.groups.is_empty() || cfg.groups.iter().any(|g| g == name) {
                sigma[r.clone()].fill(cfg.sigma_of(name));
            }
            ranges.push((*name, r));
        }
        debug_assert_eq!(at, total);
        Ok(ParamSpace { ranges, sigma, total })
    }

    /// Takes `indices` out of the search (`sigma = 0`).
    pub fn exclude(&mut self, indices: &[usize]) {
        for &i in indices {
            self.sigma[i] = 0.0;
        }
    }

    /// Number of searched parameters.
    pub fn searched(&self) -> usize {
        self.sigma.iter().filter(|&&s| s > 0.0).count()
    }

    pub fn group_len(&self, name: &str) -> usize {
        self.ranges.iter().find(|(n, _)| *n == name).map_or(0, |(_, r)| r.len())
    }
}

/// The flat parameter vector of a bundle (the layout of `FlyLearner::params`).
pub fn flatten(b: &FlyBundle) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend_from_slice(&b.fly_params.a);
    out.extend_from_slice(&b.fly_params.b);
    out.extend_from_slice(&b.fly_params.theta);
    out.extend_from_slice(&b.encoder_params.g);
    out.extend_from_slice(&b.encoder_params.c);
    out.extend_from_slice(&b.encoder_params.bin_gain);
    push_decoder(&mut out, &b.decoder_params);
    out
}

/// `base` with its parameters replaced by `flat` (everything else, the frozen calibration and thresholds included, is `base`'s).
pub fn with_params(base: &FlyBundle, flat: &[f32]) -> Result<FlyBundle, String> {
    let l = Layout::of(&base.fly_params, &base.encoder_params, &base.decoder_params);
    if flat.len() != l.total() {
        return Err(format!("expected {} parameters, got {}", l.total(), flat.len()));
    }
    let mut at = 0;
    let a = take(flat, &mut at, l.a).to_vec();
    let b = take(flat, &mut at, l.b).to_vec();
    let theta = take(flat, &mut at, l.theta).to_vec();
    let g = take(flat, &mut at, l.g).to_vec();
    let c = take(flat, &mut at, l.c).to_vec();
    let bin_gain = take(flat, &mut at, l.bin).to_vec();
    let dec: DecoderParams = decoder_from_flat(&flat[l.decoder_start()..], &l);
    let mut out = base.clone();
    out.fly_params = FlyParams { a, b, theta };
    out.encoder_params = EncoderParams { g, c, bin_gain };
    out.decoder_params = dec;
    Ok(out)
}

/// The flat indices of the per-distance-bin gains that nothing reads: a gain is read only by a *spatial* visual channel (walls, hazards,
/// the opponent's position / approach / hook, other players); for own-motion, proprioceptive and opponent-state channels the encoder
/// never looks at it, so its gradient is exactly zero and searching it only drifts it (review of 8.5a, NIT).
pub fn dead_bin_gains(base: &FlyBundle, flyg: &ddai_flyg::Flyg) -> Result<Vec<usize>, String> {
    let cfg = ddai_fly::brain_config::parse_brain_config(&base.brain_config_toml).map_err(|e| e.to_string())?;
    let model =
        ddai_fly::FlyModel::new(flyg.clone(), base.fly_config, base.fly_params.clone()).map_err(|e| e.to_string())?;
    let enc = cfg.encoder_model(&model).map_err(|e| e.to_string())?;
    let nb = enc.ray_grid_config().num_distance_bins;
    if base.encoder_params.bin_gain.is_empty() {
        return Ok(Vec::new());
    }
    let l = Layout::of(&base.fly_params, &base.encoder_params, &base.decoder_params);
    let start = l.a + l.b + l.theta + l.g + l.c;
    let mut dead = Vec::new();
    for (pid, a) in enc.assignments().iter().enumerate() {
        let spatial = ddai_fly::encoder::Channel::parse(a.channel).is_some_and(|c| c.is_spatial());
        if !spatial {
            dead.extend((0..nb).map(|b| start + pid * nb + b));
        }
    }
    Ok(dead)
}
