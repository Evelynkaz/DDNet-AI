//! Every tunable of the pipeline in one serializable struct; its sha256 goes into the manifest
//! ("config hash"), so two datasets are comparable only when the hashes agree.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Thresholds of the technique detectors (`technique.rs`), pixels and ticks (50 Hz).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TechniqueConfig {
    /// Hook range of the default tuning (`hook_length`).
    pub hook_range: f32,
    /// A hook held for at least this many ticks counts as an episode (not a stray click).
    pub min_hook_ticks: i32,
    /// A freeze tile within this many px of the victim (Chebyshev, along the pull/throw or below)
    /// makes an attack an *attempt* (otherwise nothing is at stake).
    pub freeze_near_px: f32,
    /// How long after the start of an attack/escape a freeze still counts as its outcome.
    pub outcome_ticks: i32,
    /// Hammer reach: the target centre must be within this distance of the hit point.
    pub hammer_reach_px: f32,
    /// T3: attacker must be at least this far above the victim.
    pub swing_min_above_px: f32,
    /// T3: victim horizontal speed (px/tick) that makes the finishing hammer a throw.
    pub swing_victim_vx: f32,
    /// T4: victim falling speed (px/tick) at hook start.
    pub pull_victim_vy: f32,
    /// T5: body push distance (`d < 35`) and the allowed distance of the victim from a freeze tile.
    pub push_dist_px: f32,
    pub push_edge_px: f32,
    /// T10: upward speed (px/tick, negative is up) after a hit that counts as "thrown up".
    pub thrown_up_vy: f32,
    /// T10: freeze ceiling within this many px above the thrown character.
    pub thrown_ceiling_px: f32,
    /// T12: an air jump adding at least this much upward speed within this many px above freeze.
    pub save_jump_dvy: f32,
    pub save_freeze_below_px: f32,
    /// Wall hook: the hook must change the velocity by at least this much (px/tick) *relative to
    /// the no-hook ballistic path* to count as "used to change trajectory".
    pub wall_hook_min_dv: f32,
    /// Wall hook: an enemy counts as closing in when the distance shrinks by at least this many
    /// px/tick between two snapshots (and it is within `hook_range`).
    pub threat_closing_speed: f32,
    /// Wall hook: how far ahead (ticks after the hook ended) the no-hook ballistic path is checked
    /// for freeze, and how far the hook may have moved the character from that path (px) to count
    /// as a position change.
    pub ballistic_horizon_ticks: i32,
    pub wall_hook_min_dpos: f32,
    /// T11: edge stance duration in ticks and the maximum distance of the centre to the edge.
    pub edge_stand_ticks: i32,
    pub edge_dist_px: f32,
    /// T17: radius around a frozen enemy that counts as "standing by".
    pub near_frozen_px: f32,
}

impl Default for TechniqueConfig {
    fn default() -> Self {
        TechniqueConfig {
            hook_range: 380.0,
            min_hook_ticks: 4,
            freeze_near_px: 160.0,
            outcome_ticks: 75,
            hammer_reach_px: 50.0,
            swing_min_above_px: 20.0,
            swing_victim_vx: 2.6,
            pull_victim_vy: 4.2,
            push_dist_px: 35.0,
            push_edge_px: 48.0,
            thrown_up_vy: -8.0,
            thrown_ceiling_px: 192.0,
            save_jump_dvy: 5.0,
            save_freeze_below_px: 96.0,
            wall_hook_min_dv: 2.0,
            threat_closing_speed: 2.0,
            ballistic_horizon_ticks: 25,
            wall_hook_min_dpos: 16.0,
            edge_stand_ticks: 25,
            edge_dist_px: 16.0,
            near_frozen_px: 100.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Snapshot spacing that counts as one decision step (25 Hz at 50 ticks/s).
    pub decision_ticks: i32,
    /// Freeze duration used to estimate the remaining time when it is not on the wire.
    pub freeze_ticks: i32,
    /// D-030 attribution window: the last toucher (hook or hammer) counts within this many ticks.
    pub attribution_ticks: i32,
    /// A credited freeze becomes a *block* when the victim stays frozen this long (D-030 dense
    /// variant: one second) or dies frozen.
    pub block_hold_ticks: i32,
    /// Tag window: samples this many ticks before a signal event are tagged.
    pub signal_lead_ticks: i32,
    /// Skill ranking (see `skill.rs`): minimum visible time to be ranked.
    pub min_ranked_seconds: f32,
    /// Fraction of the ranked players in the Top / Mid buckets (the rest is Low).
    pub top_fraction: f32,
    pub mid_fraction: f32,
    /// Self freezes are weighted by this in the skill score.
    pub self_freeze_weight: f32,
    /// Replay comparison: "within" tolerance in px.
    pub within_px: f32,
    /// `ACTIVE` sample: speed threshold in px/tick.
    pub active_speed: f32,
    /// Target rule: opponents within this distance are "in hook range".
    pub target_range: f32,
    pub chunk_frames: usize,
    pub zstd_level: i32,
    pub technique: TechniqueConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            decision_ticks: 2,
            freeze_ticks: 150,
            attribution_ticks: 50,
            block_hold_ticks: 50,
            signal_lead_ticks: 50,
            min_ranked_seconds: 30.0,
            top_fraction: 0.2,
            mid_fraction: 0.4,
            self_freeze_weight: 0.5,
            within_px: 1.0,
            active_speed: 1.0,
            target_range: 380.0,
            chunk_frames: 2048,
            zstd_level: 3,
            technique: TechniqueConfig::default(),
        }
    }
}

impl Config {
    /// sha256 (hex) of the canonical JSON of the config: field order is the struct order, so the
    /// hash is stable across runs and machines.
    pub fn hash_hex(&self) -> String {
        let json = serde_json::to_vec(self).expect("Config always serializes");
        hex(&Sha256::digest(&json))
    }
}

pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("writing to a String cannot fail");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_and_sensitive() {
        let a = Config::default();
        let mut b = Config::default();
        assert_eq!(a.hash_hex(), b.hash_hex());
        assert_eq!(a.hash_hex().len(), 64);
        b.technique.hook_range += 1.0;
        assert_ne!(a.hash_hex(), b.hash_hex());
    }
}
