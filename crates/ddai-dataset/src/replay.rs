//! Reconstruction-quality bookkeeping (acceptance criterion 2): how well the physics replay of the
//! reconstructed inputs reproduces the demo's next state, per class, per input channel and per
//! group (map, skill bucket).

use ddai_physics::core::CharacterCore;
use serde::{Deserialize, Serialize};

use crate::types::ReplayClass;

/// The five reconstructed input channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(usize)]
pub enum Channel {
    Direction = 0,
    Jump = 1,
    Hook = 2,
    Fire = 3,
    Aim = 4,
}

impl Channel {
    pub const ALL: [Channel; 5] = [
        Channel::Direction,
        Channel::Jump,
        Channel::Hook,
        Channel::Fire,
        Channel::Aim,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Channel::Direction => "direction",
            Channel::Jump => "jump",
            Channel::Hook => "hook",
            Channel::Fire => "fire",
            Channel::Aim => "aim",
        }
    }
}

/// Ablation result for one channel: among the samples where the channel was *active* (direction
/// != 0, jump/hook/fire pressed, aim relevant because hook or fire is active), did replacing it by
/// its neutral value make the physics replay worse (`confirmed`: the channel is needed to explain
/// the next state), leave it unchanged (`unconstrained`: the demo cannot tell, e.g. holding a key
/// against a wall) or better (`contradicted`: the reconstructed value is probably wrong)?
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelStats {
    pub active: u64,
    pub confirmed: u64,
    pub unconstrained: u64,
    pub contradicted: u64,
    /// Active samples whose value came from a wire core that was fresh at the next snapshot.
    pub fresh: u64,
}

impl ChannelStats {
    pub fn merge(&mut self, o: &ChannelStats) {
        self.active += o.active;
        self.confirmed += o.confirmed;
        self.unconstrained += o.unconstrained;
        self.contradicted += o.contradicted;
        self.fresh += o.fresh;
    }
}

/// Upper edges (px) of the position-error histogram; the last bucket is "more".
pub const ERR_EDGES: [f32; 6] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0];

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplayStats {
    pub samples: u64,
    /// Indexed by [`ReplayClass`] as `u8`.
    pub by_class: [u64; 4],
    /// The same restricted to `ACTIVE` samples (something happened, so a match is not trivial).
    pub active_samples: u64,
    pub active_by_class: [u64; 4],
    /// Samples whose next-snapshot wire core was fresh, i.e. the inputs were observable.
    pub fresh_samples: u64,
    pub fresh_by_class: [u64; 4],
    /// Velocity within 0.5 px/tick of the demo's next state.
    pub vel_close: u64,
    /// Position error (max axis, px) histogram: `[<=0, <=1, <=2, <=4, <=8, <=16, >16]`.
    pub err_hist: [u64; 7],
    pub channels: [ChannelStats; 5],
    /// Character-snapshots with no usable next snapshot (absent or a gap): no sample emitted.
    pub no_next: u64,
    /// Fire validation by `attack_tick`: samples with a fire event, and those where the replayed
    /// weapon fired on exactly the tick of the demo's `attack_tick`.
    pub fire_events: u64,
    pub fire_tick_match: u64,
}

impl ReplayStats {
    pub fn add_sample(&mut self, class: ReplayClass, active: bool, fresh: bool, pos_err: f32, vel_err: f32) {
        self.samples += 1;
        self.by_class[class as usize] += 1;
        if active {
            self.active_samples += 1;
            self.active_by_class[class as usize] += 1;
        }
        if fresh {
            self.fresh_samples += 1;
            self.fresh_by_class[class as usize] += 1;
        }
        if vel_err <= 0.5 {
            self.vel_close += 1;
        }
        let bucket = ERR_EDGES.iter().position(|&e| pos_err <= e).unwrap_or(ERR_EDGES.len());
        self.err_hist[bucket] += 1;
    }

    pub fn merge(&mut self, o: &ReplayStats) {
        self.samples += o.samples;
        self.active_samples += o.active_samples;
        self.fresh_samples += o.fresh_samples;
        self.vel_close += o.vel_close;
        self.no_next += o.no_next;
        self.fire_events += o.fire_events;
        self.fire_tick_match += o.fire_tick_match;
        for i in 0..4 {
            self.by_class[i] += o.by_class[i];
            self.active_by_class[i] += o.active_by_class[i];
            self.fresh_by_class[i] += o.fresh_by_class[i];
        }
        for i in 0..self.err_hist.len() {
            self.err_hist[i] += o.err_hist[i];
        }
        for i in 0..5 {
            self.channels[i].merge(&o.channels[i]);
        }
    }

    /// Fraction (0..=1) of samples in the given classes, `0` for an empty set.
    pub fn frac(count: u64, total: u64) -> f64 {
        if total == 0 { 0.0 } else { count as f64 / total as f64 }
    }

    /// Exact or within 1 px.
    pub fn confident(by_class: &[u64; 4]) -> u64 {
        by_class[ReplayClass::Within1px as usize] + by_class[ReplayClass::Exact as usize]
    }
}

/// Comparison of a replayed core against the demo's reconstructed core for the same tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diff {
    pub class: ReplayClass,
    /// Max-axis position error in px (quantized).
    pub pos_err: f32,
    /// Max-axis velocity error in px/tick (quantized).
    pub vel_err: f32,
    /// Scalar used to compare two replays of the same sample (ablation).
    pub score: f32,
}

/// Compares `replayed` with `target` in the wire quantization (`CCharacterCore::Write`): the same
/// integers a snapshot would carry. `Exact` needs identical position and velocity *and* the same
/// hook state/hooked player; `Within1px` needs the position within `within_px` on both axes.
pub fn diff(replayed: &CharacterCore<f32>, target: &CharacterCore<f32>, within_px: f32) -> Diff {
    let a = replayed.write();
    let b = target.write();
    let dx = (a.x - b.x).abs() as f32;
    let dy = (a.y - b.y).abs() as f32;
    let pos_err = dx.max(dy);
    let dvx = (a.vel_x - b.vel_x).abs() as f32 / 256.0;
    let dvy = (a.vel_y - b.vel_y).abs() as f32 / 256.0;
    let vel_err = dvx.max(dvy);
    let hook_same = a.hook_state == b.hook_state && a.hooked_player == b.hooked_player;
    let class = if pos_err == 0.0 && a.vel_x == b.vel_x && a.vel_y == b.vel_y && hook_same {
        ReplayClass::Exact
    } else if pos_err <= within_px {
        ReplayClass::Within1px
    } else {
        ReplayClass::Off
    };
    let score = pos_err + 2.0 * vel_err + if hook_same { 0.0 } else { 4.0 };
    Diff {
        class,
        pos_err,
        vel_err,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::vmath::Vec2;

    fn core_at(x: f32, y: f32, vx: f32) -> CharacterCore<f32> {
        let mut c = CharacterCore::<f32>::default();
        c.init();
        c.pos = Vec2::new(x, y);
        c.vel = Vec2::new(vx, 0.0);
        c
    }

    #[test]
    fn diff_classifies_exact_within_and_off() {
        let t = core_at(100.0, 50.0, 2.0);
        assert_eq!(diff(&t, &t, 1.0).class, ReplayClass::Exact);
        let near = core_at(101.0, 50.0, 2.0);
        assert_eq!(diff(&near, &t, 1.0).class, ReplayClass::Within1px);
        let vel_off = core_at(100.0, 50.0, 3.0);
        let d = diff(&vel_off, &t, 1.0);
        assert_eq!(
            d.class,
            ReplayClass::Within1px,
            "same position but different velocity is not exact"
        );
        assert!((d.vel_err - 1.0).abs() < 1e-3);
        let far = core_at(103.0, 50.0, 2.0);
        assert_eq!(diff(&far, &t, 1.0).class, ReplayClass::Off);
        assert!(diff(&far, &t, 1.0).score > diff(&near, &t, 1.0).score);
    }

    #[test]
    fn hook_mismatch_prevents_exact() {
        let t = core_at(10.0, 10.0, 0.0);
        let mut h = t;
        h.hook_state = 5;
        assert_eq!(diff(&h, &t, 1.0).class, ReplayClass::Within1px);
    }

    #[test]
    fn stats_bucket_and_merge() {
        let mut s = ReplayStats::default();
        s.add_sample(ReplayClass::Exact, true, true, 0.0, 0.0);
        s.add_sample(ReplayClass::Off, false, false, 20.0, 3.0);
        s.add_sample(ReplayClass::Within1px, true, false, 1.0, 0.2);
        assert_eq!(s.samples, 3);
        assert_eq!(s.err_hist[0], 1);
        assert_eq!(s.err_hist[1], 1);
        assert_eq!(s.err_hist[6], 1);
        assert_eq!(ReplayStats::confident(&s.by_class), 2);
        assert_eq!(s.vel_close, 2);
        let mut t = ReplayStats::default();
        t.merge(&s);
        t.merge(&s);
        assert_eq!(t.samples, 6);
        assert_eq!(t.active_by_class[ReplayClass::Exact as usize], 2);
    }
}
