//! The controls' input vector: the fly's ray-grid features and proprioception, flattened.
//!
//! Layout: seven spatial channels of `num_directions * num_bins` values each (opponent position,
//! opponent approach, opponent hook, other players, walls, freeze/death tiles, no-hook tiles, in
//! that order), the three own-velocity scalars, then the six proprioceptive values. This is every
//! number the fly's encoder reads, so a control and the fly see the same information.

use ddai_brain::Observation;
use ddai_fly::encoder::{
    Channel, OPPONENT_CHANNELS, ProprioceptionValues, RayGridConfig, RayGridFeatures, compute_proprioception_values,
};

const SPATIAL: [Channel; 7] = [
    Channel::OpponentPosition,
    Channel::OpponentApproach,
    Channel::OpponentHook,
    Channel::OtherPlayers,
    Channel::Walls,
    Channel::FreezeDeathTiles,
    Channel::NoHookTiles,
];

/// The five scalar channels about the target opponent's own state (frozen, freeze time left, velocity x/y, hook state; task 8.5a)
/// that a control may also read (task 8.8: the equal-parameter controls of the FlyGM pilot must see everything the fly's encoder
/// sees, or they are not controls). A control that reads them has an input `OPPONENT_STATE_DIM` longer; that length is all the
/// checkpoint stores, so every control checkpoint written before 8.8 (without them) loads and plays unchanged.
pub const OPPONENT_STATE_DIM: usize = OPPONENT_CHANNELS.len();

/// Length of the flattened vector for a ray-grid configuration (without the opponent-state channels).
pub fn input_dim(cfg: &RayGridConfig) -> usize {
    SPATIAL.len() * cfg.num_directions * cfg.num_distance_bins + 3 + 6
}

/// Length of the flattened vector with the opponent-state channels.
pub fn input_dim_with_opponent_state(cfg: &RayGridConfig) -> usize {
    input_dim(cfg) + OPPONENT_STATE_DIM
}

/// Whether a net with `dim` inputs reads the opponent-state channels (`Some(false)` for [`input_dim`], `Some(true)` for
/// [`input_dim_with_opponent_state`]); `None` for any other length.
pub fn reads_opponent_state(cfg: &RayGridConfig, dim: usize) -> Option<bool> {
    if dim == input_dim(cfg) {
        Some(false)
    } else if dim == input_dim_with_opponent_state(cfg) {
        Some(true)
    } else {
        None
    }
}

/// Appends the flattened features to `out` (clearing it first).
pub fn flatten_into(features: &RayGridFeatures, an: &ProprioceptionValues, out: &mut Vec<f32>) {
    flatten_into_with(features, an, false, out);
}

/// [`flatten_into`], optionally followed by the five opponent-state channels.
pub fn flatten_into_with(
    features: &RayGridFeatures,
    an: &ProprioceptionValues,
    opponent_state: bool,
    out: &mut Vec<f32>,
) {
    out.clear();
    for ch in SPATIAL {
        out.extend_from_slice(features.spatial(ch));
    }
    out.push(features.scalar(Channel::SelfVelocityX));
    out.push(features.scalar(Channel::SelfVelocityY));
    out.push(features.scalar(Channel::SelfVelocityFlow));
    out.extend_from_slice(&[
        an.grounded,
        an.airborne,
        an.own_hook,
        an.jumps_left,
        an.freeze_timer,
        an.speed,
    ]);
    if opponent_state {
        out.extend(OPPONENT_CHANNELS.iter().map(|&c| features.opponent(c)));
    }
}

/// Extracts the control input of one observation (allocation-free apart from `out`'s growth).
pub fn extract(obs: &Observation, cfg: &RayGridConfig, scratch: &mut RayGridFeatures, out: &mut Vec<f32>) {
    extract_with(obs, cfg, scratch, false, out);
}

/// [`extract`], optionally with the opponent-state channels.
pub fn extract_with(
    obs: &Observation,
    cfg: &RayGridConfig,
    scratch: &mut RayGridFeatures,
    opponent_state: bool,
    out: &mut Vec<f32>,
) {
    let an = compute_proprioception_values(&obs.self_state, cfg);
    scratch.compute(obs, cfg);
    flatten_into_with(scratch, &an, opponent_state, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattened_length_matches_input_dim() {
        let cfg = RayGridConfig::default();
        let features = RayGridFeatures::new(&cfg);
        let an = ProprioceptionValues {
            grounded: 1.0,
            airborne: 0.0,
            own_hook: 0.5,
            jumps_left: 0.5,
            freeze_timer: 0.0,
            speed: 0.2,
        };
        let mut v = Vec::new();
        flatten_into(&features, &an, &mut v);
        assert_eq!(v.len(), input_dim(&cfg));
        assert_eq!(v.len(), 7 * 48 * 4 + 9);
        assert_eq!(v[v.len() - 6..], [1.0, 0.0, 0.5, 0.5, 0.0, 0.2]);
    }

    #[test]
    fn the_opponent_state_channels_extend_the_input_and_leave_the_old_layout_alone() {
        let cfg = RayGridConfig::default();
        let mut features = RayGridFeatures::new(&cfg);
        let an = ProprioceptionValues {
            grounded: 1.0,
            airborne: 0.0,
            own_hook: 0.5,
            jumps_left: 0.5,
            freeze_timer: 0.0,
            speed: 0.2,
        };
        // A frozen target with its hook out: the five channels carry something.
        let mut me = ddai_brain::CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        let mut opp = ddai_brain::CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(420.0, 300.0);
        opp.is_frozen = true;
        opp.freeze_ticks_remaining = 150;
        opp.vel = ddai_physics::vmath::Vec2::new(5.0, -2.0);
        let obs = Observation {
            map: std::sync::Arc::new(ddai_physics::map::MapData {
                width: 20,
                height: 20,
                game: vec![Default::default(); 400],
                front: None,
                tele: None,
                speedup: None,
                switch: None,
                tune: None,
                settings: Vec::new(),
            }),
            tick: 0,
            self_state: me,
            others: vec![opp],
            target_id: Some(1),
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        features.compute(&obs, &cfg);
        let (mut plain, mut extended) = (Vec::new(), Vec::new());
        flatten_into_with(&features, &an, false, &mut plain);
        flatten_into_with(&features, &an, true, &mut extended);
        assert_eq!(plain.len(), input_dim(&cfg));
        assert_eq!(extended.len(), input_dim_with_opponent_state(&cfg));
        assert_eq!(extended.len(), plain.len() + OPPONENT_STATE_DIM);
        assert_eq!(extended[..plain.len()], plain[..], "the old layout is a prefix");
        let tail = &extended[plain.len()..];
        assert_eq!(tail[0], 1.0, "frozen");
        assert!((tail[1] - 0.5).abs() < 1e-6, "150 of 300 freeze ticks left");
        assert!(tail[2] > 0.0 && tail[3] < 0.0, "velocity x right, y up");
        assert_eq!(reads_opponent_state(&cfg, plain.len()), Some(false));
        assert_eq!(reads_opponent_state(&cfg, extended.len()), Some(true));
        assert_eq!(reads_opponent_state(&cfg, plain.len() + 1), None);
        // `extract` is the old extraction.
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut scratch = RayGridFeatures::new(&cfg);
        extract(&obs, &cfg, &mut scratch, &mut a);
        extract_with(&obs, &cfg, &mut scratch, false, &mut b);
        assert_eq!(a, b);
    }
}
