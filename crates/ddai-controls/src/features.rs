//! The controls' input vector: the fly's ray-grid features and proprioception, flattened.
//!
//! Layout: seven spatial channels of `num_directions * num_bins` values each (opponent position,
//! opponent approach, opponent hook, other players, walls, freeze/death tiles, no-hook tiles, in
//! that order), the three own-velocity scalars, then the six proprioceptive values. This is every
//! number the fly's encoder reads, so a control and the fly see the same information.

use ddai_brain::Observation;
use ddai_fly::encoder::{Channel, ProprioceptionValues, RayGridConfig, RayGridFeatures, compute_proprioception_values};

const SPATIAL: [Channel; 7] = [
    Channel::OpponentPosition,
    Channel::OpponentApproach,
    Channel::OpponentHook,
    Channel::OtherPlayers,
    Channel::Walls,
    Channel::FreezeDeathTiles,
    Channel::NoHookTiles,
];

/// Length of the flattened vector for a ray-grid configuration.
pub fn input_dim(cfg: &RayGridConfig) -> usize {
    SPATIAL.len() * cfg.num_directions * cfg.num_distance_bins + 3 + 6
}

/// Appends the flattened features to `out` (clearing it first).
pub fn flatten_into(features: &RayGridFeatures, an: &ProprioceptionValues, out: &mut Vec<f32>) {
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
}

/// Extracts the control input of one observation (allocation-free apart from `out`'s growth).
pub fn extract(obs: &Observation, cfg: &RayGridConfig, scratch: &mut RayGridFeatures, out: &mut Vec<f32>) {
    let an = compute_proprioception_values(&obs.self_state, cfg);
    scratch.compute(obs, cfg);
    flatten_into(scratch, &an, out);
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
}
