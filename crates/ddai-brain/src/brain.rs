//! [`Brain`]: the shared decision-maker interface (task 7.3, acceptance criterion 1) every
//! controller implements — the planner, a scripted bot, the fly, and later human-replay — so the
//! live bot/arena code never needs to know which one is currently deciding.

use std::sync::Arc;

use ddai_physics::map::MapData;

/// What a [`Brain`] is told at the start of an episode/map (`Brain::reset`) — everything it needs
/// to set up any internal state (a fly's warm-up, a planner's search cache, ...) before the first
/// [`Brain::decide`] call, without yet knowing any character's position.
#[derive(Debug, Clone)]
pub struct ResetContext {
    /// The map this episode is played on (same handle convention as
    /// [`crate::observation::Observation::map`]).
    pub map: Arc<MapData>,
    /// This brain's own client id for the episode.
    pub self_id: i32,
    /// Seed for any internal randomness (sampled action selection, exploration noise, ...) — a
    /// brain that samples must do so reproducibly from this, not from an unseeded global RNG.
    pub seed: u64,
}

/// The shared decision-maker interface (task 7.3, acceptance criterion 1). Every implementer is
/// `&mut self` in [`Brain::decide`]: a brain is allowed to carry internal dynamical state across
/// decisions (the fly's membrane potentials; a planner's warm-started search tree) — nothing here
/// requires it to be pure.
pub trait Brain {
    /// Called once at the start of an episode (a new map, a reconnect, or any other point where
    /// this brain's internal state should not carry over from before). Implementers that keep
    /// dynamical state (the fly's `V`) reset it here — the fly's own convention is "warm up to a
    /// converged resting state", but that policy belongs to the implementer, not this trait.
    fn reset(&mut self, ctx: &ResetContext);

    /// Decides one action from one observation. May mutate internal state (a recurrent brain's
    /// hidden state; a stateful RNG for sampled action selection).
    fn decide(&mut self, obs: &crate::observation::Observation) -> crate::action::Action;

    /// A short, human-readable name for logs/telemetry (e.g. `"fly-S"`, `"planner"`,
    /// `"scripted-bot"`).
    fn name(&self) -> &str;

    /// An optional implementation-specific telemetry snapshot (e.g. the fly's per-group activity
    /// for the web panel, FLY.md §10), serialized to a JSON string by the implementer. Kept as a
    /// plain `String` (not a richer associated type) so this trait stays dependency-free — very
    /// different brains have very different telemetry shapes, and a caller that wants structure
    /// re-parses the string itself. `None` (the default) means this brain has nothing to report.
    fn telemetry(&self) -> Option<String> {
        None
    }
}
