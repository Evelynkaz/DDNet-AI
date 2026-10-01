//! [`Brain`]: the shared decision-maker interface (task 7.3, acceptance criterion 1) every
//! controller implements — the planner, a scripted bot, the fly, and later human-replay — so the
//! live bot/arena code never needs to know which one is currently deciding.

use std::sync::Arc;

use ddai_physics::core::PlayerInput;
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::World;

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

/// The exact world a decision is made in (task 8.1), for brains that search: the offline arena
/// (`ddai-env`) and, later, the live bot's forward-predicted `LiveWorld` hand one of these to
/// [`Brain::decide_in`] next to the plain-data [`crate::observation::Observation`].
///
/// A view is a *borrow*: it is valid only for the duration of the call, so a brain that wants to
/// keep anything must copy it (`World::restore_from` into its own scratch world is the intended
/// pattern; the world type is `Clone` and `restore_from` reuses allocations).
///
/// Timing contract (this is what makes lag exact rather than guessed): the world is at tick
/// `world.tick` and has *not yet* stepped that tick. The brain's own client will have
/// `in_flight[k]` in effect for world step `world.tick + k` (`k < lag_ticks`); the action the
/// brain returns now takes effect on step `world.tick + lag_ticks`. With `lag_ticks == 0`,
/// `in_flight` is empty and the returned action applies to the very next step. A search brain
/// therefore rolls `world` forward `lag_ticks` steps with `in_flight` for itself before planning.
#[derive(Debug, Clone, Copy)]
pub struct WorldView<'a> {
    /// The true world the decision is made in. Every character (including dead/frozen ones) is
    /// in here, with its real held inputs in `world.characters[id].input`.
    pub world: &'a World<f32>,
    /// The deciding brain's own client id (same as [`ResetContext::self_id`]).
    pub self_id: i32,
    /// Input lag of this brain's client, in ticks.
    pub lag_ticks: u32,
    /// The brain's own inputs already sent but not yet applied, oldest first (wire format, real
    /// `fire` counters), `len() == lag_ticks`. See the timing contract above.
    pub in_flight: &'a [PlayerInput],
}

/// What the live bot knows that no world can tell a brain (task 4.1): who it must not hook and
/// where it should head. Handed over with [`Brain::set_live_context`] right before each
/// [`Brain::decide_in`]; the arena never calls it (everyone there is fair game and the target is
/// always close), so every implementer that ignores it — the default — behaves exactly as before.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiveContext<'a> {
    /// Tees that must not be caught by a hook (friends, ignored, out of the game, AFK — `spared`,
    /// `bot.ts:3009-3018`): `(position, velocity)` in pixels and pixels per tick. The planner
    /// vetoes candidate plans whose rope would catch one (`setSpareBystanders`).
    pub spares: &'a [(Vec2<f32>, Vec2<f32>)],
    /// An intermediate point to head for when the target is far or behind a wall (`pathGoal` /
    /// `trekGoal`, `setTravelGoal`); `None` (always, until task 4.2's navigation) means head for the
    /// target itself.
    pub travel_goal: Option<Vec2<f32>>,
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

    /// Like [`Brain::decide`], but additionally given the exact world the decision is made in
    /// (task 8.1): the hook for search-based brains (the planner, the D-041 hybrid) that must
    /// simulate the true physics instead of re-deriving a world from the plain-data observation.
    /// `world` is `None` when no exact world exists (e.g. a replay of recorded observations); a
    /// brain must then fall back to what [`Brain::decide`] does. **The default ignores the view
    /// and calls `decide`**, so every existing implementer (the fly, `ddnet-ai play`'s demo
    /// brains, all current tests) is unchanged and needs no edit.
    fn decide_in(
        &mut self,
        obs: &crate::observation::Observation,
        world: Option<&WorldView<'_>>,
    ) -> crate::action::Action {
        let _ = world;
        self.decide(obs)
    }

    /// Task 4.1: the live bot's extra knowledge for the next [`Brain::decide_in`] (see
    /// [`LiveContext`]). The default ignores it.
    fn set_live_context(&mut self, ctx: &LiveContext<'_>) {
        let _ = ctx;
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::observation::{CharacterObservation, Observation};
    use ddai_physics::map::{MapData, Tile};
    use ddai_physics::tuning::TuningParams;

    struct Counting {
        decides: u32,
    }
    impl Brain for Counting {
        fn reset(&mut self, _ctx: &ResetContext) {}
        fn decide(&mut self, _obs: &Observation) -> Action {
            self.decides += 1;
            Action {
                direction: 1,
                ..Action::neutral()
            }
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    fn tiny_map() -> Arc<MapData> {
        Arc::new(MapData {
            width: 2,
            height: 2,
            game: vec![Tile::default(); 4],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    fn obs(map: &Arc<MapData>) -> Observation {
        Observation {
            map: map.clone(),
            tick: 0,
            self_state: CharacterObservation::at_rest(0),
            others: Vec::new(),
            target_id: None,
            tuning: TuningParams::default(),
        }
    }

    /// The provided `decide_in` must be a pure delegation to `decide`, with or without a view:
    /// this is what keeps every pre-8.1 implementer (the fly, the demo brains) unchanged.
    #[test]
    fn default_decide_in_delegates_to_decide() {
        let map = tiny_map();
        let world = World::<f32>::from_map(&map, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let mut b = Counting { decides: 0 };
        let o = obs(&map);
        let plain = b.decide(&o);
        assert_eq!(b.decide_in(&o, None), plain);
        assert_eq!(b.decide_in(&o, Some(&view)), plain);
        assert_eq!(b.decides, 3);
    }

    #[test]
    fn idle_brain_is_neutral_through_both_entry_points() {
        let map = tiny_map();
        let mut idle = crate::IdleBrain;
        idle.reset(&ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: 1,
        });
        let o = obs(&map);
        assert_eq!(idle.decide(&o), Action::neutral());
        assert_eq!(idle.decide_in(&o, None), Action::neutral());
        assert_eq!(idle.name(), "idle");
    }
}
