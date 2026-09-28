//! [`Observation`]: everything a [`crate::Brain`] is given to decide with, for one tick — plain
//! data, `Clone`, no references into a live world (task 7.3, acceptance criterion 1), so it can be
//! produced by the server world (task 1.6), `LiveWorld` (task 2.4), or a replay reader equally
//! well, and so a recorded sequence of `Observation`s can be stashed in a training buffer without
//! borrowing from anything.
//!
//! Re-exports [`ddai_physics::core`]'s hook-state constants (`HOOK_IDLE`, `HOOK_FLYING`, ...) so a
//! caller never has to add a second dependency on `ddai-physics` just to compare
//! [`CharacterObservation::hook_state`] against them.

use std::sync::Arc;

use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use ddai_physics::vmath::Vec2;

pub use ddai_physics::core::{
    HOOK_FLYING, HOOK_GRABBED, HOOK_IDLE, HOOK_RETRACT_END, HOOK_RETRACT_START, HOOK_RETRACTED,
};

/// One character's observable state — used both for [`Observation::self_state`] and for every
/// entry of [`Observation::others`] (same fields, per the task spec: "other characters (same
/// fields + id/team)").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterObservation {
    /// This character's client id (`0..MAX_CLIENTS`, matching `ddai_physics::core::MAX_CLIENTS`).
    pub id: i32,
    /// DDRace team id (`0` = no team / solo in most block setups); kept even though block mode
    /// makes little use of teams, since the field costs nothing and other game modes do.
    pub team: i32,
    /// World pixels (DDNet's native position unit — `CharacterCore::pos`, unquantized).
    pub pos: Vec2<f32>,
    /// **Pixels per server tick** (review round 1, F4 — the orchestrator's decision): matches
    /// `ddai_physics::core::CharacterCore::vel` directly, i.e. the raw physics velocity *before*
    /// the network snapshot's `* 256` quantization (`NetCharacterCore::vel_x/vel_y`) — never
    /// pixels-per-second. A producer building this from a live `CharacterCore` copies `vel`
    /// as-is; a producer building it from a snapshot must divide the quantized `vel_x`/`vel_y` by
    /// `256.0` first. At the server's default 50 Hz tick rate, 1 px/tick = 50 px/s (DDNet's
    /// default ground run speed, `tuning.ground_control_speed()`, is `10.0` in this unit).
    pub vel: Vec2<f32>,
    /// `ddai_physics::core::CharacterCore::hook_state` — compare against the re-exported
    /// `HOOK_*` constants (`HOOK_IDLE`, `HOOK_FLYING`, `HOOK_GRABBED`, ...).
    pub hook_state: i32,
    pub hook_pos: Vec2<f32>,
    /// `-1` = not hooking anyone, matching `CharacterCore::hooked_player()`'s own convention
    /// (never an `Option` here — this mirrors the engine's own field exactly, so a caller that
    /// already knows that convention doesn't have to learn a second one).
    pub hooked_player: i32,
    /// Regular (non-deep) freeze — `CharacterCore::is_in_freeze`-ish; "the tee cannot move".
    pub is_frozen: bool,
    /// Deep freeze (`CharacterCore::deep_frozen`): can't be unfrozen by a normal unfreeze tile.
    pub is_deep_frozen: bool,
    /// Live-frozen (`CharacterCore::live_frozen`, `CHARACTERFLAG_MOVEMENTS_DISABLED`): frozen by
    /// DDRace admin/vote tooling rather than a freeze tile. Distinct from `is_frozen` because the
    /// game can combine these independently.
    pub is_live_frozen: bool,
    /// Ticks left before this character thaws on its own; `0` when not frozen at all, and left at
    /// `0` (not a sentinel) for deep freeze, which has no fixed thaw tick to report (matches
    /// `CharacterCore::freeze_end`'s own `-1` "no timer" case — see that field's doc comment).
    pub freeze_ticks_remaining: i32,
    pub jumps_left: i32,
    /// Total jumps used so far this air-time (`CharacterCore::jumped_total`), not just the current
    /// tick's jump bitmask — the feature the encoder actually wants ("how spent is this jump").
    pub jumps_used: i32,
    pub grounded: bool,
    pub weapon: i32,
    /// `-1`/`0`/`1`: last horizontal input direction (`CharacterCore::direction`).
    pub direction: i32,
}

impl CharacterObservation {
    /// A resting, ungrounded, unhooked character at the origin — a starting point a caller fills
    /// in the fields that matter for its scenario (tests, the synthetic demo) rather than repeating
    /// every field every time.
    pub fn at_rest(id: i32) -> Self {
        CharacterObservation {
            id,
            team: 0,
            pos: Vec2::new(0.0, 0.0),
            vel: Vec2::new(0.0, 0.0),
            hook_state: HOOK_IDLE,
            hook_pos: Vec2::new(0.0, 0.0),
            hooked_player: -1,
            is_frozen: false,
            is_deep_frozen: false,
            is_live_frozen: false,
            freeze_ticks_remaining: 0,
            jumps_left: 2,
            jumps_used: 0,
            grounded: false,
            weapon: 0,
            direction: 0,
        }
    }
}

/// Everything a [`crate::Brain`] sees for one decision. Plain data, `Clone`, no references into a
/// live world (task 7.3, acceptance criterion 1): [`Observation::map`] is an `Arc` handle to an
/// immutable snapshot of the map's tile classes, never a borrow of a live world's own storage, so
/// cloning an `Observation` (e.g. into a training buffer) is cheap and never fights a borrow
/// checker against whatever produced it.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Tile-class handle (task spec: "tile classes from `ddai_physics::map::MapData`"). An `Arc`
    /// because the map itself doesn't change tick to tick (only characters do), so every
    /// `Observation` of one episode/map shares one allocation instead of cloning the whole tile
    /// grid on every decision.
    pub map: Arc<MapData>,
    pub tick: i32,
    pub self_state: CharacterObservation,
    /// Every other character currently visible/relevant (task spec: "other characters"). Order is
    /// not meaningful (a `Brain` that cares about a specific one finds it by `id`).
    pub others: Vec<CharacterObservation>,
    /// Which of [`Observation::others`] the bot's own target selection (outside this crate —
    /// whatever policy picks who to chase/hook) has chosen, by `id` — review round 1, F13
    /// (CONFIRMED): an earlier revision had every consumer hardcode `others[0]` as "the"
    /// opponent while this struct's own doc comment said ordering carries no meaning, so a
    /// producer that returned `others` in, say, distance order would silently feed the wrong
    /// character into the encoder the moment more than one was ever present. `None` means "no
    /// explicit selection" — a consumer that needs exactly one opponent (`crate::demo_brain`'s
    /// synthetic scenarios always have exactly one, so this is `None` there) falls back to the
    /// nearest one by distance; see [`Observation::target_or_nearest`]. (`CharacterObservation`
    /// carries no "alive"/"dead" flag at all — block mode has no kill state to filter on — so
    /// "nearest", not "nearest alive", is the actual fallback.)
    pub target_id: Option<i32>,
    /// The tuning subset needed for features (task spec) — kept as the engine's own
    /// [`TuningParams`] wholesale rather than a hand-picked slimmer struct: it is already small
    /// (47 `f32`-sized fields, `Copy`) and this way a new feature that needs one more tuning field
    /// never has to touch this struct's shape again.
    pub tuning: TuningParams,
}

impl Observation {
    /// Finds `id` among [`Observation::others`] (never [`Observation::self_state`] — a caller
    /// asking "where is player N" should get `None` for their own id, not silently the self state
    /// under someone else's id).
    pub fn other_by_id(&self, id: i32) -> Option<&CharacterObservation> {
        self.others.iter().find(|c| c.id == id)
    }

    /// [`Observation::target_id`] resolved to an actual [`CharacterObservation`] (review round 1,
    /// F13): the explicitly-selected target if it's `Some` and still present in [`Observation::
    /// others`] (a stale id from a target that has since disconnected/left falls through to the
    /// same distance-based fallback as `None`, rather than returning nothing), otherwise the
    /// nearest entry in `others` by straight-line distance to [`Observation::self_state`]'s own
    /// `pos`. `None` only when `others` is empty.
    pub fn target_or_nearest(&self) -> Option<&CharacterObservation> {
        if let Some(id) = self.target_id
            && let Some(target) = self.other_by_id(id)
        {
            return Some(target);
        }
        self.others.iter().min_by(|a, b| {
            let da = ddai_physics::vmath::length_squared(a.pos - self.self_state.pos);
            let db = ddai_physics::vmath::length_squared(b.pos - self.self_state.pos);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_map() -> MapData {
        MapData {
            width: 2,
            height: 2,
            game: vec![Default::default(); 4],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn other_by_id_finds_the_right_character_and_excludes_self() {
        let obs = Observation {
            map: Arc::new(tiny_map()),
            tick: 0,
            self_state: CharacterObservation::at_rest(0),
            others: vec![CharacterObservation::at_rest(1), CharacterObservation::at_rest(2)],
            target_id: None,
            tuning: TuningParams::default(),
        };
        assert_eq!(obs.other_by_id(2).map(|c| c.id), Some(2));
        assert_eq!(obs.other_by_id(0), None, "self's own id must not be found among others");
        assert_eq!(obs.other_by_id(99), None);
    }

    #[test]
    fn observation_clone_shares_the_map_allocation() {
        let obs = Observation {
            map: Arc::new(tiny_map()),
            tick: 5,
            self_state: CharacterObservation::at_rest(0),
            others: Vec::new(),
            target_id: None,
            tuning: TuningParams::default(),
        };
        let cloned = obs.clone();
        assert!(Arc::ptr_eq(&obs.map, &cloned.map));
    }

    #[test]
    fn target_or_nearest_prefers_the_explicit_target_id() {
        let mut near = CharacterObservation::at_rest(1);
        near.pos = Vec2::new(10.0, 0.0);
        let mut far = CharacterObservation::at_rest(2);
        far.pos = Vec2::new(1000.0, 0.0);
        let obs = Observation {
            map: Arc::new(tiny_map()),
            tick: 0,
            self_state: CharacterObservation::at_rest(0),
            others: vec![near, far],
            target_id: Some(2), // the *farther* one, on purpose.
            tuning: TuningParams::default(),
        };
        assert_eq!(obs.target_or_nearest().map(|c| c.id), Some(2));
    }

    #[test]
    fn target_or_nearest_falls_back_to_nearest_when_target_id_is_none_or_stale() {
        let mut near = CharacterObservation::at_rest(1);
        near.pos = Vec2::new(10.0, 0.0);
        let mut far = CharacterObservation::at_rest(2);
        far.pos = Vec2::new(1000.0, 0.0);

        let obs_none = Observation {
            map: Arc::new(tiny_map()),
            tick: 0,
            self_state: CharacterObservation::at_rest(0),
            others: vec![near, far],
            target_id: None,
            tuning: TuningParams::default(),
        };
        assert_eq!(obs_none.target_or_nearest().map(|c| c.id), Some(1));

        let obs_stale = Observation {
            target_id: Some(99), // disconnected/left -- not in `others` any more.
            ..obs_none.clone()
        };
        assert_eq!(obs_stale.target_or_nearest().map(|c| c.id), Some(1));
    }

    #[test]
    fn target_or_nearest_is_none_when_others_is_empty() {
        let obs = Observation {
            map: Arc::new(tiny_map()),
            tick: 0,
            self_state: CharacterObservation::at_rest(0),
            others: Vec::new(),
            target_id: Some(5),
            tuning: TuningParams::default(),
        };
        assert_eq!(obs.target_or_nearest(), None);
    }
}
