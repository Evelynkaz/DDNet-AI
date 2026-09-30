//! `PlanWorld`/`PlanCollision`: the two traits that capture exactly what the ported planner needs
//! from a world (acceptance criterion 1) — save/restore state, apply tee state, set input/held
//! input, step, and the queries `src/plan`/`src/env` use (see `crates/ddai-tsworld/README.md`'s
//! "SimWorld API coverage" table, which this mirrors 1:1 in shape).
//!
//! Two implementations exist: [`crate::ts_adapter`] (`ddai_tsworld::SimWorld`, feature
//! `ts-parity`) proves the port is decision-for-decision identical to the real TS planner;
//! [`crate::physics_adapter`] (`ddai_physics::World<f32>`) is the production/teaching backend. All
//! positions/velocities/angles the trait hands back are already `f64` (acceptance criterion 2:
//! scoring is always `f64`, even when the backend's own physics runs in `f32`) — a `f32` backend
//! widens on read and narrows on write, once, at the trait boundary; nothing in `crate::planner`
//! ever sees an `f32`.

use crate::types::{PlayerInput, TeeState, WorldEvent};
use crate::vmath::Vec2;

/// `LineHit` (`collision.ts`'s `intersectLine`/`intersectLineHook` return shape).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineHit {
    /// `0` = no hit; otherwise a `CFLAG_*` bitmask (`CFLAG_SOLID = 1`, `CFLAG_DEATH = 2`,
    /// `CFLAG_NOHOOK = 4`).
    pub collision: i32,
    pub out_pos: Vec2,
    pub out_before_pos: Vec2,
}

pub const CFLAG_SOLID: i32 = 1;
pub const CFLAG_DEATH: i32 = 2;
pub const CFLAG_NOHOOK: i32 = 4;

/// The tile-collision queries the planner/fields/seal/shield modules need
/// (`Collision::isSolid`/`isDeath`/`isFreeze`/`isUnFreeze`/`isNoHook`/`testBox`/`intersectLine`/
/// `intersectLineHook`/`hasTele`/`teleAt`/`teleOutsFor`, plus the raw game-layer tile grid the BFS
/// fields scan). Coordinates are pixels; tile coordinates (`tile_class`) are `(x / 32, y / 32)`.
pub trait PlanCollision {
    /// A cheap, stable-for-the-lifetime-of-the-map identifier, used the way TS uses `Collision`
    /// object identity (`WeakMap<Collision, HazardField>` in `planner.ts:343-344`, and the
    /// `this.thawScratch.collision !== collision` check in `thawEscapable`,
    /// `planner.ts:1873`): [`crate::planner::Planner`] caches the BFS hazard/unfreeze fields and
    /// the `noThaw` scratch world keyed by this value, recomputing only when it changes (a new
    /// map). Two different maps must never return the same value; the same map's collision
    /// returning a stable value across calls is all correctness requires (an address, a
    /// generation counter, a hash of the tile grid -- whichever is cheapest for a given backend).
    fn identity(&self) -> u64;

    fn width(&self) -> i32;
    fn height(&self) -> i32;

    /// Raw game-layer tile id at `(tx, ty)` (row-major, `0 <= tx < width`, `0 <= ty < height`),
    /// using the DDNet numeric convention every backend already shares (`TILE_SOLID = 1`,
    /// `TILE_DEATH = 2`, `TILE_NOHOOK = 3`, `TILE_FREEZE = 9`, `TILE_UNFREEZE = 11`) — this is
    /// what `bfsField`/`hazardField`/`unfreezeField`/`travelField` (`docs/research/orig-plan.md`
    /// §1.11) scan; front-layer freeze/death is deliberately ignored (same quirk TS has, §9).
    fn game_tile(&self, tx: i32, ty: i32) -> u8;

    fn is_solid(&self, x: f64, y: f64) -> bool;
    fn is_death(&self, x: f64, y: f64) -> bool;
    fn is_freeze(&self, x: f64, y: f64) -> bool;
    fn is_un_freeze(&self, x: f64, y: f64) -> bool;
    fn is_no_hook(&self, x: f64, y: f64) -> bool;

    fn test_box(&self, pos: Vec2, size: Vec2) -> bool;

    fn intersect_line(&self, pos0: Vec2, pos1: Vec2) -> LineHit;
    fn intersect_line_hook(&self, pos0: Vec2, pos1: Vec2) -> LineHit;

    fn has_tele(&self) -> bool;
    /// `(type, number)` at the tile containing `(x, y)`; `(0, 0)` when there is no tele layer or
    /// the tile isn't a tele tile.
    fn tele_at(&self, x: f64, y: f64) -> (i32, i32);
    fn tele_outs_for(&self, number: i32) -> Vec<Vec2>;
}

/// The world API the planner needs (acceptance criterion 1). `Self: Sized` on
/// [`PlanWorld::new_scratch`] only (not the whole trait) so it stays object-safe otherwise; no
/// implementation here needs trait objects, but nothing forces the choice either way.
pub trait PlanWorld {
    type Collision: PlanCollision;
    /// Opaque saved-state snapshot (`SimState` on the TS-parity backend; whatever the production
    /// backend finds cheapest — see that adapter's doc comment). Never inspected by
    /// [`crate::planner`], only round-tripped through [`PlanWorld::save_state`]/
    /// [`PlanWorld::save_state_into`]/[`PlanWorld::restore_state`].
    type SavedState: Clone;

    fn collision(&self) -> &Self::Collision;
    /// Current world tick (`world.tick`); the planner never advances this itself, only reads it
    /// (cadence bookkeeping, `heldTicks`, `warmShift`).
    fn tick(&self) -> i64;

    fn get_tee(&self, id: i32) -> Option<TeeState>;
    /// Fills `out` in place and returns whether `id` exists (`world.readTee`) — avoids allocating
    /// a fresh `TeeState` on the hot per-tick score path.
    fn read_tee(&self, id: i32, out: &mut TeeState) -> bool;
    /// All present tees, order not meaningful (only `seal::sealed_in` calls this).
    fn all_tees(&self) -> Vec<TeeState>;

    fn set_input(&mut self, id: i32, input: PlayerInput);
    fn set_held_input(&mut self, id: i32, input: PlayerInput);
    fn step(&mut self) -> Vec<WorldEvent>;
    /// [`PlanWorld::step`] into a caller-owned buffer (cleared first), so a hot rollout loop can
    /// reuse one allocation (task 3.5). The default forwards to `step`; the production backend
    /// overrides it with an allocation-free version.
    fn step_into(&mut self, events: &mut Vec<WorldEvent>) {
        events.clear();
        events.extend(self.step());
    }

    fn save_state(&self) -> Self::SavedState;
    fn save_state_into(&self, into: &mut Self::SavedState);
    fn restore_state(&mut self, state: &Self::SavedState);

    fn apply_tee_state(&mut self, id: i32, st: &TeeState);
    fn add_tee(&mut self, id: i32, pos: Vec2);
    fn remove_tee(&mut self, id: i32);

    fn apply_force(&mut self, id: i32, force: Vec2);
    fn unfreeze(&mut self, id: i32);

    /// Builds an independent scratch world sharing this world's map/collision — the generic
    /// counterpart of TS's `thawScratch = new SimWorld(collision, {svHit:true,
    /// respawnDelayTicks:0, infiniteAmmo:true})` (`planner.ts:1873-1876`), used only by
    /// `noThaw`'s escape simulation (`crate::planner::Planner::thaw_escapable`). Cached by the
    /// caller (rebuilt only when the main world's collision identity changes), never per-decision.
    fn new_scratch(&self) -> Self
    where
        Self: Sized;
}
