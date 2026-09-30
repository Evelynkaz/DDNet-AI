//! `PlanWorld`/`PlanCollision` for `ddai_physics::World<f32>` -- the production/teaching backend
//! (D-041): the fly proposes, this search decides on real physics. Unlike [`crate::ts_adapter`],
//! **bit parity with anything is not the goal here** -- `ddai_physics::World<f32>` is already
//! bit-exact with the real DDNet server (task 1.6); this adapter's only job is a *correct enough*
//! translation between the planner's plain data types and DDNet's own richer state model, fast
//! enough for the deadline mode's latency budget (D-041: whole live decision <= 5 ms p99).
//!
//! **Known simplifications (documented, not silently swept under the rug -- see the build report
//! for the honest list):**
//! - Review round 2, F15: this bullet used to say [`PhysicsWorld::step`] does not derive
//!   `WorldEvent::HammerHit`/`HammerFire` at all -- stale since review round 1, F9; it derives
//!   both from an observable-state heuristic (`step`'s own doc comment has the exact rule:
//!   `attack_tick == tick` for a fire signal, `reload_timer` against `hammer_fire_delay_ms`/
//!   `hammer_hit_fire_delay_ms` to tell a hit from a miss, nearest present tee within the hammer's
//!   own hit radius for "hit whom"). DDNet's own `World::step` still has no built-in event/
//!   callback list the way `ddai-tsworld::SimWorld::step` does (those TS events come from a
//!   bespoke check `world.ts` performs inline), so this is a derivation from observable state,
//!   not a passthrough of a real event list -- not tested against real hammer swings by this
//!   crate's own tests (see the build report's "not verified" list).
//! - `set_held_input` behaves like `set_input` (does not additionally seed DDNet's
//!   `prev_input`/`latest_prev_input` edge-detection state the way TS's `setHeldInput` does) --
//!   acceptable here because nothing in the ported planner's production path relies on a
//!   held-input's *edge* (only `direction`/`jump`/`hook`/aim levels, which this does carry).
//! - `TeeState::ddnet_flags`/`since_attack` are always `None` (nothing in this crate's production
//!   path reads them back through `apply_tee_state`; both are optional fields precisely because
//!   TS itself treats a missing value as "leave the current value alone").

use crate::plan_world::{CFLAG_DEATH, CFLAG_NOHOOK, CFLAG_SOLID, LineHit, PlanCollision, PlanWorld};
use crate::types::{PlayerInput, TeeState, WorldEvent};
use crate::vmath::Vec2;
use ddai_physics::core::MAX_CLIENTS;
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2 as PVec2;
use ddai_physics::world::{TickInput, World};
use std::sync::Arc;

type Collision32 = ddai_physics::collision::Collision<f32>;

fn to_p_vec2(v: Vec2) -> PVec2<f32> {
    PVec2::new(v.x as f32, v.y as f32)
}

fn from_p_vec2(v: PVec2<f32>) -> Vec2 {
    Vec2 {
        x: f64::from(v.x),
        y: f64::from(v.y),
    }
}

/// `computeJumpsLeft` (mirrors `ddai-tsworld`'s own helper of the same shape,
/// `crates/ddai-tsworld/src/world.rs:215-228`, against DDNet's real `CharacterCore`/`Collision`
/// instead of TS's).
fn compute_jumps_left(core: &ddai_physics::core::CharacterCore<f32>, col: &Collision32) -> i32 {
    if core.jumps <= 0 {
        return 0;
    }
    let half = 14.0f32;
    let grounded = col.check_point(core.pos.x + half, core.pos.y + half + 5.0)
        || col.check_point(core.pos.x - half, core.pos.y + half + 5.0);
    if grounded {
        return core.jumps;
    }
    if core.jumped & 2 != 0 {
        return 0;
    }
    (core.jumps - 1 - core.jumped_total).max(0)
}

/// DDNet's `LineHit::hit` is a raw tile id (`TILE_SOLID`/`TILE_NOHOOK`/`TILE_DEATH`/...); this
/// crate's [`crate::plan_world::LineHit::collision`] follows TS's `CFLAG_*` bitmask convention
/// instead (matching `ts_adapter`) -- this is the translation between the two.
fn tile_to_cflags(hit: i32) -> i32 {
    if hit == i32::from(ddai_physics::map::TILE_SOLID) {
        CFLAG_SOLID
    } else if hit == i32::from(ddai_physics::map::TILE_NOHOOK) {
        CFLAG_SOLID | CFLAG_NOHOOK
    } else if hit == i32::from(ddai_physics::map::TILE_DEATH) {
        CFLAG_DEATH
    } else if hit != 0 {
        CFLAG_SOLID
    } else {
        0
    }
}

/// Raw front-layer tile id at `(tx, ty)` (`0` off-map or on a map with no front layer) -- review
/// round 1, F9's front-layer freeze/unfreeze fix. A free function, not an inherent method on
/// `Collision32`, since that type is foreign to this crate (the orphan rule forbids adding
/// inherent methods to it directly).
fn front_tile(col: &Collision32, tx: i32, ty: i32) -> u8 {
    if tx < 0 || ty < 0 || tx >= col.width() || ty >= col.height() {
        return 0;
    }
    col.get_front_index(tx, ty).clamp(0, 255) as u8
}

impl PlanCollision for Collision32 {
    fn identity(&self) -> u64 {
        std::ptr::from_ref(self) as u64
    }
    fn width(&self) -> i32 {
        Collision32::width(self)
    }
    fn height(&self) -> i32 {
        Collision32::height(self)
    }
    fn game_tile(&self, tx: i32, ty: i32) -> u8 {
        if tx < 0 || ty < 0 || tx >= self.width() || ty >= self.height() {
            return 0;
        }
        self.get_index(tx, ty).clamp(0, 255) as u8
    }
    fn is_solid(&self, x: f64, y: f64) -> bool {
        self.check_point(x as f32, y as f32)
    }
    fn is_death(&self, x: f64, y: f64) -> bool {
        let tx = (x / 32.0).floor() as i32;
        let ty = (y / 32.0).floor() as i32;
        self.game_tile(tx, ty) == ddai_physics::map::TILE_DEATH
    }
    /// Review round 1, F9: real DDNet freezes on a front-layer `TILE_FREEZE` too, not just the
    /// game layer (`world.rs::handle_tiles`: `tile_index == TILE_FREEZE || tile_findex ==
    /// TILE_FREEZE`; BlmapChill alone has 161 front-layer freeze tiles). TS's own `isFreeze`
    /// (`collision.ts`) only ever looks at the game layer -- a real TS quirk, deliberately
    /// reproduced on the parity path (`ts_adapter`'s `Collision` impl checks the game layer only,
    /// matching TS bit-for-bit). This backend never claims TS parity (see the module doc comment),
    /// so its own hazard heuristics see the real hazard instead of TS's blind spot -- this *is*
    /// "behind the production flag" in the sense the constraint asks for: the fix lives only in
    /// this production-only file, the parity path (`ts_adapter.rs`) is untouched.
    fn is_freeze(&self, x: f64, y: f64) -> bool {
        let tx = (x / 32.0).floor() as i32;
        let ty = (y / 32.0).floor() as i32;
        self.game_tile(tx, ty) == ddai_physics::map::TILE_FREEZE
            || front_tile(self, tx, ty) == ddai_physics::map::TILE_FREEZE
    }
    /// See [`Collision32::is_freeze`]'s doc comment (review round 1, F9) -- same reasoning for
    /// `TILE_UNFREEZE`.
    fn is_un_freeze(&self, x: f64, y: f64) -> bool {
        let tx = (x / 32.0).floor() as i32;
        let ty = (y / 32.0).floor() as i32;
        self.game_tile(tx, ty) == ddai_physics::map::TILE_UNFREEZE
            || front_tile(self, tx, ty) == ddai_physics::map::TILE_UNFREEZE
    }
    fn is_no_hook(&self, x: f64, y: f64) -> bool {
        let tx = (x / 32.0).floor() as i32;
        let ty = (y / 32.0).floor() as i32;
        self.game_tile(tx, ty) == ddai_physics::map::TILE_NOHOOK
    }
    fn test_box(&self, pos: Vec2, size: Vec2) -> bool {
        Collision32::test_box(self, to_p_vec2(pos), to_p_vec2(size))
    }
    /// Review round 1, F7: `ddai_physics::Collision::intersect_line` (`collision.rs:607`) is the
    /// exact analogue of TS's plain `intersectLine` (`collision.ts:553-581`) -- a bare raymarch
    /// that stops only at `isSolid`/solid tiles, no laser/through-tile logic at all -- so it's used
    /// directly here instead of the laser-shaped `intersect_no_laser` an earlier revision reached
    /// for (which additionally stops at `TILE_NOLASER`/front-`TILE_NOLASER`, tiles TS's own
    /// `intersectLine` does not know about).
    fn intersect_line(&self, pos0: Vec2, pos1: Vec2) -> LineHit {
        let h = Collision32::intersect_line(self, to_p_vec2(pos0), to_p_vec2(pos1));
        LineHit {
            collision: tile_to_cflags(h.hit),
            out_pos: from_p_vec2(h.collision),
            out_before_pos: from_p_vec2(h.before_collision),
        }
    }
    fn intersect_line_hook(&self, pos0: Vec2, pos1: Vec2) -> LineHit {
        let h = Collision32::intersect_line_tele_hook(self, to_p_vec2(pos0), to_p_vec2(pos1), false);
        LineHit {
            collision: tile_to_cflags(h.hit),
            out_pos: from_p_vec2(h.collision),
            out_before_pos: from_p_vec2(h.before_collision),
        }
    }
    fn has_tele(&self) -> bool {
        self.has_hook_tele_ins(false) || (1..=255u8).any(|n| !self.tele_outs(n).is_empty())
    }
    fn tele_at(&self, x: f64, y: f64) -> (i32, i32) {
        let index = self.get_pure_map_index(x as f32, y as f32);
        let kind = self.is_teleport(index as i32);
        if kind != 0 {
            return (kind, kind);
        }
        (0, 0)
    }
    fn tele_outs_for(&self, number: i32) -> Vec<Vec2> {
        if !(0..=255).contains(&number) {
            return Vec::new();
        }
        self.tele_outs(number as u8).iter().map(|&v| from_p_vec2(v)).collect()
    }
}

const _: () = {
    assert!(CFLAG_SOLID == 1);
    assert!(CFLAG_DEATH == 2);
    assert!(CFLAG_NOHOOK == 4);
};

fn to_ddnet_input(i: &PlayerInput) -> ddai_physics::core::PlayerInput {
    ddai_physics::core::PlayerInput {
        direction: i.direction,
        target_x: i.target_x.round() as i32,
        target_y: i.target_y.round() as i32,
        jump: i.jump,
        fire: i.fire,
        hook: i.hook,
        player_flags: i.player_flags,
        wanted_weapon: i.wanted_weapon,
        next_weapon: i.next_weapon,
        prev_weapon: i.prev_weapon,
    }
}

/// Inverse of [`to_ddnet_input`]: the wire-format input a `World<f32>` character holds, as the
/// planner's plain [`PlayerInput`] (task 8.1: seeding a planning world from a live/arena world).
pub fn from_ddnet_input(i: &ddai_physics::core::PlayerInput) -> PlayerInput {
    PlayerInput {
        direction: i.direction,
        target_x: f64::from(i.target_x),
        target_y: f64::from(i.target_y),
        jump: i.jump,
        fire: i.fire,
        hook: i.hook,
        player_flags: i.player_flags,
        wanted_weapon: i.wanted_weapon,
        next_weapon: i.next_weapon,
        prev_weapon: i.prev_weapon,
    }
}

/// [`PlanWorld::SavedState`] for [`PhysicsWorld`]: a full `World<f32>` clone (cheap -- `O(character
/// count)`, not `O(map size)`, per that type's own doc comment) plus this wrapper's own pending-
/// input/present bookkeeping, which `World` itself knows nothing about.
#[derive(Clone)]
pub struct PhysicsSavedState {
    world: World<f32>,
    pending_input: Box<[PlayerInput; MAX_CLIENTS]>,
    present: Box<[bool; MAX_CLIENTS]>,
    last_known: Box<[Option<TeeState>; MAX_CLIENTS]>,
}

impl PhysicsSavedState {
    /// Overwrites this snapshot with `other` reusing its buffers (`World::restore_from`); the
    /// derived `clone_from` would build and drop a whole fresh world (~105 kB) instead. Task 3.5:
    /// the hybrid search copies one decision snapshot per worker per decision.
    pub fn assign_from(&mut self, other: &PhysicsSavedState) {
        self.world.restore_from(&other.world);
        self.pending_input.clone_from(&other.pending_input);
        self.present.clone_from(&other.present);
        self.last_known.clone_from(&other.last_known);
    }
}

/// Reused per-tick scratch of [`PhysicsWorld::step`] (never part of a saved state).
#[derive(Default)]
struct StepScratch {
    ids: Vec<i32>,
    alive_before: Vec<bool>,
    fire_before: Vec<i32>,
    pos_before: Vec<Option<PVec2<f32>>>,
    inputs: Vec<TickInput>,
    victims: Vec<i32>,
}

/// [`PlanWorld`] over `ddai_physics::world::World<f32>`. Owns the map handle (`Arc<MapData>`) so
/// [`PlanWorld::new_scratch`] can build an independent fresh world sharing the same map without
/// this crate needing to reconstruct one from just a `Collision` (see the module doc comment).
pub struct PhysicsWorld {
    world: World<f32>,
    map: Arc<MapData>,
    pending_input: Box<[PlayerInput; MAX_CLIENTS]>,
    present: Box<[bool; MAX_CLIENTS]>,
    /// Review round 1, finding F2: real DDNet physics (`ddai_physics::world::die`) fully removes
    /// a dead character's core from `WorldCore` (`world.cores.remove`), so `get_tee` on a
    /// just-died id would otherwise start returning `None` -- but TS's `SimWorld` tee record
    /// never disappears on death, only `alive` flips to `false` (`world.ts`'s `tees` `Map` keeps
    /// every entry forever; `scoreTick`'s `!en.alive -> +15`/`!me.alive -> -15` terms, and every
    /// other read of a dead tee's last position, depend on this). This is the last full live
    /// snapshot taken for each id (updated every time [`PlanWorld::get_tee`] finds a live core),
    /// returned (with `alive` forced `false`) once the core disappears -- see `get_tee`'s doc
    /// comment on this field for the exact contract.
    last_known: Box<[Option<TeeState>; MAX_CLIENTS]>,
    step_scratch: StepScratch,
}

fn neutral_input_array() -> Box<[PlayerInput; MAX_CLIENTS]> {
    Box::new([crate::types::empty_input(); MAX_CLIENTS])
}

fn no_last_known() -> Box<[Option<TeeState>; MAX_CLIENTS]> {
    Box::new([None; MAX_CLIENTS])
}

impl PhysicsWorld {
    /// Builds a fresh world from a map (`World::from_map` + `World::init`, matching how every
    /// other consumer of `ddai_physics::World` sets one up). `seed`: the core PRNG seed
    /// (`World::from_map`'s own doc comment).
    pub fn new(map: Arc<MapData>, seed: u64) -> Self {
        let mut world = World::from_map(&map, seed);
        let _ = world.init(std::iter::empty::<&str>());
        PhysicsWorld {
            world,
            map,
            pending_input: neutral_input_array(),
            present: Box::new([false; MAX_CLIENTS]),
            last_known: no_last_known(),
            step_scratch: StepScratch::default(),
        }
    }

    /// Wraps an already-built `World<f32>` (task 8.1): the arena builds one template world per
    /// map (`World::from_map` + `init` is milliseconds on a real block map) and clones it per
    /// game, so games must not pay the map scan again. No tee is registered as present -- add
    /// them with [`PlanWorld::add_tee`], or seed everything from another world with
    /// [`PhysicsWorld::sync_from`].
    pub fn from_world(world: World<f32>, map: Arc<MapData>) -> Self {
        PhysicsWorld {
            world,
            map,
            pending_input: neutral_input_array(),
            present: Box::new([false; MAX_CLIENTS]),
            last_known: no_last_known(),
            step_scratch: StepScratch::default(),
        }
    }

    /// Makes this planning world an exact copy of `src` (task 8.1, `Brain::decide_in`): the
    /// physics state via `World::restore_from` (allocation-free once warm; `src` must be built
    /// on the same map), and the wrapper's own bookkeeping from it -- every character that still
    /// has a core is present, its held input is what `src` last applied to it, and the dead-tee
    /// fallback cache is refreshed. A character `src` has already killed is *not* present (its
    /// [`PlanWorld::get_tee`] is `None`), which every planner entry point reads as "no such
    /// tee".
    pub fn sync_from(&mut self, src: &World<f32>) {
        self.world.restore_from(src);
        for id in 0..MAX_CLIENTS {
            let alive_core = self.world.cores.slot_of(id as u8).is_some();
            self.present[id] = alive_core;
            self.pending_input[id] = match (alive_core, self.world.characters[id].as_ref()) {
                (true, Some(character)) => from_ddnet_input(&character.input),
                _ => crate::types::empty_input(),
            };
            if alive_core {
                self.refresh_cache(id as i32);
            } else {
                self.last_known[id] = None;
            }
        }
    }

    /// The map this world was built on.
    pub fn map(&self) -> &Arc<MapData> {
        &self.map
    }

    /// Read-only access to the underlying `World<f32>` -- for callers that need DDNet-specific
    /// state this crate's [`PlanWorld`] surface deliberately doesn't expose (telemetry, a live
    /// bot's own snapshot sync). Not part of the [`PlanWorld`] trait itself.
    pub fn inner(&self) -> &World<f32> {
        &self.world
    }

    /// Mutable access to the underlying `World<f32>`, for the same reason as [`PhysicsWorld::inner`]
    /// (e.g. a live caller applying a server snapshot via `ddai_physics`'s own state-sync helpers
    /// before handing control to the planner).
    pub fn inner_mut(&mut self) -> &mut World<f32> {
        &mut self.world
    }

    /// The live `TeeState` for `id` if (and only if) DDNet's own physics still has a core for it
    /// right now -- `None` once real DDNet has removed it (death; review round 1, F2), regardless
    /// of whether this crate's own [`PhysicsWorld::last_known`]/`present` bookkeeping still
    /// considers `id` tracked. Pure read, no caching side effect (see [`PhysicsWorld::refresh_cache`]
    /// for the `&mut self` callers that actually populate the cache from this).
    fn snapshot_live(&self, id: i32) -> Option<TeeState> {
        if !(0..MAX_CLIENTS as i32).contains(&id) {
            return None;
        }
        let slot = self.world.cores.slot_of(id as u8)?;
        let core = self.world.cores.core_at(slot);
        let character = self.world.characters[id as usize].as_ref()?;
        let jumps_left = compute_jumps_left(core, &self.world.collision);
        Some(TeeState {
            id,
            alive: character.alive,
            pos: from_p_vec2(core.pos),
            vel: from_p_vec2(core.vel),
            hook_state: core.hook_state,
            hook_pos: from_p_vec2(core.hook_pos),
            hook_dir: from_p_vec2(core.hook_dir),
            hooked_player: core.hooked_player(),
            jumped: core.jumped,
            jumps_left,
            direction: core.direction,
            angle: f64::from(core.angle),
            active_weapon: core.active_weapon,
            frozen: character.freeze_time > 0,
            freeze_ticks_left: i64::from(character.freeze_time),
            attack_tick: i64::from(character.attack_tick),
            hook_tick: Some(i64::from(core.hook_tick)),
            jumped_total: Some(core.jumped_total),
            reload_ticks: Some(i64::from(character.reload_timer)),
            frozen_for: (character.freeze_time > 0).then(|| i64::from(self.world.tick - core.freeze_start)),
            deep_frozen: Some(core.deep_frozen),
            jumps: Some(core.jumps),
            ddnet_flags: None,
            since_attack: None,
        })
    }

    /// Updates [`PhysicsWorld::last_known`]`[id]` from [`PhysicsWorld::snapshot_live`] if (and
    /// only if) the core is currently live -- a no-op once the core is gone, which is exactly
    /// right: that's what leaves the *previous* (pre-death) snapshot in place for `get_tee`'s
    /// fallback (review round 1, F2) to keep returning.
    fn refresh_cache(&mut self, id: i32) {
        if let Some(live) = self.snapshot_live(id) {
            self.last_known[id as usize] = Some(live);
        }
    }
}

impl PlanWorld for PhysicsWorld {
    type Collision = Collision32;
    type SavedState = PhysicsSavedState;

    fn collision(&self) -> &Self::Collision {
        &self.world.collision
    }
    fn tick(&self) -> i64 {
        i64::from(self.world.tick)
    }

    /// See [`PhysicsWorld::last_known`]'s doc comment (review round 1, F2): `id` stays visible
    /// (with `alive: false`) forever once added, exactly like TS, even after real DDNet physics
    /// has fully removed its core -- only [`PlanWorld::remove_tee`] makes `get_tee` return `None`
    /// again. `last_known` itself is kept fresh by the `&mut self` methods that can actually
    /// observe a live core (`step`/`apply_tee_state`/`add_tee`, via the private
    /// [`PhysicsWorld::snapshot_live`] + [`PhysicsWorld::refresh_cache`]), not by this method --
    /// `get_tee` takes `&self` (matching the trait) and never needs interior mutability for that.
    fn get_tee(&self, id: i32) -> Option<TeeState> {
        if !(0..MAX_CLIENTS as i32).contains(&id) || !self.present[id as usize] {
            return None;
        }
        match self.snapshot_live(id) {
            Some(live) => Some(live),
            None => {
                // The core is gone (real DDNet removed it on death, F2's actual bug): fall back
                // to the last live snapshot, forcing only `alive` -- every other field (position,
                // velocity, `frozen`/`freeze_ticks_left`, ...) stays exactly as it was the tick
                // before death, matching TS (`die()` itself never touches `freeze_time`, so a tee
                // that happened to be frozen right when something else killed it stays reported as
                // frozen too, same as TS's own tee record would).
                let mut dead = self.last_known[id as usize]?;
                dead.alive = false;
                Some(dead)
            }
        }
    }

    fn read_tee(&self, id: i32, out: &mut TeeState) -> bool {
        match self.get_tee(id) {
            Some(t) => {
                *out = t;
                true
            }
            None => false,
        }
    }

    fn all_tees(&self) -> Vec<TeeState> {
        (0..MAX_CLIENTS as i32).filter_map(|id| self.get_tee(id)).collect()
    }

    fn set_input(&mut self, id: i32, input: PlayerInput) {
        if (0..MAX_CLIENTS as i32).contains(&id) {
            self.pending_input[id as usize] = input;
        }
    }
    fn set_held_input(&mut self, id: i32, input: PlayerInput) {
        self.set_input(id, input);
    }

    /// Review round 1, F9: derives `WorldEvent::HammerFire`/`HammerHit` from observable
    /// `World<f32>` state (it has no built-in event list -- see the module doc comment), on top
    /// of the existing `Death` detection. **Derivation, production-path-only**:
    /// `fire_weapon` (`world.rs`) sets `character.attack_tick` on *every* fire attempt
    /// (weapon-agnostic) and, only for the hammer, leaves `character.reload_timer` at one of two
    /// distinct values depending on whether it hit anything: `hammer_fire_delay_ms` (miss) or the
    /// much longer `hammer_hit_fire_delay_ms` (hit). A fire attempt with the hammer in hand whose
    /// `reload_timer` lands past the midpoint of those two delays is a hit.
    ///
    /// Task 8.1 (arena review of the derivation, four fixes): (0) a fire attempt is detected by
    /// `attack_tick` *changing*: comparing it with the post-step tick counter (as this used to)
    /// only ever matched swings made from the held-button path, never a fresh press, i.e. almost
    /// no real hammer blow was seen at all; (1) the weapon is read *after* the
    /// step (the weapon that actually fired; a switch requested in the same input only takes effect
    /// one input later, so a gun shot is never mistaken for a hammer blow); (2) "whom" is
    /// resolved with *pre-step* positions -- a swing is resolved before the character ticks, and
    /// the victim has already been thrown out of reach by the time `step` returns (post-step
    /// positions missed real hits at 45 px); (3) it uses `fire_hammer`'s own geometry -- everybody
    /// within 14 + 28 px of the point `proximity * 0.75` in front of the swinger along its aim --
    /// and reports every victim, not only the nearest. A proposal for a real event hook in
    /// `World::step` (which would make this derivation unnecessary) is in the task-8.1 report.
    fn step(&mut self) -> Vec<WorldEvent> {
        let mut events = Vec::new();
        self.step_into(&mut events);
        events
    }

    fn step_into(&mut self, events: &mut Vec<WorldEvent>) {
        events.clear();
        // Task 3.5: the per-tick scratch lives in the world and is reused (five fresh `Vec`s per
        // tick before) -- a rollout is dozens of ticks and the hybrid search runs dozens of
        // rollouts per decision, so the allocations were a measurable share of the cost and made
        // the worker rollouts impossible to keep allocation-free.
        let mut sc = std::mem::take(&mut self.step_scratch);
        sc.ids.clear();
        sc.alive_before.clear();
        sc.fire_before.clear();
        sc.pos_before.clear();
        sc.inputs.clear();
        for id in 0..MAX_CLIENTS {
            if self.present[id] {
                sc.ids.push(id as i32);
            }
        }
        for &id in &sc.ids {
            sc.alive_before
                .push(self.world.characters[id as usize].as_ref().is_some_and(|c| c.alive));
            sc.fire_before
                .push(self.world.characters[id as usize].map_or(-1, |c| c.attack_tick));
            // Task 8.1: positions *before* the step. A hammer swing is resolved in the
            // direct-input phase, ahead of the character ticks, so the swinger's origin and every
            // candidate victim's position are exactly these -- a hit victim has already been
            // thrown away from where the swing found it by the time `step` returns.
            sc.pos_before.push(
                self.world
                    .cores
                    .slot_of(id as u8)
                    .map(|slot| self.world.cores.core_at(slot).pos),
            );
            sc.inputs.push(TickInput {
                id: id as u8,
                input: to_ddnet_input(&self.pending_input[id as usize]),
                kill: false,
            });
        }
        self.world.step(&sc.inputs);
        let (ids, alive_before, fire_before, pos_before) = (&sc.ids, &sc.alive_before, &sc.fire_before, &sc.pos_before);
        for (i, &id) in ids.iter().enumerate() {
            let now_alive = self.world.characters[id as usize].as_ref().is_some_and(|c| c.alive);
            if alive_before[i] && !now_alive {
                events.push(WorldEvent::Death { id, by: -1 });
            }
            // `attack_tick` is written only by `fire_weapon`, so a changed value means a fire
            // attempt in this step. It cannot be compared with the tick counter: the direct-input
            // fire (a fresh press) runs *before* the counter increments, the `handle_weapons`
            // fire (held button, reload done) after. The weapon is the one that fired (read after
            // the step).
            let fired_this_tick = self.world.characters[id as usize].is_some_and(|c| c.attack_tick != fire_before[i]);
            if !fired_this_tick {
                continue;
            }
            let Some(slot) = self.world.cores.slot_of(id as u8) else {
                continue;
            };
            let core = self.world.cores.core_at(slot);
            if core.active_weapon != ddai_physics::core::WEAPON_HAMMER {
                continue;
            }
            let miss_ticks = core.tuning.hammer_fire_delay_ms() / 1000.0 * ddai_physics::core::SERVER_TICK_SPEED as f32;
            let hit_ticks =
                core.tuning.hammer_hit_fire_delay_ms() / 1000.0 * ddai_physics::core::SERVER_TICK_SPEED as f32;
            let character = self.world.characters[id as usize].as_ref();
            let reload = character.map_or(0, |c| c.reload_timer);
            let hit = reload as f32 > (miss_ticks + hit_ticks) / 2.0;
            let mut victims = std::mem::take(&mut sc.victims);
            victims.clear();
            if hit && let (Some(origin), Some(character)) = (pos_before[i], character) {
                // `fire_hammer`'s own search: everybody within `proximity/2 + proximity` (14 + 28)
                // of the swing's start point, `proximity * 0.75` in front of the swinger along the
                // aim (`character.cpp:520-526`), using pre-step positions.
                let aim = PVec2::new(
                    character.latest_input.target_x as f32,
                    character.latest_input.target_y as f32,
                );
                let aim_len = (aim.x * aim.x + aim.y * aim.y).sqrt();
                let dir = if aim_len > 0.0 {
                    PVec2::new(aim.x / aim_len, aim.y / aim_len)
                } else {
                    PVec2::new(0.0, 0.0)
                };
                let start = PVec2::new(origin.x + dir.x * 28.0 * 0.75, origin.y + dir.y * 28.0 * 0.75);
                let reach = 28.0 * 0.5 + 28.0;
                for (j, &other) in ids.iter().enumerate() {
                    if other == id {
                        continue;
                    }
                    if let Some(p) = pos_before[j]
                        && alive_before[j]
                        && ((p.x - start.x).powi(2) + (p.y - start.y).powi(2)).sqrt() < reach
                    {
                        victims.push(other);
                    }
                }
                if victims.is_empty() {
                    // The reload says it hit but the geometry found nobody (an aim edge case):
                    // credit the nearest tee rather than lose the hit.
                    let nearest = ids
                        .iter()
                        .enumerate()
                        .filter(|&(j, &other)| other != id && alive_before[j])
                        .filter_map(|(j, &other)| {
                            pos_before[j].map(|p| (other, (p.x - origin.x).hypot(p.y - origin.y)))
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1));
                    victims.extend(nearest.map(|(other, _)| other));
                }
            }
            events.push(WorldEvent::HammerFire {
                from: id,
                hits: victims.len().max(usize::from(hit)) as i32,
            });
            for &to in &victims {
                events.push(WorldEvent::HammerHit { from: id, to });
            }
            sc.victims = victims;
        }
        for &id in ids {
            self.refresh_cache(id);
        }
        self.step_scratch = sc;
    }

    fn save_state(&self) -> Self::SavedState {
        PhysicsSavedState {
            world: self.world.clone(),
            pending_input: self.pending_input.clone(),
            present: self.present.clone(),
            last_known: self.last_known.clone(),
        }
    }
    fn save_state_into(&self, into: &mut Self::SavedState) {
        // `World::restore_from` (not the derived `clone_from`, which builds a fresh clone and drops
        // the old one): field-wise copy into the existing buffers, no allocation once warm.
        into.world.restore_from(&self.world);
        into.pending_input.clone_from(&self.pending_input);
        into.present.clone_from(&self.present);
        into.last_known.clone_from(&self.last_known);
    }
    fn restore_state(&mut self, state: &Self::SavedState) {
        // See `save_state_into`: `restore_from` reuses the world's buffers.
        self.world.restore_from(&state.world);
        self.pending_input.clone_from(&state.pending_input);
        self.present.clone_from(&state.present);
        self.last_known.clone_from(&state.last_known);
    }

    /// Review round 1, F3: an earlier revision silently no-op'd here when the core was missing
    /// (real DDNet physics removes a dead character's core entirely, `ddai_physics::world::die`),
    /// which broke `thaw_escapable`'s cached scratch world forever after its first death (every
    /// later `apply_tee_state(0, ...)` on that same scratch tee became a no-op, so it stayed
    /// whatever it last was -- every subsequent escape attempt looked "caught"). Missing core now
    /// means "revive": rebuild the character fresh (`spawn_character`) at the target position,
    /// then apply the full state on top exactly as before.
    fn apply_tee_state(&mut self, id: i32, st: &TeeState) {
        if !self.present[id as usize] {
            return;
        }
        if self.world.cores.slot_of(id as u8).is_none() {
            ddai_physics::world::spawn_character(&mut self.world, id, to_p_vec2(st.pos));
        }
        let Some(slot) = self.world.cores.slot_of(id as u8) else {
            return;
        };
        {
            let core = self.world.cores.core_at_mut(slot);
            core.pos = to_p_vec2(st.pos);
            core.vel = to_p_vec2(st.vel);
            core.hook_state = st.hook_state;
            core.hook_pos = to_p_vec2(st.hook_pos);
            core.hook_dir = to_p_vec2(st.hook_dir);
            core.jumped = st.jumped;
            core.jumps = st.jumps.unwrap_or(2);
            core.jumped_total = st.jumped_total.unwrap_or(0);
            core.direction = st.direction;
            core.angle = st.angle as i32;
            core.active_weapon = st.active_weapon;
            if let Some(df) = st.deep_frozen {
                core.deep_frozen = df;
            }
        }
        {
            let mut me = *self.world.cores.core_at(slot);
            ddai_physics::core::set_hooked_player(&mut self.world.cores, &mut me, id as u8, st.hooked_player);
            *self.world.cores.core_at_mut(slot) = me;
        }
        let tick = self.world.tick;
        if let Some(character) = self.world.characters[id as usize].as_mut() {
            character.alive = st.alive;
            character.freeze_time = if st.frozen {
                (st.freeze_ticks_left.max(1)) as i32
            } else {
                0
            };
            character.attack_tick = st.attack_tick as i32;
            if let Some(rt) = st.reload_ticks {
                character.reload_timer = rt as i32;
            }
            character.prev_pos = PVec2::new((st.pos.x - st.vel.x) as f32, (st.pos.y - st.vel.y) as f32);
        }
        if st.frozen {
            self.world.cores.core_at_mut(slot).freeze_start = tick - (st.frozen_for.unwrap_or(0) as i32);
        }
        self.present[id as usize] = true;
        self.refresh_cache(id);
    }

    fn add_tee(&mut self, id: i32, pos: Vec2) {
        if !(0..MAX_CLIENTS as i32).contains(&id) {
            return;
        }
        ddai_physics::world::spawn_character(&mut self.world, id, to_p_vec2(pos));
        self.present[id as usize] = true;
        self.pending_input[id as usize] = crate::types::empty_input();
        self.refresh_cache(id);
    }

    /// Review round 1, F3 ("truly removes"): an earlier revision only cleared this wrapper's own
    /// `present`/`alive` bookkeeping, leaving the character's core (and `entity_order` entry)
    /// fully intact -- `World::world_tick` ticks *every* character in `entity_order` regardless of
    /// whether `step()`'s `inputs` slice mentions it (see `World::step`'s own doc comment: "every
    /// connected player", not only the ones with an input this tick), so the "removed" tee kept
    /// physically simulating forever on its last-set input (a ghost). `die()` is what actually
    /// retracts a character from `cores`/`entity_order` in real DDNet; calling it here (only when
    /// a core still exists -- a tee already dead in-game has none to remove) makes `remove_tee`
    /// match TS's `removeTee` (`world.ts:236-240`: gone from `order`/`byId`, never ticked again).
    fn remove_tee(&mut self, id: i32) {
        if !(0..MAX_CLIENTS as i32).contains(&id) {
            return;
        }
        if self.world.cores.slot_of(id as u8).is_some() {
            ddai_physics::world::die(&mut self.world, id, -1, -1);
        }
        self.present[id as usize] = false;
        self.last_known[id as usize] = None;
    }

    fn apply_force(&mut self, id: i32, force: Vec2) {
        let Some(slot) = self.world.cores.slot_of(id as u8) else {
            return;
        };
        let move_restrictions = self.world.characters[id as usize]
            .as_ref()
            .map_or(0, |c| c.move_restrictions);
        let core = self.world.cores.core_at_mut(slot);
        ddai_physics::world::take_damage(core, to_p_vec2(force), move_restrictions);
        self.refresh_cache(id);
    }

    fn unfreeze(&mut self, id: i32) {
        let Some(slot) = self.world.cores.slot_of(id as u8) else {
            return;
        };
        let core_copy = *self.world.cores.core_at(slot);
        if let Some(character) = self.world.characters[id as usize].as_mut() {
            let mut core = core_copy;
            ddai_physics::world::unfreeze(character, &mut core);
            *self.world.cores.core_at_mut(slot) = core;
        }
        self.refresh_cache(id);
    }

    /// See the module doc comment: builds a fresh `World` from the same map handle (not a deep
    /// copy of `self.world`'s live game state -- matching TS's `thawScratch`, which is a brand
    /// new `SimWorld`, not a clone of the live planning world).
    fn new_scratch(&self) -> Self {
        PhysicsWorld::new(self.map.clone(), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_map() -> Arc<MapData> {
        let w = 8usize;
        let h = 8usize;
        let mut game = vec![ddai_physics::map::Tile::default(); w * h];
        for x in 0..w {
            game[(h - 1) * w + x] = ddai_physics::map::Tile {
                index: ddai_physics::map::TILE_SOLID,
                flags: 0,
                skip: 0,
                reserved: 0,
            };
        }
        Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    #[test]
    fn add_tee_then_get_tee_round_trips_position() {
        let map = tiny_map();
        let mut w = PhysicsWorld::new(map, 1);
        <PhysicsWorld as PlanWorld>::add_tee(&mut w, 0, Vec2 { x: 100.0, y: 100.0 });
        let tee = <PhysicsWorld as PlanWorld>::get_tee(&w, 0).unwrap();
        assert!(tee.alive);
        assert!((tee.pos.x - 100.0).abs() < 1.0);
    }

    #[test]
    fn step_moves_a_tee_that_holds_direction() {
        let map = tiny_map();
        let mut w = PhysicsWorld::new(map, 1);
        <PhysicsWorld as PlanWorld>::add_tee(&mut w, 0, Vec2 { x: 100.0, y: 200.0 });
        let mut input = crate::types::empty_input();
        input.direction = 1;
        <PhysicsWorld as PlanWorld>::set_input(&mut w, 0, input);
        for _ in 0..25 {
            <PhysicsWorld as PlanWorld>::step(&mut w);
        }
        let tee = <PhysicsWorld as PlanWorld>::get_tee(&w, 0).unwrap();
        assert!(
            tee.pos.x > 100.0,
            "expected the tee to have moved right, got {}",
            tee.pos.x
        );
    }

    #[test]
    fn save_and_restore_round_trips_state() {
        let map = tiny_map();
        let mut w = PhysicsWorld::new(map, 1);
        <PhysicsWorld as PlanWorld>::add_tee(&mut w, 0, Vec2 { x: 100.0, y: 200.0 });
        let saved = <PhysicsWorld as PlanWorld>::save_state(&w);
        let mut input = crate::types::empty_input();
        input.direction = 1;
        <PhysicsWorld as PlanWorld>::set_input(&mut w, 0, input);
        for _ in 0..25 {
            <PhysicsWorld as PlanWorld>::step(&mut w);
        }
        <PhysicsWorld as PlanWorld>::restore_state(&mut w, &saved);
        let tee = <PhysicsWorld as PlanWorld>::get_tee(&w, 0).unwrap();
        assert!((tee.pos.x - 100.0).abs() < 1.0);
    }

    #[test]
    fn new_scratch_shares_the_map_but_has_no_tees() {
        let map = tiny_map();
        let mut w = PhysicsWorld::new(map, 1);
        <PhysicsWorld as PlanWorld>::add_tee(&mut w, 0, Vec2 { x: 100.0, y: 200.0 });
        let scratch = <PhysicsWorld as PlanWorld>::new_scratch(&w);
        assert!(<PhysicsWorld as PlanWorld>::get_tee(&scratch, 0).is_none());
    }

    /// A death tile at tile `(4, 4)`, floor along the bottom row -- reproduces the reviewer's own
    /// `zz_review_physics.rs` repro map exactly (review round 1, F2/F3).
    fn map_with_death() -> Arc<MapData> {
        let w = 12usize;
        let h = 10usize;
        let mut game = vec![ddai_physics::map::Tile::default(); w * h];
        for x in 0..w {
            game[(h - 1) * w + x] = ddai_physics::map::Tile {
                index: ddai_physics::map::TILE_SOLID,
                flags: 0,
                skip: 0,
                reserved: 0,
            };
        }
        game[8 * w + 8] = ddai_physics::map::Tile {
            index: ddai_physics::map::TILE_DEATH,
            flags: 0,
            skip: 0,
            reserved: 0,
        };
        Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    /// Review round 1, F9: a front-layer `TILE_FREEZE`/`TILE_UNFREEZE` with an all-air game layer
    /// (`front_tile`'s whole reason to exist -- real DDNet's `handle_tiles` ORs the two layers).
    fn map_with_front_freeze() -> Arc<MapData> {
        let w = 12usize;
        let h = 10usize;
        let mut game = vec![ddai_physics::map::Tile::default(); w * h];
        for x in 0..w {
            game[(h - 1) * w + x] = ddai_physics::map::Tile {
                index: ddai_physics::map::TILE_SOLID,
                flags: 0,
                skip: 0,
                reserved: 0,
            };
        }
        let mut front = vec![ddai_physics::map::Tile::default(); w * h];
        front[3 * w + 5] = ddai_physics::map::Tile {
            index: ddai_physics::map::TILE_FREEZE,
            flags: 0,
            skip: 0,
            reserved: 0,
        };
        front[3 * w + 6] = ddai_physics::map::Tile {
            index: ddai_physics::map::TILE_UNFREEZE,
            flags: 0,
            skip: 0,
            reserved: 0,
        };
        Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: Some(front),
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    #[test]
    fn is_freeze_and_is_un_freeze_see_the_front_layer_too() {
        let w = PhysicsWorld::new(map_with_front_freeze(), 1);
        let col = w.collision();
        // Front-layer freeze at tile (5, 3); the game layer there is plain air.
        assert_eq!(
            col.game_tile(5, 3),
            0,
            "game layer must be air for this to test the front layer"
        );
        assert!(
            PlanCollision::is_freeze(col, 5.0 * 32.0 + 16.0, 3.0 * 32.0 + 16.0),
            "front-layer TILE_FREEZE must count as freeze"
        );
        // Front-layer unfreeze at tile (6, 3), same story.
        assert_eq!(col.game_tile(6, 3), 0);
        assert!(
            PlanCollision::is_un_freeze(col, 6.0 * 32.0 + 16.0, 3.0 * 32.0 + 16.0),
            "front-layer TILE_UNFREEZE must count as unfreeze"
        );
        // A plain-air tile with no front-layer tile either is neither.
        assert!(!PlanCollision::is_freeze(col, 1.0 * 32.0 + 16.0, 3.0 * 32.0 + 16.0));
        assert!(!PlanCollision::is_un_freeze(col, 1.0 * 32.0 + 16.0, 3.0 * 32.0 + 16.0));
    }

    /// Review round 1, F2: a tee that dies (real DDNet removes its core entirely) must keep
    /// showing up from `get_tee` with `alive: false` and its last known position, exactly like
    /// TS's tee record -- never `None` (which `score_tick` reads as "no such tee", −1000 every
    /// tick, punishing the planner for successfully drowning the enemy).
    #[test]
    fn dead_tee_stays_visible_with_alive_false() {
        let mut w = PhysicsWorld::new(map_with_death(), 1);
        w.add_tee(0, Vec2 { x: 80.0, y: 272.0 });
        w.add_tee(
            1,
            Vec2 {
                x: 8.0 * 32.0 + 16.0,
                y: 8.0 * 32.0 + 16.0,
            },
        );
        w.set_input(1, crate::types::empty_input());
        let mut died = false;
        for _ in 0..10 {
            let events = w.step();
            if events.iter().any(|e| matches!(e, WorldEvent::Death { id: 1, .. })) {
                died = true;
            }
        }
        assert!(died, "expected tee 1 to die on the death tile");
        assert!(
            w.inner().cores.slot_of(1).is_none(),
            "real DDNet should have removed the core"
        );
        let tee = w.get_tee(1).expect("F2: a dead tee must still be visible, not None");
        assert!(!tee.alive);
        assert!(
            (tee.pos.x - (8.0 * 32.0 + 16.0)).abs() < 64.0,
            "position should be near where it died, got {:?}",
            tee.pos
        );
    }

    /// Review round 1, F3: `apply_tee_state` on a dead tee (core missing) must revive it (rebuild
    /// the character) instead of silently no-op'ing -- otherwise a cached scratch world (like
    /// `Planner::thaw_escapable`'s) is permanently broken after its first death.
    #[test]
    fn apply_tee_state_revives_a_dead_tee() {
        let mut w = PhysicsWorld::new(map_with_death(), 1);
        w.add_tee(
            0,
            Vec2 {
                x: 8.0 * 32.0 + 16.0,
                y: 8.0 * 32.0 + 16.0,
            },
        );
        w.set_input(0, crate::types::empty_input());
        for _ in 0..10 {
            w.step();
        }
        assert!(
            w.inner().cores.slot_of(0).is_none(),
            "should have died and lost its core"
        );

        let mut revived = w.get_tee(0).unwrap();
        revived.pos = Vec2 { x: 48.0, y: 272.0 };
        revived.alive = true;
        revived.frozen = false;
        revived.freeze_ticks_left = 0;
        w.apply_tee_state(0, &revived);
        assert!(
            w.inner().cores.slot_of(0).is_some(),
            "F3: apply_tee_state must revive the core"
        );
        let after = w.get_tee(0).expect("revived tee must be visible");
        assert!(after.alive, "F3: revived tee must report alive again");
        assert!((after.pos.x - 48.0).abs() < 1.0);
    }

    /// Review round 1, F3: `remove_tee` must truly retract the character from physics (the real
    /// `die()`), not just this wrapper's own bookkeeping -- otherwise `World::step` keeps ticking
    /// a "removed" tee forever on its last input (a ghost).
    #[test]
    fn remove_tee_truly_stops_the_character_from_being_simulated() {
        let mut w = PhysicsWorld::new(tiny_map(), 1);
        w.add_tee(0, Vec2 { x: 80.0, y: 80.0 });
        let mut falling_input = crate::types::empty_input();
        falling_input.direction = 1;
        w.set_input(0, falling_input);
        w.remove_tee(0);
        assert!(w.get_tee(0).is_none(), "removed tee must not be visible at all");
        assert!(
            w.inner().cores.slot_of(0).is_none(),
            "F3: remove_tee must retract the core, not leave a ghost"
        );
        let pos_before = w.inner().characters[0].map(|c| c.prev_pos);
        for _ in 0..20 {
            w.step();
        }
        assert!(
            w.inner().cores.slot_of(0).is_none(),
            "a removed tee must never come back on its own"
        );
        let _ = pos_before;
    }
}
