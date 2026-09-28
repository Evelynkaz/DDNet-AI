//! Literal port of `src/core/world.ts` (TS, `Wranked1/DDNet-AI`, GPL-3.0) — `SimWorld`, its
//! per-tee bookkeeping (`TeeRecord`), and the `saveState`/`restoreState` wire types
//! (`SimState`/`CoreState`/`TeeStateSnapshot`). Also carries the parts of
//! `src/core/characterCore.ts` that need more than one tee's data at once (`tick`,
//! `tick_deferred`, `move_`, `set_hooked_player`) — see `character_core.rs`'s module doc comment
//! for why those moved here.
//!
//! # Known TS quirks, reproduced on purpose
//!
//! - **`order` uses `unshift`, not `push`.** `addTee`/`respawnTee` both prepend to `order`
//!   (`world.ts:231`, `:717-718`); every physics/weapon loop in `step()` iterates `order` in that
//!   (newest/most-recently-respawned-first) sequence — see `docs/research/orig-plan.md` §2.5 and
//!   the crate README "«новые первыми»". `byId` is a *separate* array, re-sorted ascending by id
//!   on every `add_tee` (`world.ts:233`) — `CharacterCore.tick`/`tick_deferred`/`move_`'s own
//!   "all other cores" loops use *this* order (`allCores()`, `world.ts:509-513`), not `order`.
//! - **`applyTeeState`/`restoreState` do not overwrite every field.** `applyTeeState`
//!   (`world.ts:353-398`) leaves `queuedWeapon`, `weapons`, `teleCheckpoint`,
//!   `frozenLastTick` untouched (only `restoreState`/`reset` touch those) — see
//!   `docs/research/orig-plan.md` §2.5.
//! - **`addTee` with a duplicate id does not remove the old record.** `this.tees.set(id, rec)`
//!   (a `Map`) replaces the id→record lookup, but the *old* record stays in `order`/`byId` as an
//!   orphan that still ticks every `step()` (`world.ts:224-234`) — reproduced via
//!   [`TeeStorage::add_tee`] below (never removes a same-id slot on add, only `remove_tee` does).
//! - **`pendingCoreInput` is a dead field** (`world.ts:133`, `:347`, `:397`, `:930`): assigned in
//!   three places, read nowhere in the whole `src/` tree (verified by `grep -rn
//!   pendingCoreInput src/`), so it cannot affect any observable output — intentionally omitted
//!   from [`TeeRecord`].
//! - **`frozenInput` is per-tick scratch, not persistent state**: `preTick` overwrites every
//!   field of it before reading any of them (`world.ts:923-928`) and it is never read outside
//!   `preTick`, so [`pre_tick`] uses a local variable instead of a struct field (same observable
//!   result, no cross-tick state to carry).

use std::collections::HashMap;

use crate::character_core::{
    COREEVENT_AIR_JUMP, COREEVENT_GROUND_JUMP, COREEVENT_HOOK_ATTACH_GROUND, COREEVENT_HOOK_ATTACH_PLAYER,
    COREEVENT_HOOK_HIT_NOHOOK, COREEVENT_HOOK_LAUNCH, COREEVENT_HOOK_RETRACT, CharacterCore, HOOK_FLYING, HOOK_GRABBED,
    HOOK_IDLE, HOOK_RETRACT_END, HOOK_RETRACT_START, HOOK_RETRACTED, MOVE_SIZE, saturated_add, velocity_ramp,
};
use crate::collision::Collision;
use crate::projectile::{EntityWorld, Laser, LaserState2, Projectile, ProjectileState2, is_game_layer_clipped};
use crate::tuning::{
    CANTMOVE_DOWN, CFLAG_NOHOOK, PHYSICAL_SIZE, SERVER_TICK_SPEED, TILE_DEATH, TILE_DFREEZE, TILE_DUNFREEZE,
    TILE_FREEZE, TILE_LFREEZE, TILE_TELECHECK, TILE_TELECHECKIN, TILE_TELECHECKINEVIL, TILE_TELEIN, TILE_TELEINEVIL,
    TILE_UNFREEZE, tuning,
};
use crate::types::{
    NUM_WEAPONS, PlayerInput, ProjectileState, TeeState, WEAPON_GRENADE, WEAPON_GUN, WEAPON_HAMMER, WEAPON_LASER,
    WEAPON_SHOTGUN, WorldEvent, blank_tee_state, copy_input, empty_input,
};
use crate::vmath::{Vec2, closest_point_on_line_or_null, vadd, vdistance, vmul, vnormalize, vsub};
use ddai_jsmath as js;

const FREEZE_SECONDS: f64 = 3.0;
const INPUT_STATE_MASK: i32 = 0x3f;

/// `countPresses(prev, cur)` (`world.ts:18-28`).
fn count_presses(prev: i32, cur: i32) -> i32 {
    let prev = prev & INPUT_STATE_MASK;
    let cur = cur & INPUT_STATE_MASK;
    let mut i = prev;
    let mut presses = 0;
    while i != cur {
        i = (i + 1) & INPUT_STATE_MASK;
        if i & 1 != 0 {
            presses += 1;
        }
    }
    presses
}

/// `weaponFireDelayMs(weapon)` (`world.ts:30-45`).
fn weapon_fire_delay_ms(weapon: i32, tune: &crate::tuning::Tuning) -> f64 {
    if weapon == WEAPON_HAMMER {
        tune.hammer_fire_delay
    } else if weapon == WEAPON_GUN {
        tune.gun_fire_delay
    } else if weapon == WEAPON_SHOTGUN {
        tune.shotgun_fire_delay
    } else if weapon == WEAPON_GRENADE {
        tune.grenade_fire_delay
    } else if weapon == WEAPON_LASER {
        tune.laser_fire_delay
    } else {
        0.0
    }
}

/// `fireDelayTicks(weapon)` (`world.ts:47-49`).
fn fire_delay_ticks(weapon: i32, tune: &crate::tuning::Tuning) -> i64 {
    js::trunc(weapon_fire_delay_ms(weapon, tune) * SERVER_TICK_SPEED / 1000.0) as i64
}

/// `WeaponSlot` (`world.ts:51`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeaponSlot {
    pub got: bool,
    pub ammo: i32,
}

fn has_ammo(slot: WeaponSlot) -> bool {
    slot.ammo != 0
}

/// `defaultWeapons(infiniteAmmo, allWeapons)` (`world.ts:57-69`).
fn default_weapons(infinite_ammo: bool, all_weapons: bool) -> [WeaponSlot; NUM_WEAPONS as usize] {
    let mut slots = [WeaponSlot { got: false, ammo: 0 }; NUM_WEAPONS as usize];
    let ammo = if infinite_ammo { -1 } else { 10 };
    slots[WEAPON_HAMMER as usize] = WeaponSlot { got: true, ammo: -1 };
    slots[WEAPON_GUN as usize] = WeaponSlot { got: true, ammo };
    if all_weapons {
        slots[WEAPON_SHOTGUN as usize] = WeaponSlot { got: true, ammo };
        slots[WEAPON_GRENADE as usize] = WeaponSlot { got: true, ammo };
        slots[WEAPON_LASER as usize] = WeaponSlot { got: true, ammo };
    }
    slots
}

/// `TeeRecord` (`world.ts:110-144`), minus `pendingCoreInput`/`frozenInput` — see the module doc
/// comment.
#[derive(Debug, Clone)]
struct TeeRecord {
    core: CharacterCore,
    alive: bool,
    spawn_pos: Vec2,
    weapons: [WeaponSlot; NUM_WEAPONS as usize],
    queued_weapon: i32,
    reload_timer: i64,

    attack_tick: i64,
    freeze_ticks_left: i64,

    frozen_last_tick: bool,
    deep_frozen: bool,
    tele_checkpoint: i32,

    move_restrictions: i32,
    input: PlayerInput,
    prev_input_for_edge: PlayerInput,

    prev_pos: Vec2,
    respawn_at_tick: Option<i64>,
}

impl TeeRecord {
    fn new(core: CharacterCore, spawn_pos: Vec2, infinite_ammo: bool, all_weapons: bool) -> Self {
        TeeRecord {
            core,
            alive: true,
            spawn_pos,
            weapons: default_weapons(infinite_ammo, all_weapons),
            queued_weapon: -1,
            reload_timer: 0,
            attack_tick: -1000,
            freeze_ticks_left: 0,
            frozen_last_tick: false,
            deep_frozen: false,
            tele_checkpoint: 0,
            move_restrictions: 0,
            input: empty_input(),
            prev_input_for_edge: empty_input(),
            prev_pos: spawn_pos,
            respawn_at_tick: None,
        }
    }
}

/// `freezeTee(currentTick, rec, seconds)` (`world.ts:71-80`).
fn freeze_tee(current_tick: i64, rec: &mut TeeRecord, seconds: f64) -> bool {
    if seconds <= 0.0 {
        return false;
    }
    let seconds_ticks = js::trunc(seconds * SERVER_TICK_SPEED) as i64;
    if rec.freeze_ticks_left > seconds_ticks {
        return false;
    }
    if rec.freeze_ticks_left == 0 || rec.core.freeze_start < current_tick - SERVER_TICK_SPEED as i64 {
        rec.freeze_ticks_left = seconds_ticks;
        rec.core.freeze_start = current_tick;
        return true;
    }
    false
}

/// `unfreezeTee(rec)` (`world.ts:82-91`).
fn unfreeze_tee(rec: &mut TeeRecord) -> bool {
    if rec.freeze_ticks_left > 0 {
        rec.freeze_ticks_left = 0;
        rec.core.freeze_start = 0;
        rec.frozen_last_tick = true;
        return true;
    }
    false
}

/// `applyJumpRules(core)` (`world.ts:93-98`).
// Several branches below produce the same `core.jumped |= 2;` body for different, independently
// meaningful TS conditions (`world.ts:93-98`) — kept as separate `else if` arms (not merged with
// `||`) for a literal, line-for-line correspondence with the TS source, not because the port
// couldn't express it more tersely.
#[allow(clippy::if_same_then_else)]
fn apply_jump_rules(core: &mut CharacterCore) {
    if core.jumps == -1 {
        core.jumped |= 2;
    } else if core.jumps == 0 {
        core.jumped |= 2;
    } else if core.jumps == 1 && core.jumped > 0 {
        core.jumped |= 2;
    } else if core.jumped_total < core.jumps - 1 && core.jumped > 1 {
        core.jumped = 1;
    }
}

/// `computeJumpsLeft(core, collision)` (`world.ts:100-108`).
fn compute_jumps_left(core: &CharacterCore, collision: &Collision) -> i32 {
    if core.jumps <= 0 {
        return 0;
    }
    let grounded = collision.is_solid(core.pos.x + PHYSICAL_SIZE / 2.0, core.pos.y + PHYSICAL_SIZE / 2.0 + 5.0)
        || collision.is_solid(core.pos.x - PHYSICAL_SIZE / 2.0, core.pos.y + PHYSICAL_SIZE / 2.0 + 5.0);
    if grounded {
        return core.jumps;
    }
    if core.jumped & 2 != 0 {
        return 0;
    }
    js::max(0.0, (core.jumps - 1 - core.jumped_total) as f64) as i32
}

// --- saveState/restoreState wire types (`CoreState`/`TeeStateSnapshot`/`SimState`, world.ts:146-181) --

/// `CoreState` (`world.ts:146-154`).
#[derive(Debug, Clone, PartialEq)]
pub struct CoreState {
    pub pos: Vec2,
    pub vel: Vec2,
    pub hook_pos: Vec2,
    pub hook_dir: Vec2,
    pub hook_tele_base: Vec2,
    pub hook_tick: i64,
    pub hook_state: i32,
    pub hooked_player: i32,
    pub attached_players: Vec<i32>,
    pub active_weapon: i32,
    pub new_hook: bool,
    pub jumped: i32,
    pub jumped_total: i32,
    pub jumps: i32,
    pub direction: i32,
    pub angle: i32,
    pub triggered_events: u32,
    pub colliding: i32,
    pub left_wall: bool,
    pub freeze_start: i64,
    pub freeze_end: i64,
    pub is_in_freeze: bool,
    pub move_restrictions: i32,
}

/// `TeeStateSnapshot` (`world.ts:156-173`).
#[derive(Debug, Clone, PartialEq)]
pub struct TeeStateSnapshot {
    pub core: CoreState,
    pub alive: bool,
    pub freeze_ticks_left: i64,
    pub frozen_last_tick: bool,
    pub deep_frozen: bool,
    pub tele_checkpoint: i32,
    pub move_restrictions: i32,
    pub reload_timer: i64,
    pub attack_tick: i64,
    pub queued_weapon: i32,
    pub input: PlayerInput,
    pub prev_input_for_edge: PlayerInput,
    pub prev_pos: Vec2,
    pub spawn_pos: Vec2,
    pub respawn_at_tick: Option<i64>,
    pub weapons: Vec<WeaponSlot>,
}

/// `SimState` (`world.ts:175-181`). `tees` is a `Vec<(id, snapshot)>` rather than a map: TS's
/// `Map<number, TeeStateSnapshot>` iterates in insertion order, but `saveState`/`restoreState`
/// only ever look entries up by id (never iterate order-sensitively), so a `Vec` of pairs behaves
/// identically for every observable purpose here while avoiding a `HashMap` in the public API.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SimState {
    pub tick: i64,
    pub next_entity_id: i64,
    pub tees: Vec<(i32, TeeStateSnapshot)>,
    pub projectiles: Vec<ProjectileState2>,
    pub lasers: Vec<LaserState2>,
}

fn core_state_of(c: &CharacterCore) -> CoreState {
    CoreState {
        pos: c.pos,
        vel: c.vel,
        hook_pos: c.hook_pos,
        hook_dir: c.hook_dir,
        hook_tele_base: c.hook_tele_base,
        hook_tick: c.hook_tick,
        hook_state: c.hook_state,
        hooked_player: c.hooked_player,
        attached_players: c.attached_players.iter().collect(),
        active_weapon: c.active_weapon,
        new_hook: c.new_hook,
        jumped: c.jumped,
        jumped_total: c.jumped_total,
        jumps: c.jumps,
        direction: c.direction,
        angle: c.angle,
        triggered_events: c.triggered_events,
        colliding: c.colliding,
        left_wall: c.left_wall,
        freeze_start: c.freeze_start,
        freeze_end: c.freeze_end,
        is_in_freeze: c.is_in_freeze,
        move_restrictions: c.move_restrictions,
    }
}

fn snapshot_of(rec: &TeeRecord) -> TeeStateSnapshot {
    TeeStateSnapshot {
        core: core_state_of(&rec.core),
        alive: rec.alive,
        freeze_ticks_left: rec.freeze_ticks_left,
        frozen_last_tick: rec.frozen_last_tick,
        deep_frozen: rec.deep_frozen,
        tele_checkpoint: rec.tele_checkpoint,
        move_restrictions: rec.move_restrictions,
        reload_timer: rec.reload_timer,
        attack_tick: rec.attack_tick,
        queued_weapon: rec.queued_weapon,
        input: rec.input,
        prev_input_for_edge: rec.prev_input_for_edge,
        prev_pos: rec.prev_pos,
        spawn_pos: rec.spawn_pos,
        respawn_at_tick: rec.respawn_at_tick,
        weapons: rec.weapons.to_vec(),
    }
}

// --- Tee storage: `tees`/`order`/`byId` (`world.ts:205-209`) -----------------------------------

/// Backs `SimWorld`'s `tees: Map<number, TeeRecord>` + `order: TeeRecord[]` + `byId: TeeRecord[]`
/// (`world.ts:205-209`). See the module doc comment for the arena-with-orphans design this uses
/// to reproduce the duplicate-`addTee`-id quirk in safe Rust (no reference-counted cells).
#[derive(Debug, Clone, Default)]
struct TeeStorage {
    slots: Vec<TeeRecord>,
    id_index: HashMap<i32, usize>,
    /// `order` (`world.ts:207`): slot indices, newest/respawned-first via `unshift`.
    order: Vec<usize>,
    /// `byId` (`world.ts:209`): slot indices, kept sorted ascending by `core.id`.
    by_id: Vec<usize>,
    /// Review finding F11: slot indices that are provably unreferenced (not in `order`, not in
    /// `by_id`, not in `id_index` for any id) and therefore safe to overwrite in place instead of
    /// growing `slots` forever. Only `remove_tee` can produce these — TS's own `removeTee`
    /// (`world.ts:236-240`) filters `order`/`byId` by *value* equality on `core.id`, which sweeps
    /// out every slot with that id, orphan or current alike (unlike a duplicate `addTee`, whose
    /// orphan deliberately stays referenced — F4 — so it must NOT be freed here). Reusing a freed
    /// index changes no TS-observable field: every external read goes through `id_index` or walks
    /// `order`/`by_id`, never a raw slot index, so which physical `Vec` slot a tee's data happens
    /// to live in is not part of the parity target. See `TeeStorage::add_tee`/`remove_tee`.
    free_slots: Vec<usize>,
}

impl TeeStorage {
    fn get(&self, id: i32) -> Option<&TeeRecord> {
        self.id_index.get(&id).map(|&i| &self.slots[i])
    }

    fn get_mut(&mut self, id: i32) -> Option<&mut TeeRecord> {
        match self.id_index.get(&id) {
            Some(&i) => Some(&mut self.slots[i]),
            None => None,
        }
    }

    fn index_of(&self, id: i32) -> Option<usize> {
        self.id_index.get(&id).copied()
    }

    /// `addTee` (`world.ts:224-234`). Review finding F11: reuses a slot from `free_slots` when one
    /// is available instead of always growing `slots` — a duplicate-id orphan's slot is never in
    /// `free_slots` (only `remove_tee` populates it, and only for slots it actually swept out of
    /// both `order` and `by_id`), so this cannot collide with a still-referenced orphan.
    fn add_tee(&mut self, id: i32, spawn_pos: Vec2, infinite_ammo: bool, all_weapons: bool) {
        let mut core = CharacterCore::new(id);
        core.reset();
        core.pos = spawn_pos;
        core.active_weapon = WEAPON_GUN;
        let rec = TeeRecord::new(core, spawn_pos, infinite_ammo, all_weapons);
        let slot = match self.free_slots.pop() {
            Some(slot) => {
                self.slots[slot] = rec;
                slot
            }
            None => {
                self.slots.push(rec);
                self.slots.len() - 1
            }
        };
        self.id_index.insert(id, slot);
        self.order.insert(0, slot);
        self.by_id.push(slot);
        let TeeStorage { slots, by_id, .. } = self;
        by_id.sort_by_key(|&i| slots[i].core.id);
    }

    /// `removeTee` (`world.ts:236-240`). Review finding F11: every slot this actually removes from
    /// `order` (and, in lockstep, `by_id` — both always hold the same multiset of slot indices,
    /// just in different orders, since only `add_tee`/`remove_tee` ever change that multiset) is
    /// now provably unreferenced — `id_index` no longer maps *any* id to it (TS's own filter is by
    /// `core.id` value, so a same-id orphan is swept out here too, unlike a duplicate `addTee`,
    /// which deliberately leaves the orphan referenced — F4) — so it's pushed onto `free_slots`
    /// for `add_tee` to reuse, instead of leaking forever.
    fn remove_tee(&mut self, id: i32) {
        self.id_index.remove(&id);
        let TeeStorage {
            slots,
            order,
            by_id,
            free_slots,
            ..
        } = self;
        free_slots.extend(order.iter().copied().filter(|&i| slots[i].core.id == id));
        order.retain(|&i| slots[i].core.id != id);
        by_id.retain(|&i| slots[i].core.id != id);
    }

    /// `pair_mut(i, j)`: simultaneous mutable access to two distinct slots, needed for
    /// `CharacterCore.tickDeferred`'s pairwise `this`/`other` mutation (`characterCore.ts:309-355`)
    /// — not present in TS (there, `this`/`other` are just two live object references), but the
    /// only way to express "mutate two distinct elements of one `Vec` at once" in safe Rust.
    fn pair_mut(&mut self, i: usize, j: usize) -> (&mut TeeRecord, &mut TeeRecord) {
        assert_ne!(i, j, "pair_mut requires distinct slots");
        if i < j {
            let (left, right) = self.slots.split_at_mut(j);
            (&mut left[i], &mut right[0])
        } else {
            let (left, right) = self.slots.split_at_mut(i);
            (&mut right[0], &mut left[j])
        }
    }
}

// --- CoreWorld-equivalent read helpers (used by CharacterCore.tick/tickDeferred/move ports) -----

impl TeeStorage {
    /// **Slot** indices of every alive tee, in `byId` order (`allCores()`, `world.ts:509-513`),
    /// appended into `out` (cleared first) — a reusable scratch buffer so this never allocates
    /// once `out`'s capacity has stabilized to the tee count.
    ///
    /// Deliberately **slots**, not ids: `byId` in TS is an array of live object references and
    /// can hold two distinct objects with the same `.id` (the duplicate-`addTee` orphan quirk,
    /// see the module doc comment) — `allCores()` returns *both* as separate entries (TS
    /// `other === this` self-exclusion is reference equality, `characterCore.ts:240`, `:312`).
    /// Deduplicating by id here (an earlier version of this port did, via an id-keyed variant of
    /// this method) would make an orphan invisible to every other tee's hook-scan/collision loop
    /// and let the *current* record for that id stand in for it twice — caught by review finding
    /// F4.
    fn alive_slots_by_id_into(&self, out: &mut Vec<usize>) {
        out.clear();
        for &i in &self.by_id {
            if self.slots[i].alive {
                out.push(i);
            }
        }
    }
}

// --- SimWorld ------------------------------------------------------------------------------------

/// Options accepted by [`SimWorld::new`] — `SimWorld`'s constructor's optional second argument
/// (`world.ts:215`). `None` for a field means "use the TS default", matching `options?.foo ?? default`.
#[derive(Debug, Clone, Default)]
pub struct SimWorldOptions {
    pub respawn_delay_ticks: Option<i64>,
    pub infinite_ammo: Option<bool>,
    pub sv_hit: Option<bool>,
    pub all_weapons: Option<bool>,
    pub no_weak_hook: Option<bool>,
}

/// `class SimWorld` (`world.ts:196-998`). See the module and crate doc comments for the TS
/// quirks this reproduces on purpose and the restructuring (vs. a literal object-reference graph)
/// this crate uses instead.
#[derive(Debug, Clone)]
pub struct SimWorld {
    pub collision: Collision,
    pub tick: i64,
    sv_hit: bool,
    respawn_delay_ticks: i64,
    infinite_ammo: bool,
    all_weapons: bool,
    pub no_weak_hook: bool,

    tees: TeeStorage,
    next_entity_id: i64,
    projectiles_list: Vec<Projectile>,
    lasers_list: Vec<Laser>,
    pending_events: Vec<WorldEvent>,

    // Reusable scratch buffers (never reallocated once capacity stabilizes to the tee count) —
    // acceptance criterion 1's "no allocation per tick in steady state".
    scratch_slots: Vec<usize>,
    scratch_tile_indices: Vec<i32>,
    scratch_other_snapshots: Vec<(Vec2, bool, bool)>,
    scratch_order: Vec<usize>,
    scratch_targets: Vec<usize>,
}

impl SimWorld {
    /// `constructor(collision, options?)` (`world.ts:215-222`).
    pub fn new(collision: Collision, options: SimWorldOptions) -> Self {
        let no_weak_hook = options.no_weak_hook.unwrap_or(collision.no_weak_hook);
        SimWorld {
            collision,
            tick: 0,
            sv_hit: options.sv_hit.unwrap_or(true),
            respawn_delay_ticks: options.respawn_delay_ticks.unwrap_or(0),
            infinite_ammo: options.infinite_ammo.unwrap_or(true),
            all_weapons: options.all_weapons.unwrap_or(false),
            no_weak_hook,
            tees: TeeStorage::default(),
            next_entity_id: 1,
            projectiles_list: Vec::new(),
            lasers_list: Vec::new(),
            pending_events: Vec::new(),
            scratch_slots: Vec::new(),
            scratch_tile_indices: Vec::new(),
            scratch_other_snapshots: Vec::new(),
            scratch_order: Vec::new(),
            scratch_targets: Vec::new(),
        }
    }

    /// `addTee(id, spawnPos)` (`world.ts:224-234`).
    pub fn add_tee(&mut self, id: i32, spawn_pos: Vec2) {
        self.tees.add_tee(id, spawn_pos, self.infinite_ammo, self.all_weapons);
    }

    /// `removeTee(id)` (`world.ts:236-240`).
    pub fn remove_tee(&mut self, id: i32) {
        self.tees.remove_tee(id);
    }

    /// `setInput(id, input)` (`world.ts:242-259`).
    pub fn set_input(&mut self, id: i32, input: PlayerInput) {
        let Some(rec) = self.tees.get_mut(id) else { return };
        let mut dst = input;
        if dst.target_x == 0.0 && dst.target_y == 0.0 {
            dst.target_y = -1.0;
        }
        rec.input = dst;
    }

    /// `setHeldInput(id, input)` (`world.ts:400-407`).
    pub fn set_held_input(&mut self, id: i32, input: PlayerInput) {
        let Some(rec) = self.tees.get_mut(id) else { return };
        copy_input(&input, &mut rec.input);
        copy_input(&input, &mut rec.prev_input_for_edge);
        if rec.input.target_x == 0.0 && rec.input.target_y == 0.0 {
            rec.input.target_y = -1.0;
        }
        if rec.prev_input_for_edge.target_x == 0.0 && rec.prev_input_for_edge.target_y == 0.0 {
            rec.prev_input_for_edge.target_y = -1.0;
        }
    }

    /// `saveState(into?)` (`world.ts:261-304`).
    pub fn save_state(&self) -> SimState {
        let mut st = SimState::default();
        self.save_state_into(&mut st);
        st
    }

    /// `saveState(into)` reusing an existing buffer (avoids allocating a new `SimState` every
    /// call, matching TS callers that pass their own `into`, e.g. `src/plan/planner.ts`).
    ///
    /// Two subtleties, both review-caught (findings F4/F7) — TS's `saveState` (`world.ts:261-304`)
    /// is `for (const [id, rec] of this.tees) { let t = st.tees.get(id); if (t === undefined) {
    /// ...; st.tees.set(id, t); } ... fill t from rec ...; }`:
    /// 1. It iterates `this.tees` — the id→record **`Map`** (current mapping only, one entry per
    ///    id) — never `order`/`byId` (which can hold an orphaned duplicate-id slot, see the
    ///    module doc comment). An orphan is therefore invisible to `saveState`/`getTee`/etc.
    ///    entirely, exactly like TS.
    /// 2. It never clears `into.tees` first — an existing entry is updated *in place* (`t =
    ///    st.tees.get(id)`, found or not), a new one is added, but an id that *was* in `into` and
    ///    is no longer in `this.tees` (e.g. after `removeTee`) is left stale, untouched. An
    ///    earlier version of this port cleared `st.tees` unconditionally at the top instead.
    pub fn save_state_into(&self, st: &mut SimState) {
        st.tick = self.tick;
        st.next_entity_id = self.next_entity_id;
        // `id_index.keys()` has no defined order (`HashMap`); sorted ascending for determinism —
        // only matters for *newly appended* entries' relative order (existing entries keep their
        // position), and `SimState.tees`'s own order is not itself part of the parity target
        // (`trace_ts::diff_sim_state` compares by id, order-independently) — see `reset()`'s doc
        // comment for the same reasoning.
        let mut ids: Vec<i32> = self.tees.id_index.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let slot = self.tees.id_index[&id];
            let rec = &self.tees.slots[slot];
            match st.tees.iter_mut().find(|(existing_id, _)| *existing_id == id) {
                Some(entry) => entry.1 = snapshot_of(rec),
                None => st.tees.push((id, snapshot_of(rec))),
            }
        }
        st.projectiles = self.projectiles_list.iter().map(Projectile::save_state).collect();
        st.lasers = self.lasers_list.iter().map(Laser::save_state).collect();
    }

    /// `restoreState(st)` (`world.ts:306-351`). Only updates tees that currently exist (`world.ts:310`
    /// — `if (rec === undefined) continue`); never re-adds removed tees or touches `order`/`byId`.
    pub fn restore_state(&mut self, st: &SimState) {
        self.tick = st.tick;
        self.next_entity_id = st.next_entity_id;
        for (id, t) in &st.tees {
            let Some(rec) = self.tees.get_mut(*id) else { continue };
            let c = &mut rec.core;
            let sc = &t.core;
            c.pos = sc.pos;
            c.vel = sc.vel;
            c.hook_pos = sc.hook_pos;
            c.hook_dir = sc.hook_dir;
            c.hook_tele_base = sc.hook_tele_base;
            c.hook_tick = sc.hook_tick;
            c.hook_state = sc.hook_state;
            c.hooked_player = sc.hooked_player;
            c.attached_players.clear();
            for &a in &sc.attached_players {
                c.attached_players.add(a);
            }
            c.active_weapon = sc.active_weapon;
            c.new_hook = sc.new_hook;
            c.jumped = sc.jumped;
            c.jumped_total = sc.jumped_total;
            c.jumps = sc.jumps;
            c.direction = sc.direction;
            c.angle = sc.angle;
            c.triggered_events = sc.triggered_events;
            c.colliding = sc.colliding;
            c.left_wall = sc.left_wall;
            c.freeze_start = sc.freeze_start;
            c.freeze_end = sc.freeze_end;
            c.is_in_freeze = sc.is_in_freeze;
            c.move_restrictions = sc.move_restrictions;
            rec.alive = t.alive;
            rec.freeze_ticks_left = t.freeze_ticks_left;
            rec.frozen_last_tick = t.frozen_last_tick;
            rec.deep_frozen = t.deep_frozen;
            rec.tele_checkpoint = t.tele_checkpoint;
            rec.move_restrictions = t.move_restrictions;
            rec.reload_timer = t.reload_timer;
            rec.attack_tick = t.attack_tick;
            rec.queued_weapon = t.queued_weapon;
            rec.respawn_at_tick = t.respawn_at_tick;
            rec.input = t.input;
            rec.prev_input_for_edge = t.prev_input_for_edge;
            rec.prev_pos = t.prev_pos;
            rec.spawn_pos = t.spawn_pos;
            for (dst, src) in rec.weapons.iter_mut().zip(t.weapons.iter()) {
                *dst = *src;
            }
        }
        self.projectiles_list = st.projectiles.iter().map(Projectile::from_state).collect();
        self.lasers_list = st.lasers.iter().map(Laser::from_state).collect();
    }

    /// `applyTeeState(id, st)` (`world.ts:353-398`). See the module doc comment: several
    /// `TeeRecord` fields (`queuedWeapon`, `weapons`, `teleCheckpoint`, `frozenLastTick`) are
    /// deliberately left untouched, matching TS.
    pub fn apply_tee_state(&mut self, id: i32, st: &TeeState) {
        let collision = &self.collision;
        let tick = self.tick;
        let Some(rec) = self.tees.get_mut(id) else { return };
        let c = &mut rec.core;
        c.pos = st.pos;
        c.vel = st.vel;
        c.hook_state = st.hook_state;
        c.hook_pos = st.hook_pos;
        c.hook_dir = st.hook_dir;
        c.hooked_player = st.hooked_player;

        if let Some(ht) = st.hook_tick {
            c.hook_tick = ht;
        }
        c.jumped = st.jumped;

        c.jumps = st.jumps.unwrap_or(2);
        c.jumped_total = st.jumped_total.unwrap_or_else(|| {
            js::max(
                0.0,
                (c.jumps - st.jumps_left - if st.jumped & 1 != 0 { 1 } else { 0 }) as f64,
            ) as i32
        });

        if let Some(flags) = st.ddnet_flags {
            c.solo = flags & crate::types::CHARACTERFLAG_SOLO != 0;
            c.collision_disabled = flags & crate::types::CHARACTERFLAG_COLLISION_DISABLED != 0;
            c.hook_hit_disabled = flags & crate::types::CHARACTERFLAG_HOOK_HIT_DISABLED != 0;
            c.endless_hook = flags & crate::types::CHARACTERFLAG_ENDLESS_HOOK != 0;
        }
        c.direction = st.direction;
        // TS: `c.angle = st.angle;` — a plain assignment, no `Math.trunc` (`world.ts:380-381`).
        // `st.angle` is always already integral in practice (the only place TS ever *produces*
        // one, `core.tick()`, always writes `Math.trunc(...)`), but this port's own
        // `CharacterCore.angle: i32` can't hold a fractional value regardless — `as i32` (a
        // plain truncating cast, not `js::trunc`, since TS performs no such operation here)
        // is the closest literal equivalent of "just assign whatever `f64` was given".
        c.angle = st.angle as i32;
        c.active_weapon = st.active_weapon;
        rec.alive = st.alive;

        rec.attack_tick = match st.since_attack {
            Some(since) => tick - since,
            None => st.attack_tick,
        };
        rec.freeze_ticks_left = if st.frozen {
            js::max(1.0, st.freeze_ticks_left as f64) as i64
        } else {
            0
        };

        if let Some(rt) = st.reload_ticks {
            rec.reload_timer = rt;
        }
        if st.frozen
            && let Some(frozen_for) = st.frozen_for
        {
            c.freeze_start = tick - frozen_for;
        }
        if let Some(df) = st.deep_frozen {
            rec.deep_frozen = df;
        }

        rec.prev_pos = Vec2 {
            x: st.pos.x - st.vel.x,
            y: st.pos.y - st.vel.y,
        };

        c.move_restrictions = collision.get_move_restrictions(rec.prev_pos, 18.0, -1);
        let center_index = collision.get_map_index(rec.prev_pos);
        rec.move_restrictions = collision.get_move_restrictions(rec.prev_pos, 18.0, center_index);
    }

    /// `coreOf(id)` (`world.ts:413-415`).
    pub fn core_of(&self, id: i32) -> Option<&CharacterCore> {
        self.tees.get(id).map(|r| &r.core)
    }

    /// `reset()` (`world.ts:417-440`).
    pub fn reset(&mut self) {
        // `for (const rec of this.tees.values())` (`world.ts:418`) — the *current* `tees` `Map`
        // (one entry per id, the latest `addTee` wins on a duplicate — see the module doc
        // comment's "orphan" quirk), not `order`/`byId` and not this port's raw slot storage
        // (which, unlike the `Map`, can hold an orphaned duplicate slot for a reused id — visiting
        // it here would call `reset()` on the same *record* twice for one iteration of `ids` and,
        // for a genuine orphan, touch a slot TS's `tees.values()` can never reach at all).
        // `id_index.keys()` has no defined order (`HashMap`); sorted ascending here for a
        // deterministic, reproducible replay — harmless for *this* method's own outcome (every
        // tee ends fully reset regardless of order) but still the right default for a crate whose
        // whole purpose is byte-reproducible traces.
        let mut ids: Vec<i32> = self.tees.id_index.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            // `core.reset()` (`characterCore.ts:100`) calls `this.setHookedPlayer(-1)`
            // (`characterCore.ts:110`), which can clear this tee's id out of some *other* core's
            // `attachedPlayers` — done here via the full cross-tee setter before the plain field
            // reset below, matching the net effect of TS's `core.reset()` call.
            let Some(slot) = self.tees.index_of(id) else { continue };
            self.set_hooked_player(slot, -1);
            let Some(rec) = self.tees.get_mut(id) else { continue };
            rec.core.reset();
            rec.core.pos = rec.spawn_pos;
            rec.prev_pos = rec.spawn_pos;
            rec.core.active_weapon = WEAPON_GUN;
            rec.alive = true;
            rec.weapons = default_weapons(self.infinite_ammo, self.all_weapons);
            rec.queued_weapon = -1;
            rec.reload_timer = 0;
            rec.attack_tick = 0;
            rec.freeze_ticks_left = 0;
            rec.frozen_last_tick = false;
            rec.deep_frozen = false;
            rec.tele_checkpoint = 0;
            rec.move_restrictions = 0;
            rec.input = empty_input();
            rec.prev_input_for_edge = empty_input();
            rec.respawn_at_tick = None;
        }
        self.projectiles_list.clear();
        self.lasers_list.clear();
        self.tick = 0;
    }

    /// `getTee(id)` (`world.ts:442-445`).
    pub fn get_tee(&self, id: i32) -> Option<TeeState> {
        self.tees.get(id).map(|rec| self.fill(rec, blank_tee_state()))
    }

    /// `allTees()` (`world.ts:447-451`). TS iterates `tees.values()` (`Map` insertion order).
    /// **Deviation from TS, deliberately accepted:** this port instead returns ids in ascending
    /// order (`by_id`'s own order) — `id_index` (a `HashMap`) does not track insertion order at
    /// all, and reproducing a JS `Map`'s exact iteration order would need a parallel
    /// insertion-order list purely for this one method. The only call site in `src/plan`/
    /// `src/env` (`src/plan/seal.ts:84`, `for (const other of world.allTees()) if (other.id !==
    /// id) world.removeTee(other.id)`) does not depend on order at all (it just removes every
    /// other tee), so this is inert for every real caller; documented, not exercised, per the
    /// crate README.
    pub fn all_tees(&self) -> Vec<TeeState> {
        let mut ids: Vec<i32> = self.tees.id_index.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .map(|id| self.fill(self.tees.get(id).unwrap(), blank_tee_state()))
            .collect()
    }

    fn fill(&self, rec: &TeeRecord, mut out: TeeState) -> TeeState {
        let c = &rec.core;
        out.id = c.id;
        out.alive = rec.alive;
        out.pos = c.pos;
        out.vel = c.vel;
        out.hook_state = c.hook_state;
        out.hook_pos = c.hook_pos;
        out.hook_dir = c.hook_dir;
        out.hooked_player = c.hooked_player;
        out.jumped = c.jumped;
        out.jumps_left = compute_jumps_left(c, &self.collision);
        out.direction = c.direction;
        out.angle = c.angle as f64;
        out.active_weapon = c.active_weapon;
        out.frozen = rec.freeze_ticks_left > 0;
        out.freeze_ticks_left = rec.freeze_ticks_left;
        out.attack_tick = rec.attack_tick;
        out.hook_tick = Some(c.hook_tick);
        out.jumped_total = Some(c.jumped_total);
        out.reload_ticks = Some(rec.reload_timer);
        out.frozen_for = if rec.freeze_ticks_left > 0 {
            Some(self.tick - c.freeze_start)
        } else {
            None
        };
        out.deep_frozen = Some(rec.deep_frozen);
        out
    }

    /// `readTee(id, out)` (`world.ts:504-507`).
    pub fn read_tee(&self, id: i32, out: &mut TeeState) -> bool {
        match self.tees.get(id) {
            Some(rec) => {
                *out = self.fill(rec, std::mem::replace(out, blank_tee_state()));
                true
            }
            None => false,
        }
    }

    /// `isAlive(id)` (`world.ts:520-522`).
    pub fn is_alive(&self, id: i32) -> bool {
        self.tees.get(id).map(|r| r.alive).unwrap_or(false)
    }

    /// `teePos(id)` (`world.ts:524-527`).
    pub fn tee_pos(&self, id: i32) -> Option<Vec2> {
        self.tees.get(id).filter(|r| r.alive).map(|r| r.core.pos)
    }

    /// `intersectCharacter(pos0, pos1, radius, excludeId, onlyId)` (`world.ts:529-549`). Iterates
    /// `this.order` (`world.ts:532`), not `byId`.
    pub fn intersect_character(
        &self,
        pos0: Vec2,
        pos1: Vec2,
        radius: f64,
        exclude_id: i32,
        only_id: i32,
    ) -> Option<(i32, Vec2)> {
        let mut closest_len = vdistance(pos0, pos1) * 100.0;
        let mut found = None;
        for &i in &self.tees.order {
            let rec = &self.tees.slots[i];
            if !rec.alive || rec.core.id == exclude_id {
                continue;
            }
            if only_id != -1 && rec.core.id != only_id {
                continue;
            }
            let Some(ip) = closest_point_on_line_or_null(pos0, pos1, rec.core.pos) else {
                continue;
            };
            let len = vdistance(rec.core.pos, ip);
            if len < PHYSICAL_SIZE + radius {
                let len2 = vdistance(pos0, ip);
                if len2 < closest_len {
                    closest_len = len2;
                    found = Some((rec.core.id, ip));
                }
            }
        }
        found
    }

    /// `findCharactersInRadius(pos, radius)` (`world.ts:551-558`). Iterates `this.order`.
    pub fn find_characters_in_radius(&self, pos: Vec2, radius: f64) -> Vec<i32> {
        let mut out = Vec::new();
        for &i in &self.tees.order {
            let rec = &self.tees.slots[i];
            if rec.alive && vdistance(rec.core.pos, pos) < radius + PHYSICAL_SIZE {
                out.push(rec.core.id);
            }
        }
        out
    }

    /// `applyForce(id, force)` (`world.ts:560-564`).
    pub fn apply_force(&mut self, id: i32, force: Vec2) {
        let Some(rec) = self.tees.get_mut(id) else { return };
        if !rec.alive {
            return;
        }
        rec.core.vel = crate::collision::clamp_vel(rec.move_restrictions, vadd(rec.core.vel, force));
    }

    /// `unfreeze(id)` (`world.ts:566-569`).
    pub fn unfreeze(&mut self, id: i32) {
        if let Some(rec) = self.tees.get_mut(id) {
            unfreeze_tee(rec);
        }
    }

    /// `setGrenades(list)` (`world.ts:571-578`).
    pub fn set_grenades(&mut self, list: &[GrenadeSpec]) {
        self.projectiles_list.clear();
        let tune = tuning();
        let life = js::trunc(SERVER_TICK_SPEED * tune.grenade_lifetime) as i64;
        for g in list {
            let owner = if self.tees.get(g.owner).is_some() { g.owner } else { -1 };
            self.projectiles_list.push(Projectile::new(
                self.next_entity_id as i32,
                WEAPON_GRENADE,
                owner,
                g.spawn_pos,
                g.dir,
                self.tick - g.age_ticks,
                life - g.age_ticks,
                true,
            ));
            self.next_entity_id += 1;
        }
    }

    fn spawn_projectile(&mut self, kind: i32, owner: i32, pos: Vec2, dir: Vec2, life_span: i64, explosive: bool) {
        self.projectiles_list.push(Projectile::new(
            self.next_entity_id as i32,
            kind,
            owner,
            pos,
            dir,
            self.tick,
            life_span,
            explosive,
        ));
        self.next_entity_id += 1;
    }

    fn spawn_laser(&mut self, owner: i32, kind: i32, pos: Vec2, dir: Vec2, energy: f64, events: &mut Vec<WorldEvent>) {
        let mut laser = Laser::new(self.next_entity_id as i32, owner, kind, pos, dir, energy);
        self.next_entity_id += 1;
        laser.do_bounce(self, events);
        self.lasers_list.push(laser);
    }

    fn do_weapon_switch(&mut self, slot: usize) {
        let rec = &self.tees.slots[slot];
        if rec.reload_timer != 0 || rec.queued_weapon == -1 {
            return;
        }
        if !rec.weapons[rec.queued_weapon as usize].got {
            return;
        }
        let weapon = rec.queued_weapon;
        self.set_weapon(slot, weapon);
    }

    fn set_weapon(&mut self, slot: usize, weapon: i32) {
        let rec = &mut self.tees.slots[slot];
        if weapon == rec.core.active_weapon {
            return;
        }
        rec.queued_weapon = -1;
        rec.core.active_weapon = weapon;
        if rec.core.active_weapon < 0 || rec.core.active_weapon >= NUM_WEAPONS {
            rec.core.active_weapon = 0;
        }
    }

    /// `handleWeaponSwitch(rec, input)` (`world.ts:603-634`). Slot-based: `rec` is "self".
    fn handle_weapon_switch(&mut self, slot: usize, input: PlayerInput) {
        let rec = &self.tees.slots[slot];
        let mut wanted = rec.core.active_weapon;
        if rec.queued_weapon != -1 {
            wanted = rec.queued_weapon;
        }

        let mut anything = false;
        for i in 0..(NUM_WEAPONS - 1) {
            if rec.weapons[i as usize].got {
                anything = true;
            }
        }
        if !anything {
            return;
        }

        let mut next = count_presses(rec.prev_input_for_edge.next_weapon, input.next_weapon);
        let mut prev = count_presses(rec.prev_input_for_edge.prev_weapon, input.prev_weapon);

        if next < 128 {
            while next > 0 {
                wanted = (wanted + 1) % NUM_WEAPONS;
                if rec.weapons[wanted as usize].got {
                    next -= 1;
                }
            }
        }
        if prev < 128 {
            while prev > 0 {
                wanted = if wanted - 1 < 0 { NUM_WEAPONS - 1 } else { wanted - 1 };
                if rec.weapons[wanted as usize].got {
                    prev -= 1;
                }
            }
        }

        if input.wanted_weapon != 0 {
            wanted = input.wanted_weapon - 1;
        }

        if (0..NUM_WEAPONS).contains(&wanted) && wanted != rec.core.active_weapon && rec.weapons[wanted as usize].got {
            self.tees.slots[slot].queued_weapon = wanted;
        }

        self.do_weapon_switch(slot);
    }

    /// `fireWeapon(rec, input, events)` (`world.ts:636-713`). Slot-based: `rec` is "self"; the
    /// hammer-hit loop's "other" iterates `order` **slots** directly (already did before this
    /// refactor — `order` already held slot indices — the self-exclusion just needed to compare
    /// slots, not an id re-resolved through the current mapping, review finding F4).
    fn fire_weapon(&mut self, slot: usize, input: PlayerInput, events: &mut Vec<WorldEvent>) {
        let rec = &self.tees.slots[slot];
        if rec.reload_timer != 0 {
            return;
        }
        self.do_weapon_switch(slot);
        let rec = &self.tees.slots[slot];
        let id = rec.core.id;

        let dir = vnormalize(Vec2 {
            x: input.target_x,
            y: input.target_y,
        });
        let active_weapon = rec.core.active_weapon;
        let full_auto = active_weapon == WEAPON_GRENADE
            || active_weapon == WEAPON_SHOTGUN
            || active_weapon == WEAPON_LASER
            || rec.frozen_last_tick;

        let mut will_fire = count_presses(rec.prev_input_for_edge.fire, input.fire) > 0;
        if full_auto && (input.fire & 1) != 0 && active_weapon >= 0 && has_ammo(rec.weapons[active_weapon as usize]) {
            will_fire = true;
        }
        if !will_fire {
            return;
        }
        if rec.freeze_ticks_left > 0 {
            return;
        }
        if active_weapon < 0 || !has_ammo(rec.weapons[active_weapon as usize]) {
            return;
        }

        let this_pos = rec.core.pos;
        let proj_start_pos = vadd(this_pos, vmul(dir, PHYSICAL_SIZE * 0.75));
        let tune = tuning();

        match active_weapon {
            WEAPON_HAMMER => {
                let mut hits = 0;
                if self.sv_hit {
                    let mut order = std::mem::take(&mut self.scratch_order);
                    order.clear();
                    order.extend_from_slice(&self.tees.order);
                    for &i in &order {
                        if i == slot {
                            continue;
                        }
                        let other = &self.tees.slots[i];
                        if !other.alive {
                            continue;
                        }
                        if vdistance(other.core.pos, proj_start_pos) >= PHYSICAL_SIZE * 0.5 + PHYSICAL_SIZE {
                            continue;
                        }
                        let other_id = other.core.id;
                        let other_pos = other.core.pos;
                        let to_target = vdistance(other_pos, this_pos);
                        let hit_dir = if to_target > 0.0 {
                            vnormalize(vsub(other_pos, this_pos))
                        } else {
                            Vec2 { x: 0.0, y: -1.0 }
                        };

                        let strength = tune.hammer_strength;
                        let mut boost = vmul(vnormalize(vadd(hit_dir, Vec2 { x: 0.0, y: -1.1 })), 10.0);

                        let rec_other = &mut self.tees.slots[i];
                        let mr = rec_other.move_restrictions;
                        if mr != 0 {
                            boost = vsub(
                                crate::collision::clamp_vel(mr, vadd(rec_other.core.vel, boost)),
                                rec_other.core.vel,
                            );
                        }
                        let force = vmul(vadd(Vec2 { x: 0.0, y: -1.0 }, boost), strength);
                        rec_other.core.vel = crate::collision::clamp_vel(mr, vadd(rec_other.core.vel, force));
                        unfreeze_tee(rec_other);

                        events.push(WorldEvent::HammerHit { from: id, to: other_id });
                        hits += 1;
                    }
                    self.scratch_order = order;
                    if hits > 0 {
                        let rec = &mut self.tees.slots[slot];
                        rec.reload_timer = js::trunc(tune.hammer_hit_fire_delay * SERVER_TICK_SPEED / 1000.0) as i64;
                    }
                }
                events.push(WorldEvent::HammerFire { from: id, hits });
            }
            WEAPON_GUN => {
                let lifetime = js::trunc(SERVER_TICK_SPEED * tune.gun_lifetime) as i64;
                self.spawn_projectile(WEAPON_GUN, id, proj_start_pos, dir, lifetime, false);
            }
            WEAPON_SHOTGUN => {
                self.spawn_laser(id, WEAPON_SHOTGUN, this_pos, dir, tune.laser_reach, events);
            }
            WEAPON_GRENADE => {
                let lifetime = js::trunc(SERVER_TICK_SPEED * tune.grenade_lifetime) as i64;
                self.spawn_projectile(WEAPON_GRENADE, id, proj_start_pos, dir, lifetime, true);
            }
            WEAPON_LASER => {
                self.spawn_laser(id, WEAPON_LASER, this_pos, dir, tune.laser_reach, events);
            }
            _ => {}
        }

        let rec = &mut self.tees.slots[slot];
        let weapon_slot = &mut rec.weapons[active_weapon as usize];
        if weapon_slot.ammo > 0 {
            weapon_slot.ammo -= 1;
        }
        rec.attack_tick = self.tick;
        if rec.reload_timer == 0 && rec.core.active_weapon != -1 {
            rec.reload_timer = fire_delay_ticks(rec.core.active_weapon, &tune);
        }
    }

    /// `respawnTee(rec)` (`world.ts:715-733`).
    fn respawn_tee(&mut self, slot: usize) {
        let pos_in_order = self.tees.order.iter().position(|&i| i == slot);
        if let Some(pos) = pos_in_order {
            self.tees.order.remove(pos);
        }
        self.tees.order.insert(0, slot);

        // `rec.core.reset()` (`characterCore.ts:100`) calls `this.setHookedPlayer(-1)`
        // (`characterCore.ts:110`) as its *first* action — done here explicitly (this port's
        // `CharacterCore::reset` only zeroes the plain field, see that method's doc comment) so
        // a tee that died while hooking someone correctly detaches from that target's
        // `attachedPlayers` on respawn. Missing this was review finding F5 (a real repro: kill a
        // hooking tee, let it respawn, and the target's `attachedPlayers` kept a stale entry).
        self.set_hooked_player(slot, -1);
        let rec = &mut self.tees.slots[slot];
        rec.core.reset();
        rec.core.pos = rec.spawn_pos;
        rec.prev_pos = rec.spawn_pos;
        rec.core.active_weapon = WEAPON_GUN;
        rec.alive = true;
        rec.weapons = default_weapons(self.infinite_ammo, self.all_weapons);
        rec.queued_weapon = -1;
        rec.reload_timer = 0;
        rec.freeze_ticks_left = 0;
        rec.frozen_last_tick = false;
        rec.deep_frozen = false;
        rec.tele_checkpoint = 0;
        rec.move_restrictions = 0;
        rec.respawn_at_tick = None;
    }

    /// `kill(id)` (`world.ts:735-739`). Public, id-based API: `id` resolves through the *current*
    /// id→slot mapping (`this.tees.get(id)` in TS), matching TS exactly — `kill` can never target
    /// a specific orphan (TS itself has no way to either: it too only has `id` to go on).
    pub fn kill(&mut self, id: i32) {
        let Some(slot) = self.tees.index_of(id) else { return };
        let mut events = std::mem::take(&mut self.pending_events);
        self.die_slot(slot, id, &mut events);
        self.pending_events = events;
    }

    /// `die(id, rec, by, events)` (`world.ts:742-749`). Slot-based: TS's `rec` parameter is a
    /// specific object (could be an orphan when called from `handle_tiles`'s self-death path),
    /// and `id` there is always `rec.core.id` (both call sites — `kill`'s external parameter,
    /// which was itself resolved from `rec`, and `handleTiles`' `rec.core.id` — agree), so this
    /// port reads it directly from the slot instead of taking a redundant `id` parameter.
    fn die_slot(&mut self, slot: usize, by: i32, events: &mut Vec<WorldEvent>) {
        let rec = &mut self.tees.slots[slot];
        if !rec.alive {
            return;
        }
        let id = rec.core.id;
        rec.alive = false;
        events.push(WorldEvent::Death { id, by });
        if self.respawn_delay_ticks > 0 {
            rec.respawn_at_tick = Some(self.tick + self.respawn_delay_ticks);
        }
    }

    /// `resetHook(rec)` (`world.ts:751-757`). Slot-based: `rec` is "self" (the specific tee being
    /// processed by `handleTiles`, possibly an orphan).
    fn reset_hook(&mut self, slot: usize) {
        self.set_hooked_player(slot, -1);
        let rec = &mut self.tees.slots[slot];
        rec.core.hook_state = HOOK_RETRACTED;
        rec.core.triggered_events |= COREEVENT_HOOK_RETRACT;
        rec.core.hook_pos = rec.core.pos;
    }

    /// `releaseHooked(id)` (`world.ts:759-767`). Iterates `this.order` **slots** (not
    /// id-deduplicated — an orphan can independently have `hookedPlayer === id`, and TS's
    /// `other.core.hookedPlayer !== id` check compares the plain numeric value on whichever
    /// specific object `order[i]` is, orphan or not). `id` itself stays a value (the
    /// just-teleported tee's id), not a slot — it is only ever compared against, never used to
    /// address a "self".
    fn release_hooked(&mut self, id: i32) {
        let mut targets = std::mem::take(&mut self.scratch_targets);
        targets.clear();
        for &i in &self.tees.order {
            if self.tees.slots[i].core.hooked_player == id {
                targets.push(i);
            }
        }
        for &other_slot in &targets {
            self.set_hooked_player(other_slot, -1);
            let rec = &mut self.tees.slots[other_slot];
            rec.core.hook_state = HOOK_RETRACTED;
            rec.core.triggered_events |= COREEVENT_HOOK_RETRACT;
        }
        self.scratch_targets = targets;
    }

    /// `setHookedPlayer(hookedPlayer)` (`characterCore.ts:436-448`) — moved here (see the module
    /// doc comment) since it mutates *other* cores' `attachedPlayers`. `this_slot` plays the role
    /// of `this` in TS (a specific object, possibly an orphan — review finding F4: TS's `this` is
    /// always a specific `CharacterCore` reference, never resolved fresh from an id); `new_target`
    /// is the argument, a plain id value (matching `setHookedPlayer(other.id: number)`).
    fn set_hooked_player(&mut self, this_slot: usize, new_target: i32) {
        let this_id = self.tees.slots[this_slot].core.id;
        let prev = self.tees.slots[this_slot].core.hooked_player;
        if new_target == prev {
            return;
        }
        // `coreById(id)` (`world.ts:515-518`) is `rec && rec.alive ? rec.core : undefined` — it
        // filters by **alive**, not just "id exists", and it always resolves through the
        // *current* id→record mapping (never an orphan). `setHookedPlayer`
        // (`characterCore.ts:436-448`) looks the previous/next *target* up through exactly that
        // filtered accessor (unlike `this`, which is a live reference — see above), so if the
        // previously-hooked tee has since *died* (but not been removed from the world), the
        // `if (prev) prev.attachedPlayers.delete(this.id)` cleanup **silently does not run** —
        // the dead tee's `attachedPlayers` keeps the stale entry until (if ever) that tee
        // respawns and `core.reset()` clears it. This is a real, observed TS quirk (a real trace
        // mismatch on the `freeze` synthetic map caught an earlier version of this port using a
        // plain alive-or-not id lookup here instead), reproduced on purpose: both lookups below
        // use `self.tees.get(...).filter(|r| r.alive)`, matching `coreById` exactly, not
        // `self.tees.get_mut(...)` alone.
        if prev != -1 && self.tees.get(prev).is_some_and(|r| r.alive) {
            self.tees.get_mut(prev).unwrap().core.attached_players.delete(this_id);
        }
        if new_target != -1 && self.tees.get(new_target).is_some_and(|r| r.alive) {
            self.tees
                .get_mut(new_target)
                .unwrap()
                .core
                .attached_players
                .add(this_id);
        }
        self.tees.slots[this_slot].core.hooked_player = new_target;
    }

    /// `handleTile(rec, index, events)` (`world.ts:769-823`). Slot-based: `rec` is "self".
    fn handle_tile(&mut self, slot: usize, index: i32, events: &mut Vec<WorldEvent>) {
        let id = self.tees.slots[slot].core.id;
        let pos = self.tees.slots[slot].core.pos;
        let mr = self.collision.get_move_restrictions(pos, 18.0, index);
        self.tees.slots[slot].move_restrictions = mr;
        if index < 0 {
            return;
        }
        let tele_type = self.collision.tele_type_at_index(index);
        let tele_number = self.collision.tele_number_at_index(index);
        let tile = self.collision.tiles[index as usize];

        {
            let rec = &mut self.tees.slots[slot];
            if tele_type as u8 == TILE_TELECHECK && tele_number != 0 {
                rec.tele_checkpoint = tele_number;
            }

            if tile == TILE_FREEZE && !rec.deep_frozen {
                if freeze_tee(self.tick, rec, FREEZE_SECONDS) {
                    events.push(WorldEvent::Freeze { id, by: -1 });
                }
            } else if tile == TILE_UNFREEZE && !rec.deep_frozen {
                unfreeze_tee(rec);
            }
            let rec = &mut self.tees.slots[slot];
            if tile == TILE_DFREEZE && !rec.deep_frozen {
                rec.deep_frozen = true;
            } else if tile == TILE_DUNFREEZE && rec.deep_frozen {
                rec.deep_frozen = false;
            }

            if rec.core.vel.y > 0.0 && rec.move_restrictions & CANTMOVE_DOWN != 0 {
                rec.core.jumped = 0;
                rec.core.jumped_total = 0;
            }
            if rec.move_restrictions != 0 {
                rec.core.vel = crate::collision::clamp_vel(rec.move_restrictions, rec.core.vel);
            }
        }

        if tele_number == 0 {
            return;
        }
        if tele_type as u8 == TILE_TELEIN || tele_type as u8 == TILE_TELEINEVIL {
            let outs = self.collision.tele_outs_for(tele_number);
            if outs.is_empty() {
                return;
            }
            let dest = outs[0];
            let evil = tele_type as u8 == TILE_TELEINEVIL;
            self.tees.slots[slot].core.pos = dest;
            if evil {
                self.tees.slots[slot].core.vel = Vec2 { x: 0.0, y: 0.0 };
                self.reset_hook(slot);
                self.release_hooked(id);
            } else {
                self.reset_hook(slot);
            }
            return;
        }
        if tele_type as u8 == TILE_TELECHECKINEVIL || tele_type as u8 == TILE_TELECHECKIN {
            let evil = tele_type as u8 == TILE_TELECHECKINEVIL;
            let checkpoint = self.tees.slots[slot].tele_checkpoint;
            let mut dest = None;
            let mut k = checkpoint;
            while k >= 1 && dest.is_none() {
                let outs = self.collision.tele_check_outs_for(k);
                if !outs.is_empty() {
                    dest = Some(outs[0]);
                }
                k -= 1;
            }
            let spawn_pos = self.tees.slots[slot].spawn_pos;
            {
                let rec = &mut self.tees.slots[slot];
                rec.core.pos = dest.unwrap_or(spawn_pos);
                if evil {
                    rec.core.vel = Vec2 { x: 0.0, y: 0.0 };
                }
            }
            self.reset_hook(slot);
            if evil {
                self.release_hooked(id);
            }
        }
    }

    /// `applySpeedup(rec, index)` (`world.ts:825-867`). Slot-based: `rec` is "self".
    fn apply_speedup(&mut self, slot: usize, index: i32) {
        let Some(speed) = self.collision.speedup_at(index) else {
            return;
        };
        let rec = &mut self.tees.slots[slot];
        let mut vel = rec.core.vel;
        let force = speed.force;
        let mut max_speed = speed.max_speed;
        if force == 255.0 && max_speed != 0.0 {
            let k = js::trunc(max_speed / 5.0);
            vel.x = speed.dir_x * k;
            vel.y = speed.dir_y * k;
            rec.core.vel = vel;
            return;
        }
        let mr = rec.move_restrictions;
        if max_speed > 0.0 && max_speed < 5.0 {
            max_speed = 5.0;
        }
        if max_speed > 0.0 {
            let old_angle = |x: f64, y: f64| -> f64 {
                let mut a;
                if x > 0.0000001 {
                    a = -js::atan(y / x);
                } else if x < 0.0000001 {
                    a = js::atan(y / x) + js::PI;
                } else if y > 0.0000001 {
                    a = js::PI / 2.0;
                } else {
                    a = -js::PI / 2.0;
                }
                if a < 0.0 {
                    a += 2.0 * js::PI;
                }
                a
            };
            let speeder_angle = old_angle(speed.dir_x, speed.dir_y);
            let tee_angle = old_angle(vel.x, vel.y);
            let tee_speed = js::sqrt(vel.x * vel.x + vel.y * vel.y);
            let speed_left = max_speed / 5.0 - js::cos(speeder_angle - tee_angle) * tee_speed;

            if speed_left.is_nan() {
                return;
            }
            let add = if js::abs(js::trunc(speed_left)) > force && speed_left > 0.0000001 {
                force
            } else if js::abs(js::trunc(speed_left)) > force {
                -force
            } else {
                speed_left
            };
            vel.x += speed.dir_x * add;
            vel.y += speed.dir_y * add;
        } else {
            vel.x += speed.dir_x * force;
            vel.y += speed.dir_y * force;
        }

        rec.core.vel = if mr != 0 {
            crate::collision::clamp_vel(mr, vel)
        } else {
            vel
        };
    }

    /// `handleTiles(rec, events)` (`world.ts:869-898`). Slot-based: `rec` is "self".
    fn handle_tiles(&mut self, slot: usize, events: &mut Vec<WorldEvent>) {
        let (pos, prev_pos) = {
            let rec = &self.tees.slots[slot];
            (rec.core.pos, rec.prev_pos)
        };
        let off = PHYSICAL_SIZE / 3.0;
        let death_hit = self.collision.is_death(pos.x + off, pos.y - off)
            || self.collision.is_death(pos.x + off, pos.y + off)
            || self.collision.is_death(pos.x - off, pos.y - off)
            || self.collision.is_death(pos.x - off, pos.y + off);

        let centre = self.collision.get_tile_index(pos.x, pos.y);
        let is_in_freeze = death_hit
            || centre == TILE_FREEZE
            || centre == TILE_DFREEZE
            || centre == TILE_LFREEZE
            || centre == TILE_DEATH;
        self.tees.slots[slot].core.is_in_freeze = is_in_freeze;

        if death_hit || is_game_layer_clipped(pos, &self.collision) {
            self.die_slot(slot, -1, events);
            return;
        }

        let current_index = self.collision.get_map_index(pos);
        if current_index >= 0 {
            self.apply_speedup(slot, current_index);
        }

        let mut indices = std::mem::take(&mut self.scratch_tile_indices);
        self.collision.get_map_indices(prev_pos, pos, &mut indices);
        if !indices.is_empty() {
            for &index in &indices {
                self.handle_tile(slot, index, events);
                if !self.tees.slots[slot].alive {
                    self.scratch_tile_indices = indices;
                    return;
                }
            }
        } else {
            self.handle_tile(slot, current_index, events);
        }
        self.scratch_tile_indices = indices;
    }

    /// `tickEntities(events)` (`world.ts:900-914`).
    fn tick_entities(&mut self, events: &mut Vec<WorldEvent>) {
        let mut projectiles = std::mem::take(&mut self.projectiles_list);
        let mut i = 0;
        while i < projectiles.len() {
            {
                let mut p = std::mem::replace(&mut projectiles[i], Projectile::dummy());
                p.tick(self, events);
                projectiles[i] = p;
            }
            if projectiles[i].marked_for_destroy {
                projectiles.remove(i);
            } else {
                i += 1;
            }
        }
        self.projectiles_list = projectiles;

        let mut lasers = std::mem::take(&mut self.lasers_list);
        let mut i = 0;
        while i < lasers.len() {
            {
                let mut l = std::mem::replace(&mut lasers[i], Laser::dummy());
                l.tick(self, events);
                lasers[i] = l;
            }
            if lasers[i].marked_for_destroy {
                lasers.remove(i);
            } else {
                i += 1;
            }
        }
        self.lasers_list = lasers;
    }

    /// `preTick(rec, doDeferred)` (`world.ts:916-934`). Slot-based: `rec` is "self".
    fn pre_tick(&mut self, slot: usize, do_deferred: bool) {
        let rec = &mut self.tees.slots[slot];
        let input = rec.input;
        let core_input = if rec.freeze_ticks_left > 0 {
            rec.freeze_ticks_left -= 1;
            if rec.freeze_ticks_left == 1 {
                unfreeze_tee(rec);
            }
            let mut frozen = input;
            frozen.direction = 0;
            frozen.jump = 0;
            frozen.hook = 0;
            frozen
        } else {
            input
        };
        rec.core.input = core_input;
        self.core_tick(slot, true, do_deferred);
    }

    /// `CharacterCore.tick(useInput, doDeferred)` (`characterCore.ts:128-307`). Slot-based:
    /// `this` is "self".
    fn core_tick(&mut self, slot: usize, use_input: bool, do_deferred: bool) {
        let pos = self.tees.slots[slot].core.pos;
        self.tees.slots[slot].core.move_restrictions = self.collision.get_move_restrictions(pos, 18.0, -1);
        self.tees.slots[slot].core.triggered_events = 0;

        let grounded = self
            .collision
            .is_solid(pos.x + PHYSICAL_SIZE / 2.0, pos.y + PHYSICAL_SIZE / 2.0 + 5.0)
            || self
                .collision
                .is_solid(pos.x - PHYSICAL_SIZE / 2.0, pos.y + PHYSICAL_SIZE / 2.0 + 5.0);

        let tune = tuning();
        self.tees.slots[slot].core.vel.y += tune.gravity;

        let max_speed = if grounded {
            tune.ground_control_speed
        } else {
            tune.air_control_speed
        };
        let accel = if grounded {
            tune.ground_control_accel
        } else {
            tune.air_control_accel
        };
        let friction = if grounded {
            tune.ground_friction
        } else {
            tune.air_friction
        };

        if use_input {
            let input = self.tees.slots[slot].core.input;
            self.tees.slots[slot].core.direction = input.direction;

            let pi = js::PI;
            let tmp_angle = js::atan2(input.target_y, input.target_x);
            let angle = if tmp_angle < -(pi / 2.0) {
                js::trunc((tmp_angle + 2.0 * pi) * 256.0)
            } else {
                js::trunc(tmp_angle * 256.0)
            };
            self.tees.slots[slot].core.angle = angle as i32;

            if input.jump != 0 {
                let core = &mut self.tees.slots[slot].core;
                if core.jumped & 1 == 0 {
                    if grounded && (core.jumped & 2 == 0 || core.jumps != 0) {
                        core.triggered_events |= COREEVENT_GROUND_JUMP;
                        core.vel.y = -tune.ground_jump_impulse;
                        if core.jumps > 1 {
                            core.jumped |= 1;
                        } else {
                            core.jumped |= 3;
                        }
                        core.jumped_total = 0;
                    } else if core.jumped & 2 == 0 {
                        core.triggered_events |= COREEVENT_AIR_JUMP;
                        core.vel.y = -tune.air_jump_impulse;
                        core.jumped |= 3;
                        core.jumped_total += 1;
                    }
                }
            } else {
                self.tees.slots[slot].core.jumped &= !1;
            }

            if input.hook != 0 {
                if self.tees.slots[slot].core.hook_state == HOOK_IDLE {
                    let target_direction = vnormalize(Vec2 {
                        x: input.target_x,
                        y: input.target_y,
                    });
                    let pos = self.tees.slots[slot].core.pos;
                    let core = &mut self.tees.slots[slot].core;
                    core.hook_state = HOOK_FLYING;
                    core.hook_pos = vadd(pos, vmul(target_direction, PHYSICAL_SIZE * 1.5));
                    core.hook_dir = target_direction;
                    core.triggered_events |= COREEVENT_HOOK_LAUNCH;
                    let hook_tick = js::trunc(SERVER_TICK_SPEED * (1.25 - tune.hook_duration)) as i64;
                    core.hook_tick = hook_tick;
                    self.set_hooked_player(slot, -1);
                }
            } else {
                self.set_hooked_player(slot, -1);
                let core = &mut self.tees.slots[slot].core;
                core.hook_state = HOOK_IDLE;
                core.hook_pos = core.pos;
            }
        }

        if grounded {
            let core = &mut self.tees.slots[slot].core;
            core.jumped &= !2;
            core.jumped_total = 0;
        }

        {
            let core = &mut self.tees.slots[slot].core;
            if core.direction < 0 {
                core.vel.x = saturated_add(-max_speed, max_speed, core.vel.x, -accel);
            }
            if core.direction > 0 {
                core.vel.x = saturated_add(-max_speed, max_speed, core.vel.x, accel);
            }
            if core.direction == 0 {
                core.vel.x *= friction;
            }
        }

        let hook_state = self.tees.slots[slot].core.hook_state;
        if hook_state == HOOK_IDLE {
            self.set_hooked_player(slot, -1);
            let core = &mut self.tees.slots[slot].core;
            core.hook_pos = core.pos;
        } else if (HOOK_RETRACT_START..HOOK_RETRACT_END).contains(&hook_state) {
            self.tees.slots[slot].core.hook_state += 1;
        } else if hook_state == HOOK_RETRACT_END {
            self.tees.slots[slot].core.triggered_events |= COREEVENT_HOOK_RETRACT;
            self.tees.slots[slot].core.hook_state = HOOK_RETRACTED;
        } else if hook_state == HOOK_FLYING {
            self.core_tick_hook_flying(slot, &tune);
        }

        if self.tees.slots[slot].core.hook_state == HOOK_GRABBED {
            self.core_tick_hook_grabbed(slot, &tune);
        }

        if do_deferred {
            self.core_tick_deferred(slot);
        }
    }

    /// The `hookState === HOOK_FLYING` branch of `CharacterCore.tick` (`characterCore.ts:216-265`),
    /// split out only to keep [`Self::core_tick`] a manageable size — not a separate TS function.
    /// Slot-based: `this` is "self".
    fn core_tick_hook_flying(&mut self, slot: usize, tune: &crate::tuning::Tuning) {
        let core = &self.tees.slots[slot].core;
        let hook_base = core.pos;
        let mut new_pos = vadd(core.hook_pos, vmul(core.hook_dir, tune.hook_fire_speed));
        let hook_pos = core.hook_pos;
        if vdistance(hook_base, new_pos) > tune.hook_length {
            self.tees.slots[slot].core.hook_state = HOOK_RETRACT_START;
            new_pos = vadd(hook_base, vmul(vnormalize(vsub(new_pos, hook_base)), tune.hook_length));
        }

        let mut going_to_hit_ground = false;
        let mut going_to_retract = false;

        let hit = self.collision.intersect_line_hook(hook_pos, new_pos);
        if hit.collision != 0 {
            if hit.collision & CFLAG_NOHOOK != 0 {
                going_to_retract = true;
            } else {
                going_to_hit_ground = true;
            }
            new_pos = hit.out_pos;
        }

        let hook_hit_disabled = self.tees.slots[slot].core.hook_hit_disabled;
        let new_hook = self.tees.slots[slot].core.new_hook;
        let hook_state_now = self.tees.slots[slot].core.hook_state;
        if !hook_hit_disabled && tune.player_hooking != 0.0 && (hook_state_now == HOOK_FLYING || !new_hook) {
            let this_solo = self.tees.slots[slot].core.solo;
            let mut best_distance = 0.0;
            let mut other_slots = std::mem::take(&mut self.scratch_slots);
            self.tees.alive_slots_by_id_into(&mut other_slots);
            for &other_slot in &other_slots {
                // `other === this` (`characterCore.ts:240`) — reference equality: skip this exact
                // *object* (slot), not "whatever the current record for my id is" — an orphan
                // sharing my id is a distinct `other` TS would happily consider (review F4).
                if other_slot == slot {
                    continue;
                }
                let other_solo = self.tees.slots[other_slot].core.solo;
                if other_solo || this_solo {
                    continue;
                }
                let this_hook_pos = self.tees.slots[slot].core.hook_pos;
                let other_pos = self.tees.slots[other_slot].core.pos;
                let closest = closest_point_on_line_or_null(this_hook_pos, new_pos, other_pos);
                if let Some(closest) = closest
                    && vdistance(other_pos, closest) < PHYSICAL_SIZE + 2.0
                {
                    let d = vdistance(this_hook_pos, other_pos);
                    // TS (`characterCore.ts:245-250`): `if (this.hookedPlayer === -1 || d <
                    // bestDistance) { ...; this.setHookedPlayer(other.id); bestDistance = d; }` —
                    // `hookedPlayer` is mutated *inside* this same loop by that call, read fresh
                    // (live) on every iteration — ported literally here (`set_hooked_player`
                    // called *inside* the loop, `hooked_player` re-read from the slot each time),
                    // **not** deferred to "pick the single best candidate, then set once after the
                    // loop" the way an earlier version of this port did. That earlier version was
                    // wrong in two ways review found real repros for: (a) if `hookedPlayer` was
                    // already something other than `-1` *before* this scan started (e.g. injected
                    // via `applyTeeState` with an inconsistent `hookState`/`hookedPlayer` pair),
                    // TS can never grab anyone this call (every real distance `d >= 0` fails the
                    // "first free acceptance" `d < bestDistance(0)` check) — the deferred version
                    // grabbed the first candidate regardless; (b) TS's `setHookedPlayer` runs its
                    // full attach/detach bookkeeping (`characterCore.ts:436-448`) on *every*
                    // accepted candidate, not just the final one, so an earlier accepted-then-
                    // superseded candidate's `attachedPlayers` entry is correctly removed again
                    // when a closer one replaces it — the deferred version, calling
                    // `set_hooked_player` only once at the end, skipped that intermediate
                    // detach/reattach entirely.
                    let currently_hooked = self.tees.slots[slot].core.hooked_player;
                    if currently_hooked == -1 || d < best_distance {
                        self.tees.slots[slot].core.triggered_events |= COREEVENT_HOOK_ATTACH_PLAYER;
                        self.tees.slots[slot].core.hook_state = HOOK_GRABBED;
                        let other_id = self.tees.slots[other_slot].core.id;
                        self.set_hooked_player(slot, other_id);
                        best_distance = d;
                    }
                }
            }
            self.scratch_slots = other_slots;
        }

        if self.tees.slots[slot].core.hook_state == HOOK_FLYING {
            if going_to_hit_ground {
                self.tees.slots[slot].core.triggered_events |= COREEVENT_HOOK_ATTACH_GROUND;
                self.tees.slots[slot].core.hook_state = HOOK_GRABBED;
            } else if going_to_retract {
                self.tees.slots[slot].core.triggered_events |= COREEVENT_HOOK_HIT_NOHOOK;
                self.tees.slots[slot].core.hook_state = HOOK_RETRACT_START;
            }
            self.tees.slots[slot].core.hook_pos = new_pos;
        }
    }

    /// The `hookState === HOOK_GRABBED` branch of `CharacterCore.tick` (`characterCore.ts:267-304`).
    /// Slot-based: `this` is "self".
    fn core_tick_hook_grabbed(&mut self, slot: usize, tune: &crate::tuning::Tuning) {
        let hooked_player = self.tees.slots[slot].core.hooked_player;
        if hooked_player != -1 {
            match self.tees.get(hooked_player).filter(|r| r.alive) {
                Some(other) => {
                    let other_pos = other.core.pos;
                    self.tees.slots[slot].core.hook_pos = other_pos;
                }
                None => {
                    self.set_hooked_player(slot, -1);
                    self.tees.slots[slot].core.hook_state = HOOK_RETRACTED;
                    let pos = self.tees.slots[slot].core.pos;
                    self.tees.slots[slot].core.hook_pos = pos;
                }
            }
        }

        let hooked_player = self.tees.slots[slot].core.hooked_player;
        let hook_pos = self.tees.slots[slot].core.hook_pos;
        let pos = self.tees.slots[slot].core.pos;
        if hooked_player == -1 && vdistance(hook_pos, pos) > 46.0 {
            let mut hook_vel = vmul(vnormalize(vsub(hook_pos, pos)), tune.hook_drag_accel);
            if hook_vel.y > 0.0 {
                hook_vel = Vec2 {
                    x: hook_vel.x,
                    y: hook_vel.y * 0.3,
                };
            }
            let direction = self.tees.slots[slot].core.direction;
            if (hook_vel.x < 0.0 && direction < 0) || (hook_vel.x > 0.0 && direction > 0) {
                hook_vel = Vec2 {
                    x: hook_vel.x * 0.95,
                    y: hook_vel.y,
                };
            } else {
                hook_vel = Vec2 {
                    x: hook_vel.x * 0.75,
                    y: hook_vel.y,
                };
            }

            let vel = self.tees.slots[slot].core.vel;
            let new_vel = vadd(vel, hook_vel);
            let new_vel_length = crate::vmath::vlength(new_vel);
            if new_vel_length < tune.hook_drag_speed || new_vel_length < crate::vmath::vlength(vel) {
                self.tees.slots[slot].core.vel = new_vel;
            }
        }

        self.tees.slots[slot].core.hook_tick += 1;
        let hooked_player = self.tees.slots[slot].core.hooked_player;
        let hooked_gone = hooked_player != -1 && self.tees.get(hooked_player).filter(|r| r.alive).is_none();
        let hook_tick = self.tees.slots[slot].core.hook_tick;
        if hooked_player != -1 && (hook_tick > SERVER_TICK_SPEED as i64 + SERVER_TICK_SPEED as i64 / 5 || hooked_gone) {
            self.set_hooked_player(slot, -1);
            self.tees.slots[slot].core.hook_state = HOOK_RETRACTED;
            let pos = self.tees.slots[slot].core.pos;
            self.tees.slots[slot].core.hook_pos = pos;
        }
    }

    /// `CharacterCore.tickDeferred()` (`characterCore.ts:309-355`). Slot-based: `this` is "self";
    /// "other" iterates `allCores()` **slots** (see `alive_slots_by_id_into`'s doc comment — an
    /// orphan is a distinct `other` from the current record sharing its id, review finding F4).
    fn core_tick_deferred(&mut self, this_idx: usize) {
        let mut other_slots = std::mem::take(&mut self.scratch_slots);
        self.tees.alive_slots_by_id_into(&mut other_slots);
        for &other_idx in &other_slots {
            if other_idx == this_idx {
                continue;
            }
            let (this_rec, other_rec) = self.tees.pair_mut(this_idx, other_idx);
            let this_core = &mut this_rec.core;
            let other_core = &mut other_rec.core;
            if this_core.solo || other_core.solo {
                continue;
            }
            let dist = vdistance(this_core.pos, other_core.pos);
            if dist > 0.0 {
                let tune = tuning();
                let can_collide =
                    !this_core.collision_disabled && !other_core.collision_disabled && tune.player_collision != 0.0;
                if can_collide && dist < PHYSICAL_SIZE * 1.25 {
                    let dir = vnormalize(vsub(this_core.pos, other_core.pos));
                    let a = PHYSICAL_SIZE * 1.45 - dist;
                    let mut velocity = 0.5;
                    if crate::vmath::vlength(this_core.vel) > 0.0001 {
                        velocity = 1.0 - (crate::vmath::vdot(vnormalize(this_core.vel), dir) + 1.0) / 2.0;
                    }
                    this_core.vel = vadd(this_core.vel, vmul(dir, a * (velocity * 0.75)));
                    this_core.vel = vmul(this_core.vel, 0.85);
                }

                // TS itself nests these two conditions as separate `if`s
                // (`characterCore.ts:329-344`: `if (!hookHitDisabled && hookedPlayer===other.id
                // && playerHooking) { if (dist > PHYSICAL_SIZE*1.5) { ... } }`) — kept nested
                // here for a literal line-for-line mirror rather than merged with `&&`.
                #[allow(clippy::collapsible_if)]
                if !this_core.hook_hit_disabled
                    && this_core.hooked_player == other_core.id
                    && tune.player_hooking != 0.0
                {
                    if dist > PHYSICAL_SIZE * 1.5 {
                        let dir = vnormalize(vsub(this_core.pos, other_core.pos));
                        let hook_accel = tune.hook_drag_accel * (dist / tune.hook_length);
                        let drag_speed = tune.hook_drag_speed;

                        // `other.moveRestrictions`/`this.moveRestrictions` here are
                        // `CharacterCore.moveRestrictions` (`characterCore.ts:71`, refreshed at
                        // the top of `core.tick()` this same tick), **not**
                        // `TeeRecord.moveRestrictions` (`world.ts:127`, refreshed later by
                        // `handleTile` with an index override) — the two fields share a name but
                        // are computed differently and are not interchangeable; using the wrong
                        // one here was review-caught by a real trace mismatch on the `front`
                        // synthetic map (a `STOP`-tile scenario where the two values differ).
                        other_core.vel = crate::collision::clamp_vel(
                            other_core.move_restrictions,
                            Vec2 {
                                x: saturated_add(-drag_speed, drag_speed, other_core.vel.x, hook_accel * dir.x * 1.5),
                                y: saturated_add(-drag_speed, drag_speed, other_core.vel.y, hook_accel * dir.y * 1.5),
                            },
                        );
                        this_core.vel = crate::collision::clamp_vel(
                            this_core.move_restrictions,
                            Vec2 {
                                x: saturated_add(-drag_speed, drag_speed, this_core.vel.x, -hook_accel * dir.x * 0.25),
                                y: saturated_add(-drag_speed, drag_speed, this_core.vel.y, -hook_accel * dir.y * 0.25),
                            },
                        );
                    }
                }
            }
        }
        self.scratch_slots = other_slots;

        if self.tees.slots[this_idx].core.hook_state != HOOK_FLYING {
            self.tees.slots[this_idx].core.new_hook = false;
        }

        if crate::vmath::vlength(self.tees.slots[this_idx].core.vel) > 6000.0 {
            self.tees.slots[this_idx].core.vel = vmul(vnormalize(self.tees.slots[this_idx].core.vel), 6000.0);
        }
    }

    /// `CharacterCore.move()` (`characterCore.ts:357-418`). Slot-based: `this` is "self".
    fn core_move(&mut self, slot: usize) {
        let tune = tuning();
        let vel = self.tees.slots[slot].core.vel;
        let ramp_value = velocity_ramp(
            crate::vmath::vlength(vel) * 50.0,
            tune.velramp_start,
            tune.velramp_range,
            tune.velramp_curvature,
        );
        self.tees.slots[slot].core.vel.x *= ramp_value;

        let mut new_pos = self.tees.slots[slot].core.pos;
        let mut new_vel = self.tees.slots[slot].core.vel;
        let old_vel_x = self.tees.slots[slot].core.vel.x;
        let elasticity = Vec2 {
            x: tune.ground_elasticity_x,
            y: tune.ground_elasticity_y,
        };
        self.collision
            .move_box(&mut new_pos, &mut new_vel, MOVE_SIZE, elasticity);
        self.tees.slots[slot].core.vel = new_vel;

        let mut colliding = 0;
        let mut left_wall = self.tees.slots[slot].core.left_wall;
        if new_vel.x < 0.001 && new_vel.x > -0.001 {
            if old_vel_x > 0.0 {
                colliding = 1;
            } else if old_vel_x < 0.0 {
                colliding = 2;
            }
        } else {
            left_wall = true;
        }
        self.tees.slots[slot].core.colliding = colliding;
        self.tees.slots[slot].core.left_wall = left_wall;

        self.tees.slots[slot].core.vel.x *= 1.0 / ramp_value;

        let collision_disabled = self.tees.slots[slot].core.collision_disabled;
        let solo = self.tees.slots[slot].core.solo;
        if tune.player_collision != 0.0 && !collision_disabled && !solo {
            let this_pos = self.tees.slots[slot].core.pos;
            let dist = vdistance(this_pos, new_pos);
            if dist > 0.0 {
                let end = js::trunc(dist + 1.0);

                let from = this_pos;
                let mut last = from;

                let mut snapshots = std::mem::take(&mut self.scratch_other_snapshots);
                snapshots.clear();
                let mut other_slots = std::mem::take(&mut self.scratch_slots);
                self.tees.alive_slots_by_id_into(&mut other_slots);
                for &other_slot in &other_slots {
                    if other_slot == slot {
                        continue;
                    }
                    let other = &self.tees.slots[other_slot];
                    snapshots.push((other.core.pos, other.core.solo, other.core.collision_disabled));
                }
                self.scratch_slots = other_slots;

                let mut i = 0.0_f64;
                let mut stop_pos: Option<Vec2> = None;
                'outer: while i < end {
                    let a = i / dist;
                    let px = from.x + (new_pos.x - from.x) * a;
                    let py = from.y + (new_pos.y - from.y) * a;
                    for &(other_pos, other_solo, other_collision_disabled) in &snapshots {
                        if solo || other_solo || other_collision_disabled {
                            continue;
                        }
                        let dx = px - other_pos.x;
                        let dy = py - other_pos.y;
                        let d = js::sqrt(dx * dx + dy * dy);
                        if d < PHYSICAL_SIZE {
                            if a > 0.0 {
                                stop_pos = Some(last);
                            } else if vdistance(new_pos, other_pos) > d {
                                stop_pos = Some(new_pos);
                            } else {
                                stop_pos = Some(this_pos);
                            }
                            break 'outer;
                        }
                    }
                    last = Vec2 { x: px, y: py };
                    i += 1.0;
                }
                self.scratch_other_snapshots = snapshots;

                if let Some(p) = stop_pos {
                    self.tees.slots[slot].core.pos = p;
                    return;
                }
            }
        }

        self.tees.slots[slot].core.pos = new_pos;
    }

    /// `step()` (`world.ts:936-997`).
    pub fn step(&mut self) -> Vec<WorldEvent> {
        self.tick += 1;

        // `this.pendingEvents.splice(0)` (`world.ts:939`) *drains* whatever `kill()` queued
        // between the previous `step()` and this one — the returned `events` array starts as
        // that drained content, not empty. Review finding F2 (BLOCKER): an earlier version of
        // this port called `.clear()` on `events` right after taking it, silently discarding a
        // `kill()`'s `death` event before this tick's own events were even pushed. `mem::take`
        // alone already leaves `self.pending_events` as an empty `Vec` behind (the moral
        // equivalent of `splice(0)` truncating the source to length 0), so nothing further is
        // needed here.
        let mut events = std::mem::take(&mut self.pending_events);

        let to_respawn: Vec<usize> = self
            .tees
            .by_id
            .iter()
            .copied()
            .filter(|&i| {
                !self.tees.slots[i].alive && self.tees.slots[i].respawn_at_tick.is_some_and(|t| self.tick >= t)
            })
            .collect();
        for slot in to_respawn {
            self.respawn_tee(slot);
        }

        // `const order = this.order;` (`world.ts:940`) is a **reference** to the same live array,
        // taken *after* the respawn loop above it (`world.ts:942-946`) has already run — so every
        // loop below sees this tick's respawns already reflected in `order` (a respawning tee is
        // moved to the front by `respawnTee`'s own `order.splice`+`order.unshift`, `world.ts:717-718`,
        // *before* `const order` is even read). Copying `self.tees.order` into a reused scratch
        // buffer here (after respawns, not before — a real trace mismatch caught an earlier
        // version of this port cloning it too early, at the top of `step()`) reproduces that
        // same "already includes this tick's respawns" snapshot without needing a live-reference
        // alias in Rust, and without a fresh heap allocation every tick.
        let mut order = std::mem::take(&mut self.scratch_order);
        order.clear();
        order.extend_from_slice(&self.tees.order);

        self.tick_entities(&mut events);

        // Every loop below operates directly on the **slot** from `order` (`self.tees.slots[slot]`),
        // never re-resolving "self" through `id_index` — TS's own loops operate on the live object
        // reference `order[i]` *is*, which can be an orphaned duplicate-id record distinct from
        // whatever `this.tees.get(id)` currently maps to (review finding F4). Re-resolving by id
        // here would silently redirect an orphan's own turn onto the *other* (current) record for
        // that id, double-processing it and never ticking the orphan at all.
        for &slot in &order {
            if !self.tees.slots[slot].alive {
                continue;
            }
            let input = self.tees.slots[slot].input;
            self.handle_weapon_switch(slot, input);
            self.fire_weapon(slot, input, &mut events);
            self.tees.slots[slot].prev_input_for_edge = input;
        }

        let no_weak_hook = self.no_weak_hook;
        if no_weak_hook {
            for &slot in &order {
                if self.tees.slots[slot].alive {
                    self.pre_tick(slot, false);
                }
            }
        }
        for &slot in &order {
            if !self.tees.slots[slot].alive {
                continue;
            }

            if no_weak_hook {
                self.core_tick_deferred(slot);
            } else {
                self.pre_tick(slot, true);
            }

            if self.tees.slots[slot].reload_timer > 0 {
                self.tees.slots[slot].reload_timer -= 1;
            } else {
                let input = self.tees.slots[slot].input;
                self.fire_weapon(slot, input, &mut events);
            }

            self.tees.slots[slot].frozen_last_tick = false;
            if self.tees.slots[slot].deep_frozen {
                let rec = &mut self.tees.slots[slot];
                freeze_tee(self.tick, rec, FREEZE_SECONDS);
            }
            apply_jump_rules(&mut self.tees.slots[slot].core);

            self.handle_tiles(slot, &mut events);
            self.tees.slots[slot].prev_pos = self.tees.slots[slot].core.pos;
        }

        for &slot in &order {
            if self.tees.slots[slot].alive {
                self.core_move(slot);
                self.tees.slots[slot].core.quantize();
            }
        }

        self.scratch_order = order;
        events
    }

    /// `projectiles()` (`world.ts:453-468`).
    pub fn projectiles(&self) -> Vec<ProjectileState> {
        self.projectiles_list
            .iter()
            .map(|p| ProjectileState {
                id: p.id,
                kind: p.kind,
                owner: p.owner,
                pos: p.pos_at_tick(self.tick),
                vel: p.vel,
                dir: p.dir,
                start_tick: p.start_tick,
                spawn_pos: p.pos,
            })
            .collect()
    }

    /// Not part of TS's public `SimWorld` API (only `lasersForTest()`, `world.ts:409-411`, which
    /// this crate's tests use through this same accessor) — exposed publicly here because the
    /// task's acceptance criteria list `lasers` alongside `projectiles` in the API `SimWorld`
    /// must mirror for the future planner port.
    pub fn lasers(&self) -> &[Laser] {
        &self.lasers_list
    }

    /// `order`, as tee ids (`world.ts`'s private `order` field, read the same way
    /// `tools/ts-trace/lib.mjs`'s `orderIds` reads it from the TS side — see that file's doc
    /// comment). Exposed publicly here (unlike TS, where it is only reachable at runtime because
    /// `private` carries no runtime enforcement) since the task's diagnostics tooling
    /// (`ts-diff`/`tests/parity_bulk.rs`) needs to compare it against a trace's own recorded
    /// `order` field to localize an iteration-order divergence.
    pub fn order_ids(&self) -> Vec<i32> {
        self.tees.order.iter().map(|&i| self.tees.slots[i].core.id).collect()
    }

    /// `byId`, as tee ids. See [`Self::order_ids`]'s doc comment.
    pub fn by_id_ids(&self) -> Vec<i32> {
        self.tees.by_id.iter().map(|&i| self.tees.slots[i].core.id).collect()
    }
}

/// One entry of [`SimWorld::set_grenades`]'s list (`world.ts:571`'s inline parameter type).
#[derive(Debug, Clone, Copy)]
pub struct GrenadeSpec {
    pub owner: i32,
    pub spawn_pos: Vec2,
    pub dir: Vec2,
    pub age_ticks: i64,
}

impl EntityWorld for SimWorld {
    fn tick(&self) -> i64 {
        self.tick
    }
    fn collision(&self) -> &Collision {
        &self.collision
    }
    fn sv_hit(&self) -> bool {
        self.sv_hit
    }
    fn is_alive(&self, id: i32) -> bool {
        SimWorld::is_alive(self, id)
    }
    fn tee_pos(&self, id: i32) -> Option<Vec2> {
        SimWorld::tee_pos(self, id)
    }
    fn intersect_character(
        &self,
        pos0: Vec2,
        pos1: Vec2,
        radius: f64,
        exclude_id: i32,
        only_id: i32,
    ) -> Option<(i32, Vec2)> {
        SimWorld::intersect_character(self, pos0, pos1, radius, exclude_id, only_id)
    }
    fn find_characters_in_radius(&self, pos: Vec2, radius: f64) -> Vec<i32> {
        SimWorld::find_characters_in_radius(self, pos, radius)
    }
    fn apply_force(&mut self, id: i32, force: Vec2) {
        SimWorld::apply_force(self, id, force);
    }
    fn unfreeze(&mut self, id: i32) {
        SimWorld::unfreeze(self, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::Collision;
    use crate::tuning::TILE_SOLID;

    fn flat_world(width: i32, height: i32) -> SimWorld {
        let n = (width * height) as usize;
        let mut tiles = vec![0u8; n];
        for x in 0..width {
            tiles[x as usize] = TILE_SOLID;
            tiles[((height - 1) * width + x) as usize] = TILE_SOLID;
        }
        for y in 0..height {
            tiles[(y * width) as usize] = TILE_SOLID;
            tiles[(y * width + width - 1) as usize] = TILE_SOLID;
        }
        let collision = Collision::new(width, height, tiles, None, None, None);
        SimWorld::new(collision, SimWorldOptions::default())
    }

    #[test]
    fn add_tee_prepends_to_order_and_sorts_by_id() {
        let mut w = flat_world(10, 10);
        w.add_tee(5, Vec2 { x: 64.0, y: 64.0 });
        w.add_tee(2, Vec2 { x: 96.0, y: 64.0 });
        w.add_tee(9, Vec2 { x: 128.0, y: 64.0 });
        // order: unshift each time -> most recently added first: 9, 2, 5
        assert_eq!(
            w.tees
                .order
                .iter()
                .map(|&i| w.tees.slots[i].core.id)
                .collect::<Vec<_>>(),
            vec![9, 2, 5]
        );
        // byId: sorted ascending
        assert_eq!(
            w.tees
                .by_id
                .iter()
                .map(|&i| w.tees.slots[i].core.id)
                .collect::<Vec<_>>(),
            vec![2, 5, 9]
        );
    }

    #[test]
    fn duplicate_add_tee_leaves_orphan_in_order_but_not_reachable_by_id() {
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 64.0, y: 64.0 });
        w.add_tee(1, Vec2 { x: 96.0, y: 96.0 });
        // Two entries with id 1 in `order` (orphan quirk), but `tees` map (id_index) points at
        // the newest one only.
        assert_eq!(w.tees.order.len(), 2);
        assert_eq!(w.get_tee(1).unwrap().pos, Vec2 { x: 96.0, y: 96.0 });
        // remove_tee(1) drops BOTH entries (TS: order.filter(id !== 1) removes every match).
        w.remove_tee(1);
        assert!(w.tees.order.is_empty());
        assert!(w.get_tee(1).is_none());
    }

    #[test]
    fn duplicate_orphan_slot_is_not_freed_until_remove_tee_sweeps_its_id() {
        // Review finding F11's own caveat, checked directly: a duplicate-`addTee` orphan (F4)
        // must NOT be treated as freeable just because it's unreachable by id — it is still
        // referenced by `order`/`by_id` (TS keeps ticking it) and must survive `add_tee`'s slot
        // reuse until an eventual `remove_tee(1)` sweeps out *both* the orphan and the current
        // record (TS filters `order`/`byId` by `core.id` value, not by object reference).
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 64.0, y: 64.0 }); // orphan-to-be, slot 0
        w.add_tee(1, Vec2 { x: 96.0, y: 96.0 }); // current, slot 1 — orphans slot 0
        assert!(
            w.tees.free_slots.is_empty(),
            "no slot is freeable yet — orphan is still referenced"
        );
        w.add_tee(2, Vec2 { x: 128.0, y: 128.0 }); // must NOT reuse slot 0 (still referenced)
        assert_eq!(w.tees.slots.len(), 3, "orphan's slot must not have been recycled early");
        w.remove_tee(1); // sweeps BOTH id-1 slots (orphan and current) out of order/by_id
        assert_eq!(w.tees.free_slots.len(), 2, "both id-1 slots become free together");
        assert_eq!(w.tees.order.len(), 1); // only tee 2 remains
    }

    #[test]
    fn remove_tee_frees_the_slot_so_repeated_add_remove_cycles_stay_bounded() {
        // Review finding F11: before this fix, `slots` grew by one on every single `add_tee`,
        // forever — 400k add/remove cycles on the same id measured `clone()` going from 1.9 µs to
        // 164 ms and 365 MB max RSS. `remove_tee` now frees the slot it swept out of `order`/
        // `by_id`, so a steady add/remove/add/remove... pattern reuses the same slot indefinitely
        // instead of leaking a new one every time.
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 64.0, y: 64.0 });
        w.remove_tee(1);
        let steady_state_len = w.tees.slots.len();
        assert_eq!(
            steady_state_len, 1,
            "one add+remove cycle should settle on exactly one slot"
        );
        for i in 0..400_000 {
            w.add_tee(1, Vec2 { x: 64.0, y: 64.0 });
            w.remove_tee(1);
            assert_eq!(
                w.tees.slots.len(),
                steady_state_len,
                "slots grew on cycle {i} — the slot was not reused"
            );
        }
        // The actual reported symptom (`clone()` cost) — a generous upper bound (not a tight
        // benchmark assertion, this is a correctness test on a possibly-shared/loaded machine):
        // with `slots.len()` bounded, `SimWorld`/`TeeStorage::clone()` must stay cheap regardless
        // of how many add/remove cycles preceded it.
        let start = std::time::Instant::now();
        let _ = w.clone();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "clone() took {elapsed:?} after 400k add/remove cycles — slots leaked"
        );
    }

    #[test]
    fn tee_falls_under_gravity_and_lands_on_floor() {
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 160.0, y: 64.0 });
        for _ in 0..200 {
            w.step();
        }
        let t = w.get_tee(1).unwrap();
        // Should have settled on the floor (y = (10-2)*32 - 14ish), not fallen through.
        assert!(t.pos.y < 300.0, "tee fell through the floor: {:?}", t.pos);
        assert!(t.alive);
    }

    #[test]
    fn save_restore_state_round_trips() {
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 160.0, y: 64.0 });
        w.add_tee(2, Vec2 { x: 200.0, y: 64.0 });
        for _ in 0..30 {
            w.step();
        }
        let saved = w.save_state();
        for _ in 0..30 {
            w.step();
        }
        w.restore_state(&saved);
        let after = w.save_state();
        assert_eq!(after, saved);
    }

    #[test]
    fn clone_is_a_full_independent_snapshot() {
        // Acceptance criterion 1: "`Clone` = full snapshot" — cloning `SimWorld` and continuing
        // to `step()` the original must never observably affect the clone (and vice versa).
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 160.0, y: 64.0 });
        w.add_tee(2, Vec2 { x: 200.0, y: 64.0 });
        for _ in 0..20 {
            w.step();
        }
        let mut cloned = w.clone();
        assert_eq!(w.save_state(), cloned.save_state());

        for _ in 0..50 {
            w.step();
        }
        // The clone, taken before those 50 extra steps, must be unaffected.
        assert_ne!(w.save_state(), cloned.save_state());
        for _ in 0..50 {
            cloned.step();
        }
        // Same input history from the same starting point -> identical resulting state.
        assert_eq!(w.save_state(), cloned.save_state());
    }

    #[test]
    fn reset_clears_every_tee_including_cross_hook_state() {
        let mut w = flat_world(20, 20);
        w.add_tee(1, Vec2 { x: 160.0, y: 160.0 });
        w.add_tee(2, Vec2 { x: 260.0, y: 160.0 });
        w.set_held_input(
            1,
            PlayerInput {
                target_x: 1.0,
                target_y: 0.0,
                hook: 1,
                ..empty_input()
            },
        );
        let mut grabbed = false;
        for _ in 0..30 {
            w.step();
            if w.core_of(1).unwrap().hook_state == HOOK_GRABBED {
                grabbed = true;
                break;
            }
        }
        assert!(grabbed);
        assert_eq!(w.core_of(1).unwrap().hooked_player, 2);
        assert_eq!(
            w.core_of(2).unwrap().attached_players.iter().collect::<Vec<_>>(),
            vec![1]
        );

        w.reset();

        for id in [1, 2] {
            let c = w.core_of(id).unwrap();
            assert_eq!(c.hooked_player, -1, "id {id}");
            assert_eq!(c.hook_state, HOOK_IDLE, "id {id}");
            assert!(c.attached_players.iter().collect::<Vec<_>>().is_empty(), "id {id}");
        }
        assert_eq!(w.tick, 0);
        let t1 = w.get_tee(1).unwrap();
        assert_eq!(t1.pos, Vec2 { x: 160.0, y: 160.0 });
        assert!(t1.alive);
    }

    #[test]
    fn reset_skips_orphaned_duplicate_add_tee_slot() {
        // A duplicate `add_tee(1, ...)` leaves an orphaned slot in `order`/`byId` (see
        // `duplicate_add_tee_leaves_orphan_in_order_but_not_reachable_by_id`) — `reset()` must
        // only touch the *current* id->slot mapping (TS's `tees.values()`), not every slot ever
        // pushed, or it would double-process (or, for a true orphan, wrongly process at all) that
        // record. This mostly asserts `reset()` doesn't panic and settles every currently
        // reachable id into a clean state.
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 96.0, y: 64.0 });
        w.add_tee(1, Vec2 { x: 160.0, y: 64.0 });
        w.add_tee(2, Vec2 { x: 200.0, y: 64.0 });
        w.reset();
        assert_eq!(w.get_tee(1).unwrap().pos, Vec2 { x: 160.0, y: 64.0 });
        assert_eq!(w.get_tee(2).unwrap().pos, Vec2 { x: 200.0, y: 64.0 });
    }

    #[test]
    fn kill_and_respawn() {
        let mut w = SimWorld::new(
            Collision::new(
                10,
                10,
                {
                    let n = 100usize;
                    let mut tiles = vec![0u8; n];
                    for i in 0..10 {
                        tiles[i] = TILE_SOLID;
                        tiles[90 + i] = TILE_SOLID;
                        tiles[i * 10] = TILE_SOLID;
                        tiles[i * 10 + 9] = TILE_SOLID;
                    }
                    tiles
                },
                None,
                None,
                None,
            ),
            SimWorldOptions {
                respawn_delay_ticks: Some(10),
                ..Default::default()
            },
        );
        w.add_tee(1, Vec2 { x: 160.0, y: 64.0 });
        w.kill(1);
        assert!(!w.is_alive(1));
        for _ in 0..20 {
            w.step();
        }
        assert!(w.is_alive(1));
    }

    #[test]
    fn matches_a_recorded_real_ts_run_50_ticks_default_input() {
        // Recorded once from the real `src/core/world.ts`/`collision.ts` on Node 24.21.0
        // (`node /tmp/t2.mjs`, same 10x10 solid-border map, one tee added at (160,64), 50
        // `step()` calls, no input ever set — TS's own default `emptyInput()`):
        // `{"pos":{"x":160,"y":273},"vel":{"x":0,"y":0},"hookState":0,"hookPos":{"x":160,"y":273},
        //  "angle":-402,"activeWeapon":1,"jumpsLeft":2,"attackTick":-1000}`.
        let mut w = flat_world(10, 10);
        w.add_tee(1, Vec2 { x: 160.0, y: 64.0 });
        for _ in 0..50 {
            w.step();
        }
        let t = w.get_tee(1).unwrap();
        assert_eq!(t.pos, Vec2 { x: 160.0, y: 273.0 });
        assert_eq!(t.vel, Vec2 { x: 0.0, y: 0.0 });
        assert_eq!(t.hook_state, 0);
        assert_eq!(t.hook_pos, Vec2 { x: 160.0, y: 273.0 });
        assert_eq!(t.angle, -402.0);
        assert_eq!(t.active_weapon, 1);
        assert_eq!(t.jumps_left, 2);
        assert_eq!(t.attack_tick, -1000);
        assert!(t.alive);
    }

    #[test]
    fn hook_grabs_other_player() {
        let mut w = flat_world(20, 20);
        w.add_tee(1, Vec2 { x: 160.0, y: 160.0 });
        w.add_tee(2, Vec2 { x: 260.0, y: 160.0 });
        w.set_held_input(
            1,
            PlayerInput {
                target_x: 1.0,
                target_y: 0.0,
                hook: 1,
                ..empty_input()
            },
        );
        let mut grabbed = false;
        for _ in 0..30 {
            w.step();
            if w.core_of(1).unwrap().hook_state == HOOK_GRABBED {
                grabbed = true;
                break;
            }
        }
        assert!(grabbed, "hook never attached to the other player");
    }
}
