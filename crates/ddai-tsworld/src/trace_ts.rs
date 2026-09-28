//! Reads trace-ts v1 files (`docs/formats.md` "trace-ts v1"; written by `tools/ts-trace/*.mjs`,
//! running the real TS `SimWorld`) and replays them against this crate's own [`crate::world::SimWorld`],
//! comparing every field bit-for-bit. Used by the `ts-diff` binary and by `tests/parity_*.rs`.

use serde::Deserialize;
use std::collections::BTreeMap;

use crate::types::{PlayerInput, TeeState, WorldEvent};
use crate::vmath::Vec2;
use crate::world::{CoreState, GrenadeSpec, SimState, SimWorldOptions, TeeStateSnapshot, WeaponSlot};

fn hex_to_f64(s: &str) -> f64 {
    let bits = u64::from_str_radix(s, 16).unwrap_or_else(|e| panic!("bad hex f64 {s:?}: {e}"));
    f64::from_bits(bits)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WireInput {
    pub direction: i32,
    pub target_x: String,
    pub target_y: String,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub player_flags: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
}

impl From<&WireInput> for PlayerInput {
    fn from(w: &WireInput) -> Self {
        PlayerInput {
            direction: w.direction,
            target_x: hex_to_f64(&w.target_x),
            target_y: hex_to_f64(&w.target_y),
            jump: w.jump,
            fire: w.fire,
            hook: w.hook,
            player_flags: w.player_flags,
            wanted_weapon: w.wanted_weapon,
            next_weapon: w.next_weapon,
            prev_weapon: w.prev_weapon,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireCoreState {
    pos_x: String,
    pos_y: String,
    vel_x: String,
    vel_y: String,
    hook_pos_x: String,
    hook_pos_y: String,
    hook_dir_x: String,
    hook_dir_y: String,
    hook_tele_base_x: String,
    hook_tele_base_y: String,
    hook_tick: i64,
    hook_state: i32,
    hooked_player: i32,
    attached_players: Vec<i32>,
    active_weapon: i32,
    new_hook: bool,
    jumped: i32,
    jumped_total: i32,
    jumps: i32,
    direction: i32,
    angle: i32,
    triggered_events: u32,
    colliding: i32,
    left_wall: bool,
    freeze_start: i64,
    freeze_end: i64,
    is_in_freeze: bool,
    move_restrictions: i32,
}

impl WireCoreState {
    fn to_core_state(&self) -> CoreState {
        CoreState {
            pos: Vec2 {
                x: hex_to_f64(&self.pos_x),
                y: hex_to_f64(&self.pos_y),
            },
            vel: Vec2 {
                x: hex_to_f64(&self.vel_x),
                y: hex_to_f64(&self.vel_y),
            },
            hook_pos: Vec2 {
                x: hex_to_f64(&self.hook_pos_x),
                y: hex_to_f64(&self.hook_pos_y),
            },
            hook_dir: Vec2 {
                x: hex_to_f64(&self.hook_dir_x),
                y: hex_to_f64(&self.hook_dir_y),
            },
            hook_tele_base: Vec2 {
                x: hex_to_f64(&self.hook_tele_base_x),
                y: hex_to_f64(&self.hook_tele_base_y),
            },
            hook_tick: self.hook_tick,
            hook_state: self.hook_state,
            hooked_player: self.hooked_player,
            attached_players: self.attached_players.clone(),
            active_weapon: self.active_weapon,
            new_hook: self.new_hook,
            jumped: self.jumped,
            jumped_total: self.jumped_total,
            jumps: self.jumps,
            direction: self.direction,
            angle: self.angle,
            triggered_events: self.triggered_events,
            colliding: self.colliding,
            left_wall: self.left_wall,
            freeze_start: self.freeze_start,
            freeze_end: self.freeze_end,
            is_in_freeze: self.is_in_freeze,
            move_restrictions: self.move_restrictions,
        }
    }
}

#[derive(Debug, Deserialize)]
struct WireWeapon {
    got: bool,
    ammo: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireTeeSnapshot {
    core: WireCoreState,
    alive: bool,
    freeze_ticks_left: i64,
    frozen_last_tick: bool,
    deep_frozen: bool,
    tele_checkpoint: i32,
    move_restrictions: i32,
    reload_timer: i64,
    attack_tick: i64,
    queued_weapon: i32,
    input: WireInput,
    prev_input_for_edge: WireInput,
    prev_pos_x: String,
    prev_pos_y: String,
    spawn_pos_x: String,
    spawn_pos_y: String,
    respawn_at_tick: Option<i64>,
    weapons: Vec<WireWeapon>,
}

impl WireTeeSnapshot {
    fn to_snapshot(&self) -> TeeStateSnapshot {
        TeeStateSnapshot {
            core: self.core.to_core_state(),
            alive: self.alive,
            freeze_ticks_left: self.freeze_ticks_left,
            frozen_last_tick: self.frozen_last_tick,
            deep_frozen: self.deep_frozen,
            tele_checkpoint: self.tele_checkpoint,
            move_restrictions: self.move_restrictions,
            reload_timer: self.reload_timer,
            attack_tick: self.attack_tick,
            queued_weapon: self.queued_weapon,
            input: PlayerInput::from(&self.input),
            prev_input_for_edge: PlayerInput::from(&self.prev_input_for_edge),
            prev_pos: Vec2 {
                x: hex_to_f64(&self.prev_pos_x),
                y: hex_to_f64(&self.prev_pos_y),
            },
            spawn_pos: Vec2 {
                x: hex_to_f64(&self.spawn_pos_x),
                y: hex_to_f64(&self.spawn_pos_y),
            },
            respawn_at_tick: self.respawn_at_tick,
            weapons: self
                .weapons
                .iter()
                .map(|w| WeaponSlot {
                    got: w.got,
                    ammo: w.ammo,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireProjectile {
    id: i32,
    #[serde(rename = "type")]
    kind: i32,
    owner: i32,
    pos_x: String,
    pos_y: String,
    dir_x: String,
    dir_y: String,
    start_tick: i64,
    life_span: i64,
    explosive: bool,
    marked_for_destroy: bool,
}

impl WireProjectile {
    fn to_state(&self) -> crate::projectile::ProjectileState2 {
        crate::projectile::ProjectileState2 {
            id: self.id,
            kind: self.kind,
            owner: self.owner,
            pos: Vec2 {
                x: hex_to_f64(&self.pos_x),
                y: hex_to_f64(&self.pos_y),
            },
            dir: Vec2 {
                x: hex_to_f64(&self.dir_x),
                y: hex_to_f64(&self.dir_y),
            },
            start_tick: self.start_tick,
            life_span: self.life_span,
            explosive: self.explosive,
            marked_for_destroy: self.marked_for_destroy,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireLaser {
    id: i32,
    owner: i32,
    #[serde(rename = "type")]
    kind: i32,
    pos_x: String,
    pos_y: String,
    dir_x: String,
    dir_y: String,
    energy: String,
    bounces: i32,
    eval_tick: i64,
    zero_energy_bounce_in_last_tick: bool,
    marked_for_destroy: bool,
}

impl WireLaser {
    fn to_state(&self) -> crate::projectile::LaserState2 {
        crate::projectile::LaserState2 {
            id: self.id,
            owner: self.owner,
            kind: self.kind,
            pos: Vec2 {
                x: hex_to_f64(&self.pos_x),
                y: hex_to_f64(&self.pos_y),
            },
            dir: Vec2 {
                x: hex_to_f64(&self.dir_x),
                y: hex_to_f64(&self.dir_y),
            },
            energy: hex_to_f64(&self.energy),
            bounces: self.bounces,
            eval_tick: self.eval_tick,
            zero_energy_bounce_in_last_tick: self.zero_energy_bounce_in_last_tick,
            marked_for_destroy: self.marked_for_destroy,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireSimState {
    tick: i64,
    next_entity_id: i64,
    tees: Vec<(i32, WireTeeSnapshot)>,
    projectiles: Vec<WireProjectile>,
    lasers: Vec<WireLaser>,
}

impl WireSimState {
    fn to_sim_state(&self) -> SimState {
        SimState {
            tick: self.tick,
            next_entity_id: self.next_entity_id,
            tees: self.tees.iter().map(|(id, s)| (*id, s.to_snapshot())).collect(),
            projectiles: self.projectiles.iter().map(WireProjectile::to_state).collect(),
            lasers: self.lasers.iter().map(WireLaser::to_state).collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpisodeMeta {
    pub trace_version: u32,
    pub kind: String,
    pub map_path: String,
    pub map_sha256: String,
    pub seed: String,
    pub tees: Vec<i32>,
    pub ticks: u64,
    pub options: WireOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WireOptions {
    pub respawn_delay_ticks: i64,
    pub infinite_ammo: bool,
    pub sv_hit: bool,
    pub all_weapons: bool,
    /// Review finding F8: previously always absent from the wire format (every generator only
    /// ever recorded 4 of `SimWorldOptions`'s 5 fields), so this comparison was silently never
    /// exercised — `#[serde(default)]` keeps every already-generated fixture (this crate's golden
    /// and regression fixtures, plus any already-captured corpus files) parsing unchanged, falling
    /// back to `None` (matching the JS side's own default when `noWeakHook` isn't passed to
    /// `new SimWorld(...)` — `world.ts:215-224`).
    #[serde(default)]
    pub no_weak_hook: Option<bool>,
}

impl From<&WireOptions> for SimWorldOptions {
    fn from(o: &WireOptions) -> Self {
        SimWorldOptions {
            respawn_delay_ticks: Some(o.respawn_delay_ticks),
            infinite_ammo: Some(o.infinite_ammo),
            sv_hit: Some(o.sv_hit),
            all_weapons: Some(o.all_weapons),
            no_weak_hook: o.no_weak_hook,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireInputRecord {
    id: i32,
    #[serde(flatten)]
    input: WireInput,
}

/// `WorldEvent` -> wire shape (`tools/ts-trace/lib.mjs`'s `eventJson`). Review finding F1: `step()`
/// events are part of the parity target (the planner consumes them directly,
/// `src/plan/livePlan.ts`'s `rolled.push(sim.step())`), so they must be compared field-for-field
/// like everything else, not treated as inert diagnostics.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum WireEvent {
    HammerHit { from: i32, to: i32 },
    HammerFire { from: i32, hits: i32 },
    Explosion { pos: WireVec2, owner: i32 },
    LaserHit { from: i32, to: i32, weapon: i32 },
    Freeze { id: i32, by: i32 },
    Death { id: i32, by: i32 },
}

impl WireEvent {
    fn to_world_event(&self) -> WorldEvent {
        match self {
            WireEvent::HammerHit { from, to } => WorldEvent::HammerHit { from: *from, to: *to },
            WireEvent::HammerFire { from, hits } => WorldEvent::HammerFire {
                from: *from,
                hits: *hits,
            },
            WireEvent::Explosion { pos, owner } => WorldEvent::Explosion {
                pos: pos.to_vec2(),
                owner: *owner,
            },
            WireEvent::LaserHit { from, to, weapon } => WorldEvent::LaserHit {
                from: *from,
                to: *to,
                weapon: *weapon,
            },
            WireEvent::Freeze { id, by } => WorldEvent::Freeze { id: *id, by: *by },
            WireEvent::Death { id, by } => WorldEvent::Death { id: *id, by: *by },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EpisodeLine {
    #[allow(dead_code)]
    tick: i64,
    inputs: Vec<WireInputRecord>,
    #[serde(default)]
    events: Vec<WireEvent>,
    order: Vec<i32>,
    by_id: Vec<i32>,
    state: WireSimState,
}

/// Parses one non-metadata line of an **episode** trace into `(tick, [(id, input)], state)` —
/// exposed for debugging/diagnostic tools (e.g. reproducing one tick in isolation via
/// `SimWorld::restore_state` + `SimWorld::set_input` + `SimWorld::step`) beyond what
/// [`replay_episode`]'s own end-to-end loop needs.
pub fn parse_episode_line(line: &str) -> (i64, Vec<(i32, PlayerInput)>, SimState) {
    let parsed: EpisodeLine = serde_json::from_str(line).expect("bad episode trace line");
    let inputs = parsed
        .inputs
        .iter()
        .map(|r| (r.id, PlayerInput::from(&r.input)))
        .collect();
    (parsed.tick, inputs, parsed.state.to_sim_state())
}

/// A single field-level mismatch found while replaying a trace-ts v1 file.
#[derive(Debug, Clone)]
pub struct Mismatch {
    pub step_index: usize,
    pub tee_id: Option<i32>,
    pub field: String,
    pub ts_value: String,
    pub rust_value: String,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.tee_id {
            Some(id) => write!(
                f,
                "step {} tee {}: field `{}` differs: ts={} rust={}",
                self.step_index, id, self.field, self.ts_value, self.rust_value
            ),
            None => write!(
                f,
                "step {}: field `{}` differs: ts={} rust={}",
                self.step_index, self.field, self.ts_value, self.rust_value
            ),
        }
    }
}

/// Compares two `step()`/op result event lists element-by-element (review finding F1 — events
/// are part of the parity target, not a diagnostics-only side channel).
fn diff_events(step_index: usize, ts: &[WorldEvent], rust: &[WorldEvent]) -> Option<Mismatch> {
    if ts.len() != rust.len() {
        return Some(Mismatch {
            step_index,
            tee_id: None,
            field: "events.len".to_string(),
            ts_value: format!("{ts:?}"),
            rust_value: format!("{rust:?}"),
        });
    }
    for (i, (a, b)) in ts.iter().zip(rust.iter()).enumerate() {
        if a != b {
            return Some(Mismatch {
                step_index,
                tee_id: None,
                field: format!("events[{i}]"),
                ts_value: format!("{a:?}"),
                rust_value: format!("{b:?}"),
            });
        }
    }
    None
}

/// Compares `order`/`byId` (as id lists — positional, duplicates included) — review finding F1.
fn diff_id_list(step_index: usize, field: &str, ts: &[i32], rust: &[i32]) -> Option<Mismatch> {
    if ts != rust {
        return Some(Mismatch {
            step_index,
            tee_id: None,
            field: field.to_string(),
            ts_value: format!("{ts:?}"),
            rust_value: format!("{rust:?}"),
        });
    }
    None
}

/// Compares two [`SimState`]s field-by-field (bit-for-bit on every `f64`), returning the first
/// mismatch found, if any. `step_index` is only used to label the returned [`Mismatch`].
pub fn diff_sim_state(step_index: usize, ts: &SimState, rust: &SimState) -> Option<Mismatch> {
    macro_rules! check {
        ($tee:expr, $field:expr, $a:expr, $b:expr) => {
            if $a != $b {
                return Some(Mismatch {
                    step_index,
                    tee_id: $tee,
                    field: $field.to_string(),
                    ts_value: format!("{:?}", $a),
                    rust_value: format!("{:?}", $b),
                });
            }
        };
    }
    macro_rules! check_f64 {
        ($tee:expr, $field:expr, $a:expr, $b:expr) => {
            if $a.to_bits() != $b.to_bits() {
                return Some(Mismatch {
                    step_index,
                    tee_id: $tee,
                    field: $field.to_string(),
                    ts_value: format!("{} (0x{:016x})", $a, $a.to_bits()),
                    rust_value: format!("{} (0x{:016x})", $b, $b.to_bits()),
                });
            }
        };
    }

    check!(None, "tick", ts.tick, rust.tick);
    check!(None, "next_entity_id", ts.next_entity_id, rust.next_entity_id);
    check!(None, "tees.len", ts.tees.len(), rust.tees.len());

    let ts_map: BTreeMap<i32, &TeeStateSnapshot> = ts.tees.iter().map(|(id, s)| (*id, s)).collect();
    let rust_map: BTreeMap<i32, &TeeStateSnapshot> = rust.tees.iter().map(|(id, s)| (*id, s)).collect();
    check!(
        None,
        "tee_ids",
        ts_map.keys().collect::<Vec<_>>(),
        rust_map.keys().collect::<Vec<_>>()
    );

    for (&id, a) in &ts_map {
        let b = rust_map[&id];
        let tee = Some(id);
        check_f64!(tee, "core.pos.x", a.core.pos.x, b.core.pos.x);
        check_f64!(tee, "core.pos.y", a.core.pos.y, b.core.pos.y);
        check_f64!(tee, "core.vel.x", a.core.vel.x, b.core.vel.x);
        check_f64!(tee, "core.vel.y", a.core.vel.y, b.core.vel.y);
        check_f64!(tee, "core.hook_pos.x", a.core.hook_pos.x, b.core.hook_pos.x);
        check_f64!(tee, "core.hook_pos.y", a.core.hook_pos.y, b.core.hook_pos.y);
        check_f64!(tee, "core.hook_dir.x", a.core.hook_dir.x, b.core.hook_dir.x);
        check_f64!(tee, "core.hook_dir.y", a.core.hook_dir.y, b.core.hook_dir.y);
        check_f64!(
            tee,
            "core.hook_tele_base.x",
            a.core.hook_tele_base.x,
            b.core.hook_tele_base.x
        );
        check_f64!(
            tee,
            "core.hook_tele_base.y",
            a.core.hook_tele_base.y,
            b.core.hook_tele_base.y
        );
        check!(tee, "core.hook_tick", a.core.hook_tick, b.core.hook_tick);
        check!(tee, "core.hook_state", a.core.hook_state, b.core.hook_state);
        check!(tee, "core.hooked_player", a.core.hooked_player, b.core.hooked_player);
        check!(
            tee,
            "core.attached_players",
            a.core.attached_players,
            b.core.attached_players
        );
        check!(tee, "core.active_weapon", a.core.active_weapon, b.core.active_weapon);
        check!(tee, "core.new_hook", a.core.new_hook, b.core.new_hook);
        check!(tee, "core.jumped", a.core.jumped, b.core.jumped);
        check!(tee, "core.jumped_total", a.core.jumped_total, b.core.jumped_total);
        check!(tee, "core.jumps", a.core.jumps, b.core.jumps);
        check!(tee, "core.direction", a.core.direction, b.core.direction);
        check!(tee, "core.angle", a.core.angle, b.core.angle);
        check!(
            tee,
            "core.triggered_events",
            a.core.triggered_events,
            b.core.triggered_events
        );
        check!(tee, "core.colliding", a.core.colliding, b.core.colliding);
        check!(tee, "core.left_wall", a.core.left_wall, b.core.left_wall);
        check!(tee, "core.freeze_start", a.core.freeze_start, b.core.freeze_start);
        check!(tee, "core.freeze_end", a.core.freeze_end, b.core.freeze_end);
        check!(tee, "core.is_in_freeze", a.core.is_in_freeze, b.core.is_in_freeze);
        check!(
            tee,
            "core.move_restrictions",
            a.core.move_restrictions,
            b.core.move_restrictions
        );

        check!(tee, "alive", a.alive, b.alive);
        check!(tee, "freeze_ticks_left", a.freeze_ticks_left, b.freeze_ticks_left);
        check!(tee, "frozen_last_tick", a.frozen_last_tick, b.frozen_last_tick);
        check!(tee, "deep_frozen", a.deep_frozen, b.deep_frozen);
        check!(tee, "tele_checkpoint", a.tele_checkpoint, b.tele_checkpoint);
        check!(tee, "move_restrictions", a.move_restrictions, b.move_restrictions);
        check!(tee, "reload_timer", a.reload_timer, b.reload_timer);
        check!(tee, "attack_tick", a.attack_tick, b.attack_tick);
        check!(tee, "queued_weapon", a.queued_weapon, b.queued_weapon);
        check_f64!(tee, "prev_pos.x", a.prev_pos.x, b.prev_pos.x);
        check_f64!(tee, "prev_pos.y", a.prev_pos.y, b.prev_pos.y);
        check_f64!(tee, "spawn_pos.x", a.spawn_pos.x, b.spawn_pos.x);
        check_f64!(tee, "spawn_pos.y", a.spawn_pos.y, b.spawn_pos.y);
        check!(tee, "respawn_at_tick", a.respawn_at_tick, b.respawn_at_tick);
        check!(tee, "weapons", a.weapons, b.weapons);
    }

    check!(None, "projectiles.len", ts.projectiles.len(), rust.projectiles.len());
    for (i, (a, b)) in ts.projectiles.iter().zip(rust.projectiles.iter()).enumerate() {
        let label = format!("projectiles[{i}]");
        check!(None, format!("{label}.id"), a.id, b.id);
        check!(None, format!("{label}.kind"), a.kind, b.kind);
        check!(None, format!("{label}.owner"), a.owner, b.owner);
        check_f64!(None, format!("{label}.pos.x"), a.pos.x, b.pos.x);
        check_f64!(None, format!("{label}.pos.y"), a.pos.y, b.pos.y);
        check_f64!(None, format!("{label}.dir.x"), a.dir.x, b.dir.x);
        check_f64!(None, format!("{label}.dir.y"), a.dir.y, b.dir.y);
        check!(None, format!("{label}.start_tick"), a.start_tick, b.start_tick);
        check!(None, format!("{label}.life_span"), a.life_span, b.life_span);
        check!(None, format!("{label}.explosive"), a.explosive, b.explosive);
        check!(
            None,
            format!("{label}.marked_for_destroy"),
            a.marked_for_destroy,
            b.marked_for_destroy
        );
    }

    check!(None, "lasers.len", ts.lasers.len(), rust.lasers.len());
    for (i, (a, b)) in ts.lasers.iter().zip(rust.lasers.iter()).enumerate() {
        let label = format!("lasers[{i}]");
        check!(None, format!("{label}.id"), a.id, b.id);
        check!(None, format!("{label}.owner"), a.owner, b.owner);
        check!(None, format!("{label}.kind"), a.kind, b.kind);
        check_f64!(None, format!("{label}.pos.x"), a.pos.x, b.pos.x);
        check_f64!(None, format!("{label}.pos.y"), a.pos.y, b.pos.y);
        check_f64!(None, format!("{label}.dir.x"), a.dir.x, b.dir.x);
        check_f64!(None, format!("{label}.dir.y"), a.dir.y, b.dir.y);
        check_f64!(None, format!("{label}.energy"), a.energy, b.energy);
        check!(None, format!("{label}.bounces"), a.bounces, b.bounces);
        check!(None, format!("{label}.eval_tick"), a.eval_tick, b.eval_tick);
        check!(
            None,
            format!("{label}.zero_energy_bounce_in_last_tick"),
            a.zero_energy_bounce_in_last_tick,
            b.zero_energy_bounce_in_last_tick
        );
        check!(
            None,
            format!("{label}.marked_for_destroy"),
            a.marked_for_destroy,
            b.marked_for_destroy
        );
    }

    None
}

/// The outcome of replaying one trace-ts v1 file: how many steps (ticks or ops) were replayed,
/// and the first mismatch found, if any.
pub struct ReplayReport {
    pub steps_replayed: usize,
    pub first_mismatch: Option<Mismatch>,
}

/// Replays a trace-ts v1 **episode** file's recorded per-tick inputs against a freshly
/// constructed [`crate::world::SimWorld`], reading the map from the path the trace itself
/// recorded (`meta.map_path`) — convenient for a corpus generated and replayed on the same
/// machine (`ts-diff`, the bulk corpus scripts), but not portable to a fixture committed to the
/// repository (the recorded path is this machine's absolute `~/aiddnet/data/...`, which
/// `docs/CLAUDE.md` forbids committing maps into in the first place). Golden/CI fixtures should
/// use [`replay_episode_with_map_bytes`] instead, with map bytes rebuilt on the fly from the
/// synthetic recipe the fixture names (see `tests/parity_golden.rs`).
pub fn replay_episode(jsonl: &str) -> ReplayReport {
    let meta_line = jsonl.lines().next().expect("empty trace file");
    let meta: EpisodeMeta = serde_json::from_str(meta_line).expect("bad metadata line");
    let map_bytes = std::fs::read(&meta.map_path).unwrap_or_else(|e| panic!("read map {}: {e}", meta.map_path));
    replay_episode_with_map_bytes(jsonl, &map_bytes)
}

/// Like [`replay_episode`], but takes the map's raw `.map` bytes directly instead of reading
/// `meta.map_path` from disk.
pub fn replay_episode_with_map_bytes(jsonl: &str, map_bytes: &[u8]) -> ReplayReport {
    let mut lines = jsonl.lines();
    let meta: EpisodeMeta = serde_json::from_str(lines.next().expect("empty trace file")).expect("bad metadata line");
    assert_eq!(meta.trace_version, 1);
    assert_eq!(meta.kind, "episode");

    let loaded = crate::map_load::load_map_bytes(map_bytes).expect("load map");
    let mut world = crate::world::SimWorld::new(loaded.collision, SimWorldOptions::from(&meta.options));

    let mut steps_replayed = 0;
    let mut first_line = true;
    let mut spawn_positions: BTreeMap<i32, Vec2> = BTreeMap::new();

    for raw in lines {
        let line: EpisodeLine = serde_json::from_str(raw).unwrap_or_else(|e| panic!("bad trace line: {e}\n{raw}"));

        if first_line {
            // Tees are added at whatever position tick 1's state shows as `spawn_pos` (the
            // generator's own random placement) — reading it back from the trace instead of
            // duplicating the JS placement algorithm in Rust.
            for (id, snap) in &line.state.tees {
                spawn_positions.insert(
                    *id,
                    Vec2 {
                        x: hex_to_f64(&snap.spawn_pos_x),
                        y: hex_to_f64(&snap.spawn_pos_y),
                    },
                );
            }
            for &id in &meta.tees {
                let pos = spawn_positions[&id];
                world.add_tee(id, pos);
            }
            first_line = false;
        }

        for rec in &line.inputs {
            world.set_input(rec.id, PlayerInput::from(&rec.input));
        }
        let rust_events = world.step();

        let ts_events: Vec<WorldEvent> = line.events.iter().map(WireEvent::to_world_event).collect();
        let ts_state = line.state.to_sim_state();
        let rust_state = world.save_state();
        // Review finding F1: `events`/`order`/`byId` are part of the parity target now, not just
        // `state` — checked in this order (events, then order/byId, then full state) so the
        // first mismatch found is reported, whichever of the four it's in.
        let mismatch = diff_events(steps_replayed, &ts_events, &rust_events)
            .or_else(|| diff_id_list(steps_replayed, "order", &line.order, &world.order_ids()))
            .or_else(|| diff_id_list(steps_replayed, "byId", &line.by_id, &world.by_id_ids()))
            .or_else(|| diff_sim_state(steps_replayed, &ts_state, &rust_state));
        if let Some(m) = mismatch {
            return ReplayReport {
                steps_replayed: steps_replayed + 1,
                first_mismatch: Some(m),
            };
        }
        steps_replayed += 1;
    }

    ReplayReport {
        steps_replayed,
        first_mismatch: None,
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireTeeState {
    id: i32,
    alive: bool,
    pos_x: String,
    pos_y: String,
    vel_x: String,
    vel_y: String,
    hook_state: i32,
    hook_pos_x: String,
    hook_pos_y: String,
    hook_dir_x: String,
    hook_dir_y: String,
    hooked_player: i32,
    jumped: i32,
    jumps_left: i32,
    direction: i32,
    angle: String,
    active_weapon: i32,
    frozen: bool,
    freeze_ticks_left: i64,
    attack_tick: i64,
    hook_tick: Option<i64>,
    jumped_total: Option<i32>,
    reload_ticks: Option<i64>,
    frozen_for: Option<i64>,
    deep_frozen: Option<bool>,
    jumps: Option<i32>,
    ddnet_flags: Option<i32>,
    since_attack: Option<i64>,
}

impl WireTeeState {
    fn to_tee_state(&self) -> TeeState {
        TeeState {
            id: self.id,
            alive: self.alive,
            pos: Vec2 {
                x: hex_to_f64(&self.pos_x),
                y: hex_to_f64(&self.pos_y),
            },
            vel: Vec2 {
                x: hex_to_f64(&self.vel_x),
                y: hex_to_f64(&self.vel_y),
            },
            hook_state: self.hook_state,
            hook_pos: Vec2 {
                x: hex_to_f64(&self.hook_pos_x),
                y: hex_to_f64(&self.hook_pos_y),
            },
            hook_dir: Vec2 {
                x: hex_to_f64(&self.hook_dir_x),
                y: hex_to_f64(&self.hook_dir_y),
            },
            hooked_player: self.hooked_player,
            jumped: self.jumped,
            jumps_left: self.jumps_left,
            direction: self.direction,
            angle: hex_to_f64(&self.angle),
            active_weapon: self.active_weapon,
            frozen: self.frozen,
            freeze_ticks_left: self.freeze_ticks_left,
            attack_tick: self.attack_tick,
            hook_tick: self.hook_tick,
            jumped_total: self.jumped_total,
            reload_ticks: self.reload_ticks,
            frozen_for: self.frozen_for,
            deep_frozen: self.deep_frozen,
            jumps: self.jumps,
            ddnet_flags: self.ddnet_flags,
            since_attack: self.since_attack,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireVec2 {
    x: String,
    y: String,
}
impl WireVec2 {
    fn to_vec2(&self) -> Vec2 {
        Vec2 {
            x: hex_to_f64(&self.x),
            y: hex_to_f64(&self.y),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireGrenade {
    owner: i32,
    spawn_pos: WireVec2,
    dir: WireVec2,
    age_ticks: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpLine {
    op: String,
    id: Option<i32>,
    tee_state: Option<WireTeeState>,
    input: Option<WireInput>,
    force: Option<WireVec2>,
    pos: Option<WireVec2>,
    list: Option<Vec<WireGrenade>>,
    restored_index: Option<usize>,
    /// Index into `saved_states` for a `"saveInto"` op (review finding F7's regression fixture —
    /// `gen-regression.mjs`'s `op_saveInto` — and, since finding F8, also the random
    /// `gen-opscript.mjs` generator's own `"saveInto"` case).
    slot: Option<usize>,
    #[serde(default)]
    events: Vec<WireEvent>,
    order: Vec<i32>,
    by_id: Vec<i32>,
    state: WireSimState,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpMeta {
    trace_version: u32,
    kind: String,
    map_path: String,
    options: WireOptions,
}

/// Replays a trace-ts v1 **opscript** file against a freshly constructed
/// [`crate::world::SimWorld`], comparing state after every op. See [`replay_episode`]'s doc
/// comment for why this reads the map from the trace's recorded (non-portable) path — golden/CI
/// fixtures should use [`replay_opscript_with_map_bytes`] instead.
pub fn replay_opscript(jsonl: &str) -> ReplayReport {
    let meta_line = jsonl.lines().next().expect("empty trace file");
    let meta: OpMeta = serde_json::from_str(meta_line).expect("bad metadata line");
    let map_bytes = std::fs::read(&meta.map_path).unwrap_or_else(|e| panic!("read map {}: {e}", meta.map_path));
    replay_opscript_with_map_bytes(jsonl, &map_bytes)
}

/// Like [`replay_opscript`], but takes the map's raw `.map` bytes directly instead of reading
/// `meta.map_path` from disk.
pub fn replay_opscript_with_map_bytes(jsonl: &str, map_bytes: &[u8]) -> ReplayReport {
    let mut lines = jsonl.lines();
    let meta: OpMeta = serde_json::from_str(lines.next().expect("empty trace file")).expect("bad metadata line");
    assert_eq!(meta.trace_version, 1);
    assert_eq!(meta.kind, "opscript");

    let loaded = crate::map_load::load_map_bytes(map_bytes).expect("load map");
    let mut world = crate::world::SimWorld::new(loaded.collision, SimWorldOptions::from(&meta.options));

    let mut saved_states: Vec<SimState> = Vec::new();
    let mut steps_replayed = 0;

    for raw in lines {
        let line: OpLine = serde_json::from_str(raw).unwrap_or_else(|e| panic!("bad trace line: {e}\n{raw}"));
        let mut rust_events: Vec<WorldEvent> = Vec::new();
        match line.op.as_str() {
            "step" => {
                rust_events = world.step();
            }
            "saveState" => {
                saved_states.push(world.save_state());
            }
            "restoreState" => {
                // Both sides append to `saved_states` in the same relative order (see
                // `trace_ts`'s module doc comment), so the trace only needs to carry *which*
                // index (of potentially several) TS restored from — `restoredIndex`, recorded by
                // `gen-opscript.mjs` — not the saved snapshot's own bytes.
                let idx = line.restored_index.expect("restoreState needs restoredIndex");
                let st = saved_states.get(idx).expect("restoredIndex out of range").clone();
                world.restore_state(&st);
            }
            "saveInto" => {
                // Review finding F7: TS's `world.saveState(into)` updates an *existing* `SimState`
                // object in place (never clearing stale entries first, `world.ts:292-298` —
                // see `SimWorld::save_state_into`'s own doc comment for the full quirk). Mirrored
                // here by mutating the same positional `saved_states` slot instead of pushing a
                // new one, matching `gen-regression.mjs`'s `op_saveInto`.
                let idx = line.slot.expect("saveInto needs slot");
                let st = saved_states.get_mut(idx).expect("slot out of range");
                world.save_state_into(st);
            }
            "applyTeeState" => {
                let id = line.id.expect("applyTeeState needs id");
                let st = line.tee_state.expect("applyTeeState needs teeState").to_tee_state();
                world.apply_tee_state(id, &st);
            }
            "setHeldInput" => {
                let id = line.id.expect("setHeldInput needs id");
                let input = line.input.expect("setHeldInput needs input");
                world.set_held_input(id, PlayerInput::from(&input));
            }
            "applyForce" => {
                let id = line.id.expect("applyForce needs id");
                let force = line.force.expect("applyForce needs force").to_vec2();
                world.apply_force(id, force);
            }
            "unfreeze" => {
                world.unfreeze(line.id.expect("unfreeze needs id"));
            }
            "kill" => {
                world.kill(line.id.expect("kill needs id"));
            }
            "addTee" => {
                let id = line.id.expect("addTee needs id");
                let pos = line.pos.expect("addTee needs pos").to_vec2();
                world.add_tee(id, pos);
            }
            "removeTee" => {
                world.remove_tee(line.id.expect("removeTee needs id"));
            }
            "setGrenades" => {
                let list = line.list.unwrap_or_default();
                let specs: Vec<GrenadeSpec> = list
                    .iter()
                    .map(|g| GrenadeSpec {
                        owner: g.owner,
                        spawn_pos: g.spawn_pos.to_vec2(),
                        dir: g.dir.to_vec2(),
                        age_ticks: g.age_ticks,
                    })
                    .collect();
                world.set_grenades(&specs);
            }
            "reset" => {
                world.reset();
            }
            "noop" => {}
            other => panic!("unknown op {other:?}"),
        }

        let ts_events: Vec<WorldEvent> = line.events.iter().map(WireEvent::to_world_event).collect();
        let ts_state = line.state.to_sim_state();
        let rust_state = world.save_state();
        // `step` is the only op that produces events (every other op's `record.events` is `[]`
        // on both sides, see `gen-opscript.mjs`) — `diff_events` still runs for every op, cheaply
        // confirming the empty case matches too.
        let mismatch = diff_events(steps_replayed, &ts_events, &rust_events)
            .or_else(|| diff_id_list(steps_replayed, "order", &line.order, &world.order_ids()))
            .or_else(|| diff_id_list(steps_replayed, "byId", &line.by_id, &world.by_id_ids()))
            .or_else(|| diff_sim_state(steps_replayed, &ts_state, &rust_state));
        if let Some(m) = mismatch {
            return ReplayReport {
                steps_replayed: steps_replayed + 1,
                first_mismatch: Some(m),
            };
        }
        steps_replayed += 1;
    }

    ReplayReport {
        steps_replayed,
        first_mismatch: None,
    }
}
