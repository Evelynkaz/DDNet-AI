//! Task 2.4c: the freeze-edge reconstruction. A fake server (a real `ddai_physics::World<f32>` plus the server's own
//! `m_SendCore` / `m_ReckoningTick` bookkeeping, as in `synthetic_replay.rs`) runs a tee over a field of
//! alternating freeze and unfreeze tiles; `LiveWorld` rebuilds the tee from every other-tick snapshot and
//! predicts the next one with the tee's true inputs. At freeze / unfreeze tile edges the order in which the
//! anti-skip walk (`m_PrevPos` -> `m_Pos`) handles the tiles decides whether the tee is frozen for the next
//! tick, so `m_PrevPos` has to be reconstructed, not snapped to the current position. The legacy
//! reconstruction (`set_snap_prev_pos(true)`) is the negative control: it must diverge where the new one is exact.
//!
//! Task 2.4d adds a second field with refill-jumps tiles (game and front layer): `m_LastRefillJumps` is rebuilt from
//! the anti-skip walk, and the pre-2.4d reconstruction (`set_skip_refill_jumps(true)`, flag always `false`) is the
//! negative control there.

use std::sync::Arc;

use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::DEFAULT_TUNE_PARAMS;
use ddai_net::view::CharacterView;
use ddai_physics::core::{self, CharacterCore, NetCharacterCore, PlayerInput, TeamsCore, WorldCore};
use ddai_physics::map::{
    MapData, TILE_FREEZE, TILE_REFILL_JUMPS, TILE_SOLID, TILE_TELEIN, TILE_TELEOUT, TILE_UNFREEZE, TeleTile, Tile,
};
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Player, TickInput, World};
use ddai_world::{LiveWorld, SnapshotInput};

/// How stale `m_ReckoningTick` may get before the server force-resyncs anyway
/// (`character.cpp:964`: `Server()->TickSpeed() * 3`, i.e. 3 seconds at the standard 50-tick
/// server).
const FORCE_RESYNC_AFTER_TICKS: i32 = 150;

/// `sv_high_bandwidth = 0`'s snapshot cadence (`server.cpp:1024`, `docs/research/ddnet-physics.md`
/// §3.D): a snapshot is sent every 2 ticks.
#[allow(dead_code)]
const SNAPSHOT_EVERY_TICKS: i32 = 2;

/// Per-character shadow of the server's own dead-reckoning bookkeeping
/// (`CCharacter::TickDeferred`, `character.cpp:857-925` for the reckoning-core half specifically,
/// `953-970` for the resync decision) — this is test-harness code that plays the *server's* role,
/// entirely separate from (and not shared with) `ddai_world::reckoning`, which plays the
/// *client's* role. Any bug in one is very unlikely to be masked by a matching bug in the other,
/// since they are ported from different C++ functions.
struct ServerReckoning {
    reckoning_core: CharacterCore<f32>,
    reckoning_tick: i32,
    send_core: CharacterCore<f32>,
}

impl ServerReckoning {
    fn new(initial: CharacterCore<f32>) -> Self {
        ServerReckoning {
            reckoning_core: initial,
            reckoning_tick: 0,
            send_core: initial,
        }
    }

    /// One server tick's worth of `TickDeferred`'s reckoning half, run *after* the real world has
    /// already ticked (`world_tick`): advances the idealized shadow core one step, then resyncs
    /// (`m_SendCore = m_Core`) exactly when the server would (a mismatch against that idealized
    /// continuation, first-ever tick, or the 3-second staleness cap).
    fn advance(
        &mut self,
        true_core: &CharacterCore<f32>,
        collision: &ddai_physics::collision::Collision<f32>,
        tick: i32,
    ) {
        let mut temp: WorldCore<f32, 1> = WorldCore::from_characters(&[(0u8, self.reckoning_core)]);
        if let Some(c) = temp.get_mut(0) {
            c.id = -1;
        }
        let teams = TeamsCore::new();
        core::tick(&mut temp, 0, collision, &teams, false, true);
        core::move_character(&mut temp, 0, collision, &teams);
        if let Some(c) = temp.get_mut(0) {
            core::quantize(c);
        }
        self.reckoning_core = *temp.get(0).unwrap();

        let predicted = self.reckoning_core.write();
        let current = true_core.write();
        if self.reckoning_tick == 0 || predicted != current || self.reckoning_tick + FORCE_RESYNC_AFTER_TICKS < tick {
            self.reckoning_tick = tick;
            self.send_core = *true_core;
            self.reckoning_core = *true_core;
        }
    }

    /// What `SnapCharacter` (`character.cpp:1094-1103`) would put on the wire right now.
    fn wire_core_and_tick(&self) -> (NetCharacterCore, i32) {
        (self.send_core.write(), self.reckoning_tick)
    }
}

/// Builds `flags` the same way `CCharacter::Snap`/`SnapCharacter` would (`character.cpp` DDNet
/// extension snap, `CHARACTERFLAG_*` bits) from the *true* core — this is what makes the
/// synthetic capture faithful: an earlier revision of this harness hardcoded `flags: 0`/
/// `weapon: 0`, silently telling `LiveWorld` every character had no weapons at all regardless of
/// the truth, which happened to leave `active_weapon`/`weapons[].got` wrong in the *reconstructed*
/// world without affecting position on the very next tick — but `handle_weapons`/weapon-switch
/// bookkeeping reads them every tick, and by a handful of ticks later that had already visibly
/// diverged the character's `y` position (found by this test itself, before this fix — see this
/// crate's `BUILD REPORT`).
fn ddnet_character_flags(core: &CharacterCore<f32>) -> i32 {
    use ddai_physics::core::*;
    let mut flags = 0;
    let mut set = |bit: i32, cond: bool| {
        if cond {
            flags |= bit;
        }
    };
    set(CHARACTERFLAG_SOLO, core.solo);
    set(CHARACTERFLAG_JETPACK, core.jetpack);
    set(CHARACTERFLAG_COLLISION_DISABLED, core.collision_disabled);
    set(CHARACTERFLAG_ENDLESS_HOOK, core.endless_hook);
    set(CHARACTERFLAG_ENDLESS_JUMP, core.endless_jump);
    set(CHARACTERFLAG_SUPER, core.is_super);
    set(CHARACTERFLAG_HAMMER_HIT_DISABLED, core.hammer_hit_disabled);
    set(CHARACTERFLAG_SHOTGUN_HIT_DISABLED, core.shotgun_hit_disabled);
    set(CHARACTERFLAG_GRENADE_HIT_DISABLED, core.grenade_hit_disabled);
    set(CHARACTERFLAG_LASER_HIT_DISABLED, core.laser_hit_disabled);
    set(CHARACTERFLAG_HOOK_HIT_DISABLED, core.hook_hit_disabled);
    set(CHARACTERFLAG_TELEGUN_GUN, core.has_telegun_gun);
    set(CHARACTERFLAG_TELEGUN_GRENADE, core.has_telegun_grenade);
    set(CHARACTERFLAG_TELEGUN_LASER, core.has_telegun_laser);
    set(CHARACTERFLAG_WEAPON_HAMMER, core.weapons[WEAPON_HAMMER as usize].got);
    set(CHARACTERFLAG_WEAPON_GUN, core.weapons[WEAPON_GUN as usize].got);
    set(CHARACTERFLAG_WEAPON_SHOTGUN, core.weapons[WEAPON_SHOTGUN as usize].got);
    set(CHARACTERFLAG_WEAPON_GRENADE, core.weapons[WEAPON_GRENADE as usize].got);
    set(CHARACTERFLAG_WEAPON_LASER, core.weapons[WEAPON_LASER as usize].got);
    set(CHARACTERFLAG_WEAPON_NINJA, core.weapons[WEAPON_NINJA as usize].got);
    set(CHARACTERFLAG_MOVEMENTS_DISABLED, core.live_frozen);
    set(CHARACTERFLAG_IN_FREEZE, core.is_in_freeze);
    set(CHARACTERFLAG_INVINCIBLE, core.invincible);
    flags
}

fn to_character_view(
    id: i32,
    net_core: NetCharacterCore,
    tick_field: i32,
    true_core: &CharacterCore<f32>,
    character: &world::Character<f32>,
    tick: i32,
) -> CharacterView {
    let character_obj = objects::Character {
        tick: tick_field,
        x: net_core.x,
        y: net_core.y,
        vel_x: net_core.vel_x,
        vel_y: net_core.vel_y,
        angle: net_core.angle,
        direction: net_core.direction,
        jumped: net_core.jumped,
        hooked_player: net_core.hooked_player,
        hook_state: net_core.hook_state,
        hook_tick: net_core.hook_tick,
        hook_x: net_core.hook_x,
        hook_y: net_core.hook_y,
        hook_dx: net_core.hook_dx,
        hook_dy: net_core.hook_dy,
        player_flags: playerflagflag::PLAYING,
        health: 10,
        armor: 0,
        ammo_count: -1,
        weapon: true_core.active_weapon,
        emote: 0,
        attack_tick: 0,
    };
    let freeze_end = if true_core.deep_frozen {
        -1
    } else if character.freeze_time > 0 {
        tick + character.freeze_time
    } else {
        0
    };
    let ddnet = objects::DDNetCharacter {
        flags: ddnet_character_flags(true_core),
        freeze_end,
        jumps: true_core.jumps,
        tele_checkpoint: character.tele_checkpoint,
        strong_weak_id: character.strong_weak_id,
        jumped_total: true_core.jumped_total,
        ninja_activation_tick: -1,
        freeze_start: -1,
        target_x: 0,
        target_y: 0,
        tune_zone_override: -1,
    };
    CharacterView {
        id,
        character: character_obj,
        ddnet: Some(ddnet),
    }
}

/// A small deterministic pseudo-random input generator (not `ddai_physics::prng::Prng` — this is
/// test-harness scaffolding, not physics under test) so "others" get varied, reproducible
/// direction/jump/hook input without a third-party `rand` dependency.
struct InputGen(u64);
impl InputGen {
    fn next_u32(&mut self) -> u32 {
        // xorshift64*, deterministic, good enough for test-fixture variety.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    fn input(&mut self, tick: i32) -> PlayerInput {
        let r = self.next_u32();
        let direction = match r % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        };
        let jump = i32::from((r / 3).is_multiple_of(5));
        let hook = i32::from((r / 17).is_multiple_of(2));
        PlayerInput {
            direction,
            target_x: direction.max(1) * 100 - 50,
            target_y: if (tick / 23) % 2 == 0 { -100 } else { 100 },
            jump,
            fire: 0,
            hook,
            player_flags: playerflagflag::PLAYING,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
        }
    }
}

/// A 70 x 30 solid-walled room, the starting point of both fields.
fn room() -> (u32, u32, Vec<Tile>) {
    let (w, h) = (70u32, 30u32);
    let mut game = vec![Tile::default(); (w * h) as usize];
    for x in 0..w {
        game[x as usize].index = TILE_SOLID;
        game[((h - 1) * w + x) as usize].index = TILE_SOLID;
    }
    for y in 0..h {
        game[(y * w) as usize].index = TILE_SOLID;
        game[(y * w + w - 1) as usize].index = TILE_SOLID;
    }
    (w, h, game)
}

fn map_of(w: u32, h: u32, game: Vec<Tile>, front: Option<Vec<Tile>>) -> Arc<MapData> {
    Arc::new(MapData {
        width: w,
        height: h,
        game,
        front,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

/// A 70 x 30 room whose middle band (rows 8..=24) is a checkerboard of freeze and unfreeze columns, so a tee
/// walking, jumping and falling through it crosses freeze / unfreeze edges all the time.
fn field() -> Arc<MapData> {
    let (w, h, mut game) = room();
    for y in 8..=24u32 {
        for x in 6..64u32 {
            game[(y * w + x) as usize].index = if (x + y) % 2 == 0 { TILE_FREEZE } else { TILE_UNFREEZE };
        }
    }
    map_of(w, h, game, None)
}

/// The same room with refill-jumps tiles instead: a game-layer checkerboard in rows 22..=28 (the floor the tee
/// stands and walks on, and the air above it) and a front-layer stripe pattern in rows 14..=21, so the tee jumps
/// through, stands on and leaves refill tiles of both layers, one-tick walks that end in plain air included.
fn refill_field() -> Arc<MapData> {
    let (w, h, mut game) = room();
    let mut front = vec![Tile::default(); (w * h) as usize];
    for y in 22..=28u32 {
        for x in 6..64u32 {
            if (x + y) % 2 == 0 {
                game[(y * w + x) as usize].index = TILE_REFILL_JUMPS;
            }
        }
    }
    for y in 14..=21u32 {
        for x in 6..64u32 {
            if (x / 2 + y) % 2 == 0 {
                front[(y * w + x) as usize].index = TILE_REFILL_JUMPS;
            }
        }
    }
    map_of(w, h, game, Some(front))
}

/// [`refill_field`] plus a column of tele-ins at x = 14 (rows 14..=28) whose tele-out is the refill tile at (10, 24):
/// a teleport lands the tee on a refill tile, so a walk reconstructed up to the post-teleport `m_PrevPos` would end
/// on refill although the server's last handled tile (the tele-in) is not.
fn refill_tele_field() -> Arc<MapData> {
    let base = refill_field();
    let (w, h) = (base.width, base.height);
    let mut tele = vec![TeleTile::default(); (w * h) as usize];
    for y in 14..=28u32 {
        tele[(y * w + 14) as usize] = TeleTile {
            number: 1,
            kind: TILE_TELEIN,
        };
    }
    tele[(24 * w + 10) as usize] = TeleTile {
        number: 1,
        kind: TILE_TELEOUT,
    };
    Arc::new(MapData {
        tele: Some(tele),
        ..(*base).clone()
    })
}

#[derive(Default, Debug)]
struct Score {
    steps: u32,
    exact: u32,
    /// Steps whose true freeze state changed.
    edge_steps: u32,
    edge_exact: u32,
    prev_pos_exact: u32,
    prev_pos_total: u32,
    flt_true: u32,
    flt_hit: u32,
    flt_false_pos: u32,
    /// Snapshots where the true `m_LastRefillJumps` was set.
    refill_true: u32,
    /// Snapshots where the reconstructed flag was set and the true one was not.
    refill_false_pos: u32,
    /// Ticks on which the true tee moved more than 3 tiles (a teleport).
    teleports: u32,
    field_mismatch: std::collections::BTreeMap<&'static str, u32>,
}

/// Runs the fake server for `ticks`, snapshots every second tick, and scores the 2-tick prediction (with the true
/// inputs) against the true state, with the chosen reconstruction.
/// `snapshot_every`: the cadence of the snapshots handed to `LiveWorld` (2 is the server's own).
fn run_on(
    map: Arc<MapData>,
    seed: u64,
    snap_prev_pos: bool,
    skip_refill_jumps: bool,
    snapshot_every: i32,
    ticks: i32,
) -> Score {
    let mut world: World<f32> = World::from_map(&map, seed);
    world.init(std::iter::empty::<&str>()).unwrap();
    const OWN: i32 = 0;
    world.players[OWN as usize] = Some(Player::new(0));
    world::spawn_character(&mut world, OWN, Vec2::new(10.0 * 32.0 + 16.0, 26.0 * 32.0 + 16.0));
    let collision = Arc::clone(&world.collision);
    let mut reckoning = ServerReckoning::new(*world.cores.get(OWN as u8).unwrap());
    let mut live = LiveWorld::new(Arc::clone(&map), OWN, seed);
    live.set_snap_prev_pos(snap_prev_pos);
    live.set_skip_refill_jumps(skip_refill_jumps);
    let mut rng = InputGen(seed ^ 0xC0FFEE);
    let inputs: Vec<(i32, PlayerInput)> = (1..=ticks + 4).map(|t| (t, rng.input(t))).collect();
    let mut score = Score::default();
    for tick in 1..=ticks {
        let mut input = inputs[(tick - 1) as usize].1;
        // Steer into the band now and then so the edges are crossed often.
        input.hook = 0;
        let pos_before = world.cores.get(OWN as u8).map(|c| c.pos);
        world.step(&[TickInput {
            id: OWN as u8,
            input,
            kill: false,
        }]);
        let Some(core) = world.cores.get(OWN as u8) else {
            // Died (death tiles are not in this map, but a freeze-kill rule might): start over.
            break;
        };
        if let Some(before) = pos_before {
            score.teleports += u32::from(ddai_physics::vmath::distance(before, core.pos) > 96.0);
        }
        reckoning.advance(core, &collision, world.tick);
        if tick % snapshot_every == 0 {
            let (net_core, tick_field) = reckoning.wire_core_and_tick();
            let character = world.characters[OWN as usize].unwrap();
            let cv = to_character_view(OWN, net_core, tick_field, core, &character, tick);
            live.on_snapshot(SnapshotInput {
                own_input_at_tick: Some(input),
                ..SnapshotInput::new(tick, std::slice::from_ref(&cv), DEFAULT_TUNE_PARAMS)
            });
            // Is the reconstructed m_PrevPos the real one?
            let truth_prev = character.prev_pos;
            let got_prev = live.base_world().characters[OWN as usize].unwrap().prev_pos;
            score.prev_pos_total += 1;
            score.prev_pos_exact += u32::from(truth_prev == got_prev);
            let got = live.base_world().characters[OWN as usize].unwrap();
            macro_rules! cmpf {
                ($($f:ident),*) => {$(
                    if got.$f != character.$f {
                        *score.field_mismatch.entry(stringify!($f)).or_default() += 1;
                    }
                )*};
            }
            cmpf!(
                alive,
                move_restrictions,
                ddrace_state,
                freeze_time,
                frozen_last_tick,
                tune_zone,
                tune_zone_old,
                start_time,
                tele_checkpoint,
                tile_index,
                tile_findex,
                last_refill_jumps,
                last_penalty,
                last_bonus,
                tele_gun_teleport,
                is_blue_tele_gun_teleport,
                strong_weak_id,
                spawn_tick,
                num_inputs,
                last_weapon,
                queued_weapon,
                reload_timer,
                attack_tick,
                health,
                armor,
                num_objects_hit,
                input,
                latest_input,
                latest_prev_input,
                prev_input,
                saved_input,
                prev_pos
            );
            let got_flt = live.base_world().characters[OWN as usize].unwrap().frozen_last_tick;
            score.refill_true += u32::from(character.last_refill_jumps);
            score.refill_false_pos += u32::from(!character.last_refill_jumps && got.last_refill_jumps);
            score.flt_true += u32::from(character.frozen_last_tick);
            score.flt_hit += u32::from(character.frozen_last_tick && got_flt);
            score.flt_false_pos += u32::from(!character.frozen_last_tick && got_flt);
            let frozen_before = character.freeze_time > 0;
            // Predict the next snapshot's tick with the true inputs and compare with a world copy stepped the same.
            let mut truth = world.clone();
            let mut next_inputs = Vec::new();
            for t in tick + 1..=tick + 2 {
                let i = inputs[(t - 1) as usize].1;
                truth.step(&[TickInput {
                    id: OWN as u8,
                    input: PlayerInput { hook: 0, ..i },
                    kill: false,
                }]);
                next_inputs.push((t, PlayerInput { hook: 0, ..i }));
            }
            let predicted = live.predict(tick + 2, &next_inputs);
            if let (Some(p), Some(t)) = (predicted.cores.get(OWN as u8), truth.cores.get(OWN as u8)) {
                let same = p.write() == t.write() && p.jumped_total == t.jumped_total;
                score.steps += 1;
                score.exact += u32::from(same);
                let frozen_after = truth.characters[OWN as usize].is_some_and(|c| c.freeze_time > 0);
                if frozen_after != frozen_before {
                    score.edge_steps += 1;
                    score.edge_exact += u32::from(same);
                }
            }
        }
    }
    score
}

fn total(snap: bool, seeds: std::ops::RangeInclusive<u64>) -> Score {
    total_on(field, snap, false, 2, seeds)
}

fn total_on(
    map: fn() -> Arc<MapData>,
    snap: bool,
    skip_refill: bool,
    snapshot_every: i32,
    seeds: std::ops::RangeInclusive<u64>,
) -> Score {
    let mut total = Score::default();
    for seed in seeds {
        let s = run_on(map(), seed, snap, skip_refill, snapshot_every, 4000);
        total.steps += s.steps;
        total.exact += s.exact;
        total.edge_steps += s.edge_steps;
        total.edge_exact += s.edge_exact;
        total.prev_pos_exact += s.prev_pos_exact;
        total.prev_pos_total += s.prev_pos_total;
        total.flt_true += s.flt_true;
        total.flt_hit += s.flt_hit;
        total.flt_false_pos += s.flt_false_pos;
        total.refill_true += s.refill_true;
        total.refill_false_pos += s.refill_false_pos;
        total.teleports += s.teleports;
        for (k, v) in s.field_mismatch {
            *total.field_mismatch.entry(k).or_default() += v;
        }
    }
    total
}

#[test]
fn a_tee_crossing_freeze_edges_is_predicted_exactly_and_the_legacy_snap_is_not() {
    let new = total(false, 1..=16);
    let old = total(true, 1..=16);
    eprintln!("new: {new:?}\nold: {old:?}");
    assert!(
        new.edge_steps > 1000,
        "the field must produce many freeze-state changes: {new:?}"
    );
    // The real `m_PrevPos` makes every 2-tick prediction exact, at the freeze edges too ...
    assert_eq!(new.exact, new.steps, "{new:?}");
    assert_eq!(new.edge_exact, new.edge_steps, "{new:?}");
    // ... where the snapped one (negative control) loses the anti-skip walk and diverges.
    assert!(old.exact < old.steps, "the legacy snap must diverge somewhere: {old:?}");
    assert!(
        old.edge_exact < old.edge_steps,
        "and at freeze edges in particular: {old:?}"
    );
}

#[test]
fn m_prev_pos_is_reconstructed_exactly_unless_the_tee_hit_something() {
    let new = total(false, 1..=16);
    let old = total(true, 1..=16);
    // Free flight and walking are exact; collisions (the velocity after the move is not the one it moved with)
    // fall back to the old snap: measured 99.6% exact here, 0.4% off by a pixel or two.
    assert!(
        f64::from(new.prev_pos_exact) >= 0.99 * f64::from(new.prev_pos_total),
        "{new:?}"
    );
    assert!(
        f64::from(old.prev_pos_exact) < 0.7 * f64::from(old.prev_pos_total),
        "the snap is right only while standing still: {old:?}"
    );
}

#[test]
fn m_frozen_last_tick_is_never_invented_and_mostly_found() {
    let new = total(false, 1..=16);
    let old = total(true, 1..=16);
    assert!(new.flt_true > 500, "{new:?}");
    // No false positive (a tee thawed on the earlier tick of the gap is not reported as thawed on this one) ...
    assert_eq!(new.flt_false_pos, 0, "{new:?}");
    // ... and most real ones found; the misses froze and thawed within one snapshot gap, or thawed in the
    // earlier tick's walk.
    assert!(f64::from(new.flt_hit) >= 0.7 * f64::from(new.flt_true), "{new:?}");
    assert_eq!(old.flt_hit, 0, "the legacy reconstruction never set it");
}

/// Task 2.4c, diagnosis: every per-character field of `world::Character`, reconstructed from a snapshot, against the
/// fake server's own. The ones below are exact; the rest are listed with why they cannot matter (or cannot be
/// known) in `docs/formats.md` §25.
#[test]
fn the_physics_relevant_character_fields_are_reconstructed_exactly() {
    let t = total(false, 1..=8);
    eprintln!("{:?}", t.field_mismatch);
    for exact in [
        "alive",
        "move_restrictions",
        "ddrace_state",
        "freeze_time",
        "tune_zone",
        "tune_zone_old",
        "start_time",
        "tele_checkpoint",
        "last_refill_jumps",
        "last_penalty",
        "last_bonus",
        "tele_gun_teleport",
        "is_blue_tele_gun_teleport",
        "strong_weak_id",
        "spawn_tick",
        "last_weapon",
        "queued_weapon",
        "attack_tick",
        "num_objects_hit",
        "latest_input",
        "latest_prev_input",
        "saved_input",
    ] {
        assert_eq!(
            t.field_mismatch.get(exact),
            None,
            "{exact} must be reconstructed exactly: {:?}",
            t.field_mismatch
        );
    }
}

/// Task 2.4d: `m_LastRefillJumps`. A tee that stays on a refill tile must not get `jumped` / `jumped_total` reset on
/// every predicted tick, only on the tick it enters (`character.cpp:1772-1781`).
#[test]
fn m_last_refill_jumps_is_reconstructed_exactly_and_the_unset_flag_diverges() {
    let new = total_on(refill_field, false, false, 2, 1..=8);
    let unset = total_on(refill_field, false, true, 2, 1..=8);
    let legacy = total_on(refill_field, true, true, 2, 1..=8);
    eprintln!("new: {new:?}\nunset: {unset:?}\nlegacy: {legacy:?}");
    assert!(
        new.refill_true > 1000,
        "the tee must spend many snapshots on refill tiles: {new:?}"
    );
    // The flag itself is exact on every snapshot, and so is every 2-tick prediction (`jumped` / `jumped_total`
    // included) ...
    assert_eq!(new.field_mismatch.get("last_refill_jumps"), None, "{new:?}");
    assert_eq!(new.exact, new.steps, "{new:?}");
    // ... where the flag left `false` (negative control) is wrong whenever the tee stands on a refill tile, and the
    // predictions that jump there diverge. The legacy `m_PrevPos` snap on top is no better.
    assert!(
        unset.field_mismatch.get("last_refill_jumps").copied().unwrap_or(0) >= new.refill_true,
        "{unset:?}"
    );
    assert!(
        unset.exact < unset.steps,
        "the unset flag must diverge somewhere: {unset:?}"
    );
    assert!(legacy.exact < legacy.steps, "{legacy:?}");
}

/// The flag is rebuilt from the previous snapshot's position when that is 1 or 2 ticks old; with a longer gap the tile
/// under `m_PrevPos` stands in. That fallback is not exact (a walk that ends on an unhandled air tile), but it is
/// closer than leaving the flag `false`. The few misses at the shorter gaps come from the `m_PrevPos` itself being
/// off by a pixel on a collision tick (see `m_prev_pos_is_reconstructed_exactly_unless_the_tee_hit_something`).
#[test]
fn m_last_refill_jumps_survives_other_snapshot_cadences() {
    let wrong = |s: &Score| s.field_mismatch.get("last_refill_jumps").copied().unwrap_or(0);
    for every in [1, 3] {
        let new = total_on(refill_field, false, false, every, 1..=8);
        let unset = total_on(refill_field, false, true, every, 1..=8);
        eprintln!("every {every}: {new:?}\nunset: {unset:?}");
        assert!(new.refill_true > 1000, "{new:?}");
        assert!(wrong(&unset) >= new.refill_true, "{unset:?}");
        if every == 1 {
            // Walk known exactly: only a missing previous snapshot or a misplaced `m_PrevPos` can get the flag wrong.
            let prev_pos_misses = new.prev_pos_total - new.prev_pos_exact;
            assert!(wrong(&new) <= prev_pos_misses, "{new:?}");
        } else {
            // The fallback is rough (measured: wrong in about a quarter of the true-flag snapshots here), but better
            // than the unset flag, which is wrong in all of them.
            assert!(wrong(&new) * 2 < new.refill_true, "{new:?}");
        }
        let floor = if every == 1 { 0.999 } else { 0.99 };
        assert!(f64::from(new.exact) >= floor * f64::from(new.steps), "{new:?}");
        assert!(new.exact > unset.exact, "{new:?} vs {unset:?}");
    }
}

/// Task 2.4d review F1: a teleport during the walk. The server handles the tele-in tile last, so the flag is `false`;
/// a walk reconstructed up to the (post-teleport) `m_PrevPos` ends on the tele-out's refill tile and said `true`.
/// With the previous snapshot's own `pos` as the walk's end (1-tick gap) and the `false` fallback (2-tick gap) there
/// is no wrong `true`, and the predictions stay at least as good as with the flag left unset.
#[test]
fn m_last_refill_jumps_is_never_wrongly_set_by_a_teleport() {
    for every in [1, 2] {
        let new = total_on(refill_tele_field, false, false, every, 1..=8);
        let unset = total_on(refill_tele_field, false, true, every, 1..=8);
        eprintln!("every {every}: {new:?}\nunset: {unset:?}");
        assert!(
            new.teleports >= 20,
            "the tee must teleport onto the refill tile: {new:?}"
        );
        assert!(new.refill_true > 500, "{new:?}");
        assert_eq!(new.refill_false_pos, 0, "{new:?}");
        assert!(new.exact >= unset.exact, "{new:?} vs {unset:?}");
        if every == 1 {
            // Walk known exactly, teleports included: only a missing previous snapshot or a misplaced `m_PrevPos` can
            // get the flag wrong.
            let wrong = new.field_mismatch.get("last_refill_jumps").copied().unwrap_or(0);
            assert!(wrong <= new.prev_pos_total - new.prev_pos_exact, "{new:?}");
        }
    }
}
