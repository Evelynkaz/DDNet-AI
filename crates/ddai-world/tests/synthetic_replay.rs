//! Offline, fully deterministic accuracy/determinism proof (task spec, acceptance criterion 4:
//! "replay recorded sessions (8.4a recordings or a synthetic capture) offline"). 8.4a's recorder
//! is not merged yet (`docs/STATUS.md`), so this builds its own synthetic capture instead: a
//! small "fake server" harness that runs a real `ddai_physics::World<f32>` and, alongside it,
//! reproduces the *server's own* `m_SendCore`/`m_ReckoningTick` bookkeeping
//! (`character.cpp:953-970`) exactly — so the synthetic `CNetObj_Character`/`CNetObj_DDNetCharacter`
//! values it feeds into `LiveWorld::on_snapshot` are, bit for bit, what a real DDNet 20.1 server
//! would have put on the wire for the same tick history. This is the strongest evidence this
//! crate can produce for the accuracy claim without a live server: it isolates LiveWorld's own
//! logic from every real-network variable (packet loss, jitter, the actual C++ server binary).
//!
//! Two scenarios:
//! - [`own_tee_prediction_is_bit_exact_without_interactions`]: one character, alone (task spec:
//!   "without interactions"), predicted a few ticks ahead using its *actual* future input (known
//!   in advance here, exactly like a real bot knows its own queued input) — checks the >= 99.9%
//!   bit-exact bar.
//! - [`others_prediction_error_distribution_at_1_5_10_ticks`]: several characters actively moving,
//!   hooking each other and the map, and walking through the freeze pool — predicted using only
//!   the *held* input approximation (task spec: same as a real bot would have to use, since
//!   another player's true future input is never known), reporting the resulting error-px
//!   distribution at 1/5/10 ticks (no bit-exact bar for this case — the task spec only asks this
//!   be *measured and reported*).
//!
//! A third test ([`prediction_is_deterministic_across_two_independent_runs`]) replays the exact
//! same synthetic capture through two fresh `LiveWorld`s and checks every predicted tick matches
//! bit for bit.

use std::sync::Arc;

use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::DEFAULT_TUNE_PARAMS;
use ddai_net::view::CharacterView;
use ddai_physics::core::{self, CharacterCore, NetCharacterCore, PlayerInput, TeamsCore, WorldCore};
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Player, TickInput, World};
use ddai_world::accuracy::AccuracyTracker;
use ddai_world::{LiveWorld, SnapshotInput};

/// How stale `m_ReckoningTick` may get before the server force-resyncs anyway
/// (`character.cpp:964`: `Server()->TickSpeed() * 3`, i.e. 3 seconds at the standard 50-tick
/// server).
const FORCE_RESYNC_AFTER_TICKS: i32 = 150;

/// `sv_high_bandwidth = 0`'s snapshot cadence (`server.cpp:1024`, `docs/research/ddnet-physics.md`
/// §3.D): a snapshot is sent every 2 ticks.
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

#[test]
fn own_tee_prediction_is_bit_exact_without_interactions() {
    let map = Arc::new(ddai_trace::synthetic::build("freeze").expect("recipe must exist"));
    let mut world: World<f32> = World::from_map(&map, 42);
    world
        .init(std::iter::empty::<&str>())
        .expect("synthetic recipe map settings must be recognized");

    const OWN: i32 = 0;
    let spawn = Vec2::new(30.0 * 32.0 + 16.0, 20.0 * 32.0 + 16.0); // bottom-right, clear of every hazard.
    world.players[OWN as usize] = Some(Player::new(0));
    world::spawn_character(&mut world, OWN, spawn);

    let collision = Arc::clone(&world.collision);
    let mut reckoning = ServerReckoning::new(*world.cores.get(OWN as u8).unwrap());

    let mut live = LiveWorld::new(Arc::clone(&map), OWN, 42);
    let mut tracker = AccuracyTracker::new(OWN);
    let mut rng = InputGen(0xC0FFEE);
    // The *actual* future input, precomputed for the whole run — a real bot knows exactly what it
    // queued for each upcoming tick (`own_inputs_in_flight`); this test reuses the same generator
    // both to drive the real world and to hand LiveWorld the true answer, exactly mirroring that.
    let horizons = [1, 5, 10];
    const TOTAL_TICKS: i32 = 3000;
    let mut future_inputs: Vec<(i32, PlayerInput)> = Vec::new();
    for tick in 1..=TOTAL_TICKS + horizons[horizons.len() - 1] {
        future_inputs.push((tick, rng.input(tick)));
    }

    for tick in 1..=TOTAL_TICKS {
        let input = future_inputs[(tick - 1) as usize].1;
        world.step(&[TickInput {
            id: OWN as u8,
            input,
            kill: false,
        }]);
        reckoning.advance(world.cores.get(OWN as u8).unwrap(), &collision, world.tick);
        // Ground truth for every pending prediction is the *true* world, every tick — not just on
        // the (even-only) snapshot ticks: a prediction horizon of 1 or 5 lands on an odd tick,
        // which no snapshot ever describes directly (`SNAPSHOT_EVERY_TICKS == 2`), but this test
        // harness has perfect knowledge of the true state at every tick regardless of when a
        // snapshot happens to fall.
        tracker.record_actual(tick, &world);

        if tick % SNAPSHOT_EVERY_TICKS == 0 {
            let (net_core, tick_field) = reckoning.wire_core_and_tick();
            let character = world.characters[OWN as usize].unwrap();
            let true_core = world.cores.get(OWN as u8).unwrap();
            let cv = to_character_view(OWN, net_core, tick_field, true_core, &character, tick);
            live.on_snapshot(SnapshotInput::new(tick, std::slice::from_ref(&cv), DEFAULT_TUNE_PARAMS));

            for &h in &horizons {
                let in_flight: Vec<(i32, PlayerInput)> = future_inputs
                    .iter()
                    .filter(|&&(t, _)| t > tick && t <= tick + h)
                    .copied()
                    .collect();
                let predicted = live.predict(tick + h, &in_flight);
                if let Some(core) = predicted.cores.get(OWN as u8) {
                    tracker.record_prediction(tick, tick + h, OWN, core.pos, character.freeze_time > 0);
                }
            }
        }
    }

    let summary = tracker.summarize();
    assert!(
        !summary.is_empty(),
        "must have produced at least one horizon's worth of samples"
    );
    for h in &summary {
        eprintln!(
            "own tee, horizon {} ticks: n={} exact_fraction={:.5} mean_px={:.4} p99_px={:.4} max_px={:.4}",
            h.horizon_ticks, h.count, h.exact_fraction, h.mean_error_px, h.p99_error_px, h.max_error_px
        );
        assert!(h.is_own);
        assert!(
            h.exact_fraction >= 0.999,
            "own tee without interactions must be >= 99.9% bit-exact at horizon {} (was {})",
            h.horizon_ticks,
            h.exact_fraction
        );
    }
}

#[test]
fn others_prediction_error_distribution_at_1_5_10_ticks() {
    let map = Arc::new(ddai_trace::synthetic::build("freeze").expect("recipe must exist"));
    let mut world: World<f32> = World::from_map(&map, 7);
    world
        .init(std::iter::empty::<&str>())
        .expect("synthetic recipe map settings must be recognized");

    const OWN: i32 = 0; // the observer's own id (never actually spawned - purely a POV/id choice).
    let others: [i32; 3] = [1, 2, 3];
    let spawns = [
        Vec2::new(15.0 * 32.0 + 16.0, 7.0 * 32.0 + 16.0), // inside the freeze pool.
        Vec2::new(9.0 * 32.0 + 16.0, 9.0 * 32.0 + 16.0),  // near the nohook wall.
        Vec2::new(25.0 * 32.0 + 16.0, 13.0 * 32.0 + 16.0), // near the deep-freeze pool.
    ];
    for (&id, &pos) in others.iter().zip(spawns.iter()) {
        world.players[id as usize] = Some(Player::new(0));
        world::spawn_character(&mut world, id, pos);
    }

    let collision = Arc::clone(&world.collision);
    let mut reckoning: std::collections::HashMap<i32, ServerReckoning> = others
        .iter()
        .map(|&id| (id, ServerReckoning::new(*world.cores.get(id as u8).unwrap())))
        .collect();

    let mut live = LiveWorld::new(Arc::clone(&map), OWN, 7);
    let mut tracker = AccuracyTracker::new(OWN);
    let mut rngs: std::collections::HashMap<i32, InputGen> = others
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, InputGen(0xA5A5_0000 + i as u64)))
        .collect();
    let horizons = [1, 5, 10];
    const TOTAL_TICKS: i32 = 3000;

    for tick in 1..=TOTAL_TICKS {
        let mut inputs: Vec<TickInput> = others
            .iter()
            .map(|&id| TickInput {
                id: id as u8,
                input: rngs.get_mut(&id).unwrap().input(tick),
                kill: false,
            })
            .collect();
        inputs.sort_by_key(|ti| ti.id);
        world.step(&inputs);
        // A character that died this tick (e.g. the death pit) has no `CharacterCore` any more
        // (`world::die` removes it from `WorldCore` — see that fn's own doc comment) and, in real
        // DDNet, simply is not snapped at all until it respawns; mirror that here rather than
        // unwrapping a core that no longer exists.
        for &id in &others {
            match world.cores.get(id as u8) {
                Some(core) => reckoning
                    .entry(id)
                    .or_insert_with(|| ServerReckoning::new(*core))
                    .advance(core, &collision, world.tick),
                None => {
                    reckoning.remove(&id);
                }
            }
        }
        // See the analogous comment in `own_tee_prediction_is_bit_exact_without_interactions`:
        // ground truth is checked every tick against the true world, not only on snapshot ticks.
        tracker.record_actual(tick, &world);

        if tick % SNAPSHOT_EVERY_TICKS == 0 {
            let views: Vec<CharacterView> = others
                .iter()
                .filter(|id| reckoning.contains_key(id))
                .map(|&id| {
                    let (net_core, tick_field) = reckoning[&id].wire_core_and_tick();
                    let character = world.characters[id as usize].unwrap();
                    let true_core = world.cores.get(id as u8).unwrap();
                    to_character_view(id, net_core, tick_field, true_core, &character, tick)
                })
                .collect();
            live.on_snapshot(SnapshotInput::new(tick, &views, DEFAULT_TUNE_PARAMS));

            for &h in &horizons {
                let predicted = live.predict(tick + h, &[]);
                for &id in &others {
                    if let Some(core) = predicted.cores.get(id as u8) {
                        // `base_frozen` is irrelevant here: every id in `others` differs from
                        // `OWN`, so `AccuracyTracker::new(OWN)` never treats any of these as an
                        // own-tee sample — F13's freeze filter (`summarize`'s own doc comment)
                        // never even looks at this value for a non-own id.
                        tracker.record_prediction(tick, tick + h, id, core.pos, false);
                    }
                }
            }
        }
    }

    let summary = tracker.summarize();
    assert!(!summary.is_empty());
    for h in &summary {
        eprintln!(
            "others, horizon {} ticks: n={} exact_fraction={:.4} mean_px={:.3} p50_px={:.3} p90_px={:.3} p99_px={:.3} max_px={:.3}",
            h.horizon_ticks,
            h.count,
            h.exact_fraction,
            h.mean_error_px,
            h.p50_error_px,
            h.p90_error_px,
            h.p99_error_px,
            h.max_error_px
        );
        assert!(!h.is_own);
        assert!(h.count > 0);
        assert!(h.mean_error_px.is_finite() && h.mean_error_px >= 0.0);
    }
    // Horizon 1's error should not be *worse* on average than horizon 10's — a basic sanity check
    // on the measurement itself (further-ahead predictions should never be reliably more accurate
    // than closer ones for a held-input model whose whole error source is "we don't know what
    // they'll actually do next").
    let mean_at = |h: i32| summary.iter().find(|s| s.horizon_ticks == h).map(|s| s.mean_error_px);
    if let (Some(m1), Some(m10)) = (mean_at(1), mean_at(10)) {
        assert!(
            m1 <= m10 + 1e-6,
            "horizon-1 mean error ({m1}) should not exceed horizon-10's ({m10})"
        );
    }
}

#[test]
fn prediction_is_deterministic_across_two_independent_runs() {
    let map = Arc::new(ddai_trace::synthetic::build("arena").expect("recipe must exist"));
    let mut world: World<f32> = World::from_map(&map, 99);
    world.init(std::iter::empty::<&str>()).unwrap();

    let ids = [0i32, 1];
    let spawns = [Vec2::new(300.0, 300.0), Vec2::new(340.0, 300.0)];
    for (&id, &pos) in ids.iter().zip(spawns.iter()) {
        world.players[id as usize] = Some(Player::new(0));
        world::spawn_character(&mut world, id, pos);
    }
    let collision = Arc::clone(&world.collision);
    let mut reckoning: std::collections::HashMap<i32, ServerReckoning> = ids
        .iter()
        .map(|&id| (id, ServerReckoning::new(*world.cores.get(id as u8).unwrap())))
        .collect();

    let mut live_a = LiveWorld::new(Arc::clone(&map), 0, 99);
    let mut live_b = LiveWorld::new(Arc::clone(&map), 0, 99);
    let mut rngs: std::collections::HashMap<i32, InputGen> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, InputGen(0xD00D + i as u64)))
        .collect();

    for tick in 1..=600 {
        let mut inputs: Vec<TickInput> = ids
            .iter()
            .map(|&id| TickInput {
                id: id as u8,
                input: rngs.get_mut(&id).unwrap().input(tick),
                kill: false,
            })
            .collect();
        inputs.sort_by_key(|ti| ti.id);
        world.step(&inputs);
        for &id in &ids {
            match world.cores.get(id as u8) {
                Some(core) => reckoning
                    .entry(id)
                    .or_insert_with(|| ServerReckoning::new(*core))
                    .advance(core, &collision, world.tick),
                None => {
                    reckoning.remove(&id);
                }
            }
        }

        if tick % SNAPSHOT_EVERY_TICKS == 0 {
            let views: Vec<CharacterView> = ids
                .iter()
                .filter(|id| reckoning.contains_key(id))
                .map(|&id| {
                    let (net_core, tick_field) = reckoning[&id].wire_core_and_tick();
                    let character = world.characters[id as usize].unwrap();
                    let true_core = world.cores.get(id as u8).unwrap();
                    to_character_view(id, net_core, tick_field, true_core, &character, tick)
                })
                .collect();
            live_a.on_snapshot(SnapshotInput::new(tick, &views, DEFAULT_TUNE_PARAMS));
            live_b.on_snapshot(SnapshotInput::new(tick, &views, DEFAULT_TUNE_PARAMS));

            for h in [1, 3, 7] {
                let pa = live_a.predict(tick + h, &[]);
                let pb = live_b.predict(tick + h, &[]);
                for &id in &ids {
                    let ca = pa.cores.get(id as u8);
                    let cb = pb.cores.get(id as u8);
                    assert_eq!(
                        ca.map(|c| c.pos),
                        cb.map(|c| c.pos),
                        "tick {tick} horizon {h} id {id}: positions diverged"
                    );
                    assert_eq!(
                        ca.map(|c| c.vel),
                        cb.map(|c| c.vel),
                        "tick {tick} horizon {h} id {id}: velocities diverged"
                    );
                }
            }
        }
    }
}
