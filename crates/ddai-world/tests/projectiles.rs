//! Task 2.4b: `LiveWorld` builds projectiles from the snapshot (as the DDNet client's prediction
//! does), not from the map, and starts from DDRace tuning until the first `Sv_TuneParams`.
//!
//! - the cannon bug (8.4c review): `World::from_map` spawns the map's crazy shotguns with
//!   `start_tick == 0`; `on_snapshot` then sets the world tick to the server tick (millions after a
//!   long uptime) and the first predicted step evaluated every cannon at `t = tick / 50 s`. With
//!   the vanilla shotgun curvature the per-tick line walk is ~6e6 px long: BlmapChill (12 cannons)
//!   took 0.7 s per step, `predict(+2)` 1.1-3.3 s. `budget_*` below pins the fix;
//! - `prediction_*`: a cannon bullet from the snapshot moves, bounces and freezes a tee in the
//!   prediction exactly as the real (server-side) world does over the next 10 ticks;
//! - `tuning_*` / `observation_tuning_*`: DDRace baseline tuning and `Observation.tuning`.

use std::sync::Arc;
use std::time::Instant;

use ddai_net::generated::enums::{playerflagflag, projectileflagflag};
use ddai_net::generated::objects;
use ddai_net::tuning::{DEFAULT_TUNE_PARAMS, TuneParams};
use ddai_net::view::{CharacterView, ProjectileView};
use ddai_physics::core::PlayerInput;
use ddai_physics::map::{
    ENTITY_CRAZY_SHOTGUN, ENTITY_CRAZY_SHOTGUN_EX, ENTITY_OFFSET, MapData, ROTATION_90, ROTATION_180, ROTATION_270,
    TILE_SOLID, Tile, TuneTile,
};
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Player, Projectile, TickInput, World};
use ddai_world::LiveWorld;

/// The server tick the budget tests pretend the server has been up for (~2.3 days at 50 Tps, the
/// value the 8.4c review measured at).
const BIG_TICK: i32 = 8_400_000;

/// A `TuneParams` as a real `Sv_TuneParams` message decodes to (`received` = all 47 fields).
fn received(t: TuneParams) -> TuneParams {
    TuneParams { received: 47, ..t }
}

// --- synthetic maps ------------------------------------------------------------------------------

struct Room {
    width: i32,
    height: i32,
    game: Vec<Tile>,
    tune: Option<Vec<TuneTile>>,
    settings: Vec<String>,
}

impl Room {
    /// A closed room with a solid border.
    fn new(width: i32, height: i32) -> Self {
        let mut game = vec![Tile::default(); (width * height) as usize];
        for x in 0..width {
            game[x as usize].index = TILE_SOLID;
            game[((height - 1) * width + x) as usize].index = TILE_SOLID;
        }
        for y in 0..height {
            game[(y * width) as usize].index = TILE_SOLID;
            game[(y * width + width - 1) as usize].index = TILE_SOLID;
        }
        Room {
            width,
            height,
            game,
            tune: None,
            settings: Vec::new(),
        }
    }

    fn entity(&mut self, x: i32, y: i32, entity: u8, flags: u8) {
        self.game[(y * self.width + x) as usize] = Tile {
            index: ENTITY_OFFSET + entity,
            flags,
            skip: 0,
            reserved: 0,
        };
    }

    fn tune_zone(&mut self, x: i32, y: i32, zone: u8) {
        let n = (self.width * self.height) as usize;
        self.tune.get_or_insert_with(|| vec![TuneTile::default(); n])[(y * self.width + x) as usize] =
            TuneTile { number: zone, kind: 1 };
    }

    fn build(self) -> Arc<MapData> {
        let map = MapData {
            width: self.width as u32,
            height: self.height as u32,
            game: self.game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: self.tune,
            settings: self.settings,
        };
        map.validate().expect("consistent test map");
        Arc::new(map)
    }
}

/// 12 crazy shotguns in a 60x30 room, all four directions and both variants. The room's last
/// column and row are open (air): like BlmapChill's edge, that is what lets the *old* code's line
/// walk from the far-away tick-0 position run its whole length (a solid border ends it at once).
fn twelve_cannon_map() -> Arc<MapData> {
    let mut room = Room::new(60, 30);
    for y in 0..30 {
        room.game[(y * 60 + 59) as usize] = Tile::default();
    }
    for x in 0..60 {
        room.game[(29 * 60 + x) as usize] = Tile::default();
    }
    for i in 0..12 {
        let flags = [0, ROTATION_90, ROTATION_180, ROTATION_270][i % 4];
        let kind = if i % 2 == 0 {
            ENTITY_CRAZY_SHOTGUN
        } else {
            ENTITY_CRAZY_SHOTGUN_EX
        };
        room.entity(5 + 4 * i as i32, 4 + (i as i32 % 5) * 4, kind, flags);
    }
    room.build()
}

// --- what the server would put on the wire -------------------------------------------------------

/// `CProjectile::NetInfo` (`server/entities/projectile.cpp:307-343`) for an ownerless projectile.
fn wire_ownerless(p: &Projectile<f32>) -> ProjectileView {
    let mut flags = 0;
    if p.bouncing & 1 != 0 {
        flags |= projectileflagflag::BOUNCE_HORIZONTAL;
    }
    if p.bouncing & 2 != 0 {
        flags |= projectileflagflag::BOUNCE_VERTICAL;
    }
    if p.explosive {
        flags |= projectileflagflag::EXPLOSIVE;
    }
    if p.freeze {
        flags |= projectileflagflag::FREEZE;
    }
    ProjectileView::DDNet(objects::DDNetProjectile {
        x: (p.pos.x * 100.0).round() as i32,
        y: (p.pos.y * 100.0).round() as i32,
        vel_x: (p.direction.x * 1e6).round() as i32,
        vel_y: (p.direction.y * 1e6).round() as i32,
        type_: p.weapon_type,
        start_tick: p.start_tick,
        owner: -1,
        switch_number: p.number,
        tune_zone: p.tune_zone,
        flags,
    })
}

fn wire_projectiles(world: &World<f32>, tick_shift: i32) -> Vec<(i32, ProjectileView)> {
    world
        .projectiles
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut p = *p;
            p.start_tick += tick_shift;
            (i as i32, wire_ownerless(&p))
        })
        .collect()
}

fn resting_tee(id: i32, pos: Vec2<f32>) -> CharacterView {
    let character = objects::Character {
        tick: 0,
        x: pos.x.round() as i32,
        y: pos.y.round() as i32,
        vel_x: 0,
        vel_y: 0,
        angle: 0,
        direction: 0,
        jumped: 0,
        hooked_player: -1,
        hook_state: -1,
        hook_tick: 0,
        hook_x: pos.x.round() as i32,
        hook_y: pos.y.round() as i32,
        hook_dx: 0,
        hook_dy: 0,
        player_flags: playerflagflag::PLAYING,
        health: 10,
        armor: 0,
        ammo_count: -1,
        weapon: 0,
        emote: 0,
        attack_tick: 0,
    };
    let ddnet = objects::DDNetCharacter {
        flags: 0,
        freeze_end: 0,
        jumps: 2,
        tele_checkpoint: -1,
        strong_weak_id: id,
        jumped_total: -1,
        ninja_activation_tick: -1,
        freeze_start: -1,
        target_x: 0,
        target_y: 0,
        tune_zone_override: -1,
    };
    CharacterView {
        id,
        character,
        ddnet: Some(ddnet),
    }
}

/// What `LiveWorld` holds for a character it knows nothing about: no input, hook idle.
fn neutral_input() -> PlayerInput {
    PlayerInput {
        target_y: -1,
        player_flags: playerflagflag::PLAYING,
        ..Default::default()
    }
}

// --- budget --------------------------------------------------------------------------------------

/// Median wall time of `predict(base + 2)` over `iters` calls after `warmup` calls, in nanoseconds.
fn predict_p50_ns(live: &mut LiveWorld, warmup: usize, iters: usize) -> u64 {
    for _ in 0..warmup {
        let _ = live.predict(live.base_tick() + 2, &[]);
    }
    let mut samples: Vec<u64> = (0..iters)
        .map(|_| {
            let start = Instant::now();
            let w = live.predict(live.base_tick() + 2, &[]);
            std::hint::black_box(w.tick);
            start.elapsed().as_nanos() as u64
        })
        .collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// The D-042 budget for the whole decision is 5 ms; the spec asks `predict(+2)` for < 1 ms of work
/// (D-045: work counters plus wall; the median filters the VM's ~10 ms pauses). The bug this
/// guards against made it 1-3 s, so the bound has three orders of magnitude of slack either way.
const PREDICT_BUDGET_NS: u64 = 1_000_000;

/// The scenario as a live server presents it: a server up for `BIG_TICK` ticks whose cannons have
/// recently bounced (`start_tick` within a couple of seconds of now).
fn cannons_at_big_tick(map: &Arc<MapData>) -> (Vec<(i32, ProjectileView)>, usize) {
    let mut truth: World<f32> = World::from_map(map, 1);
    truth.init(std::iter::empty::<&str>()).expect("map init");
    let n = truth.projectiles.len();
    // Let the cannons fly for 3 s so their `start_tick`s are the ticks of real bounces, then shift
    // the whole timeline up to `BIG_TICK` (positions/directions are state, ticks are only labels).
    for _ in 0..150 {
        truth.step(&[]);
    }
    let shift = BIG_TICK - truth.tick;
    (wire_projectiles(&truth, shift), n)
}

fn budget_check(map: Arc<MapData>, label: &str, tuning: TuneParams) {
    let (items, n) = cannons_at_big_tick(&map);
    let mut live = LiveWorld::new(map, 0, 1);
    live.on_snapshot(BIG_TICK, &[], tuning, &[], None, None);
    live.set_projectiles(&items);
    assert_eq!(
        live.base_world().projectiles.len(),
        n,
        "{label}: every cannon is in the snapshot"
    );
    let p50 = predict_p50_ns(&mut live, 10, 200);
    eprintln!(
        "{label}: predict(+2) at tick {BIG_TICK}, {n} cannons: p50 = {:.1} us",
        p50 as f64 / 1000.0
    );
    assert!(
        p50 < PREDICT_BUDGET_NS,
        "{label}: predict(+2) p50 {p50} ns is over the 1 ms budget"
    );
    // ... and the hazard is really there: every cannon is in the predicted world with the snapshot's
    // (recent) time origin. (Where a bullet *is* follows from that; `prediction_*` below compares
    // it with the real world.)
    let predicted = live.predict(BIG_TICK + 2, &[]);
    assert_eq!(predicted.projectiles.len(), n);
    for p in &predicted.projectiles {
        assert!(p.pos.x.is_finite() && p.pos.y.is_finite());
        assert!(
            (BIG_TICK - 200..=BIG_TICK + 2).contains(&p.start_tick),
            "{label}: start_tick {} is not a recent tick",
            p.start_tick
        );
    }
}

#[test]
fn budget_twelve_cannons_at_a_huge_server_tick_ddrace_tuning() {
    budget_check(twelve_cannon_map(), "12 cannons, DDRace tuning", DEFAULT_TUNE_PARAMS);
}

#[test]
fn budget_twelve_cannons_at_a_huge_server_tick_vanilla_shotgun_curvature() {
    // The other half of the bug: a live tuning with non-zero shotgun curvature (here vanilla's
    // 1.25 / 2750, as an actual `Sv_TuneParams`) made the old code's tick-0 projectiles cost
    // seconds. Snapshot projectiles carry recent start ticks, so the curvature is harmless.
    budget_check(
        twelve_cannon_map(),
        "12 cannons, vanilla curvature",
        received(DEFAULT_TUNE_PARAMS),
    );
}

/// The exact failure the 8.4c review measured, without the new API: a caller that only calls
/// `on_snapshot` (as every caller did) must not pay for map-native projectiles at `start_tick 0`.
/// On the pre-2.4b code this call sequence took 1.1-3.3 s per `predict(+2)` on BlmapChill.
#[test]
fn budget_on_snapshot_alone_at_a_huge_server_tick_does_not_simulate_map_cannons_from_tick_zero() {
    let map = twelve_cannon_map();
    let mut live = LiveWorld::new(map, 0, 1);
    for k in 0..3 {
        live.on_snapshot(BIG_TICK + 2 * k, &[], DEFAULT_TUNE_PARAMS, &[], None, None);
    }
    // Few iterations: the pre-2.4b code needs seconds per call here. Timing is asserted first so
    // that is what a regression reports.
    let p50 = predict_p50_ns(&mut live, 1, 5);
    assert!(
        p50 < PREDICT_BUDGET_NS,
        "predict(+2) p50 {p50} ns is over the 1 ms budget"
    );
    assert!(
        live.base_world().projectiles.is_empty(),
        "no projectiles without snapshot items"
    );
}

/// Real maps, when they are on this machine (never committed): BlmapChill (12 crazy shotguns) and
/// Swarfey's clb-cyber map (1). Skipped, loudly, otherwise. Cannons are taken from what
/// `World::from_map` places and given the recent start ticks a live server reports.
#[test]
fn budget_real_maps_at_a_huge_server_tick() {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        (
            "BlmapChill",
            format!("{home}/aiddnet/data/ddnet-server/maps/BlmapChill.map"),
            12,
        ),
        (
            "clb-cyber-2026-2",
            format!("{home}/aiddnet/data/research/block/maps/swarfey/clb-cyber-2026-2.map"),
            1,
        ),
    ];
    let mut ran = 0;
    for (label, path, cannons) in candidates {
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("SKIPPED {label}: {path} not found");
            continue;
        };
        let map = Arc::new(ddai_map::load_map(&bytes).expect("map loads").data);
        let mut native: World<f32> = World::from_map(&map, 1);
        native.init(std::iter::empty::<&str>()).expect("map init");
        assert_eq!(native.projectiles.len(), cannons, "{label}: crazy-shotgun count");
        let items = wire_projectiles(&native, BIG_TICK - 7);
        for tuning in [DEFAULT_TUNE_PARAMS, received(DEFAULT_TUNE_PARAMS)] {
            let mut live = LiveWorld::new(Arc::clone(&map), 0, 1);
            live.on_snapshot(BIG_TICK, &[], tuning, &[], None, None);
            live.set_projectiles(&items);
            let p50 = predict_p50_ns(&mut live, 10, 200);
            eprintln!(
                "{label}: predict(+2) at tick {BIG_TICK} (tuning received={}): p50 = {:.1} us",
                tuning.received,
                p50 as f64 / 1000.0
            );
            assert!(p50 < PREDICT_BUDGET_NS, "{label}: p50 {p50} ns");
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("SKIPPED: no real map available");
    }
}

/// 2.4's zero-allocation guarantee must survive live projectiles: restoring 12 cannons and stepping
/// them (bounces, explosions, freezes) allocates nothing once the buffers have grown.
#[test]
fn predict_with_projectiles_is_zero_allocation_once_warmed_up() {
    let map = twelve_cannon_map();
    let (items, n) = cannons_at_big_tick(&map);
    let mut live = LiveWorld::new(map, 0, 1);
    live.on_snapshot(
        BIG_TICK,
        &[resting_tee(0, Vec2::new(200.0, 300.0))],
        DEFAULT_TUNE_PARAMS,
        &[],
        None,
        None,
    );
    live.set_projectiles(&items);
    assert_eq!(live.base_world().projectiles.len(), n);
    for _ in 0..20 {
        let _ = live.predict(BIG_TICK + 40, &[]);
    }
    let stats = allocation_counter::measure(|| {
        let w = live.predict(BIG_TICK + 40, &[]);
        std::hint::black_box(w.tick);
    });
    assert_eq!((stats.count_total, stats.bytes_total), (0, 0));
}

// --- prediction ----------------------------------------------------------------------------------

/// A room with a cannon shooting right along the floor at a tee, and one shooting straight down.
fn shooting_gallery() -> (Arc<MapData>, Vec2<f32>) {
    let mut room = Room::new(40, 14);
    // Floor at row 12; the tee stands on it at the right end, the cannon fires from the left.
    for x in 0..40 {
        room.game[(12 * 40 + x) as usize].index = TILE_SOLID;
    }
    room.entity(3, 11, ENTITY_CRAZY_SHOTGUN, ROTATION_90); // dir (1, 0), horizontal bounce
    room.entity(20, 2, ENTITY_CRAZY_SHOTGUN, 0); // dir (0, 1), vertical bounce
    (room.build(), Vec2::new(35.0 * 32.0, 12.0 * 32.0 - 14.0))
}

/// Steps `truth` from tick 0 until a 10-tick window starting there shows `event` first at step
/// `k` in `first_k`, and returns that window's start (the snapshot tick).
fn find_window(
    truth: &mut World<f32>,
    first_k: std::ops::RangeInclusive<usize>,
    event: impl Fn(&World<f32>, &World<f32>) -> bool,
) -> Option<World<f32>> {
    for _ in 0..600 {
        let start = truth.clone();
        let mut probe = truth.clone();
        for k in 1..=10usize {
            probe.step(&[TickInput {
                id: 0,
                input: neutral_input(),
                kill: false,
            }]);
            if event(&start, &probe) {
                if first_k.contains(&k) {
                    return Some(start);
                }
                break;
            }
        }
        truth.step(&[TickInput {
            id: 0,
            input: neutral_input(),
            kill: false,
        }]);
    }
    None
}

fn truth_world(map: &Arc<MapData>, tee: Vec2<f32>) -> World<f32> {
    let mut truth: World<f32> = World::from_map(map, 1);
    truth.init(std::iter::empty::<&str>()).expect("map init");
    truth.players[0] = Some(Player::new(0));
    world::spawn_character(&mut truth, 0, tee);
    truth
}

/// Predicts 10 ticks from a snapshot taken from `start` and compares with stepping `start`.
fn assert_prediction_matches_truth(map: &Arc<MapData>, start: &World<f32>, what: &str) {
    let tick = start.tick;
    let mut live = LiveWorld::new(Arc::clone(map), 0, 1);
    let tee = start.cores.get(0).expect("tee").pos;
    live.on_snapshot(tick, &[resting_tee(0, tee)], DEFAULT_TUNE_PARAMS, &[], None, None);
    live.set_projectiles(&wire_projectiles(start, 0));
    assert_eq!(live.base_world().projectiles.len(), start.projectiles.len(), "{what}");

    let mut truth = start.clone();
    for k in 1..=10 {
        truth.step(&[TickInput {
            id: 0,
            input: neutral_input(),
            kill: false,
        }]);
        let predicted = live.predict(tick + k, &[]);
        assert_eq!(predicted.projectiles.len(), truth.projectiles.len(), "{what}: k={k}");
        for (i, (p, t)) in predicted.projectiles.iter().zip(&truth.projectiles).enumerate() {
            // Positions arrive quantised to 0.01 px; a bounce must land on the same tick.
            assert!(
                (p.pos.x - t.pos.x).abs() < 0.05 && (p.pos.y - t.pos.y).abs() < 0.05,
                "{what}: k={k} projectile {i}: predicted {:?} vs true {:?}",
                p.pos,
                t.pos
            );
            assert_eq!(
                p.start_tick, t.start_tick,
                "{what}: k={k} projectile {i} bounced on another tick"
            );
            // `m_Direction * 1e6` on the wire: the server's `sin(pi)` residue (-4e-8) rounds to 0.
            assert!(
                (p.direction.x - t.direction.x).abs() < 1e-6 && (p.direction.y - t.direction.y).abs() < 1e-6,
                "{what}: k={k} projectile {i}: direction {:?} vs {:?}",
                p.direction,
                t.direction
            );
        }
        assert_eq!(
            predicted.characters[0].unwrap().freeze_time,
            truth.characters[0].unwrap().freeze_time,
            "{what}: k={k} freeze_time"
        );
    }
}

#[test]
fn prediction_cannon_bullet_bounces_off_a_wall_on_the_same_tick_as_the_real_world() {
    let (map, tee) = shooting_gallery();
    let mut truth = truth_world(&map, tee);
    let start = find_window(&mut truth, 4..=7, |a, b| {
        a.projectiles
            .iter()
            .zip(&b.projectiles)
            .any(|(x, y)| x.start_tick != y.start_tick)
    })
    .expect("a bounce inside a 10-tick window");
    assert!(
        start.tick > 10,
        "the cannons have been flying for a while ({})",
        start.tick
    );
    assert_prediction_matches_truth(&map, &start, "bounce");
}

#[test]
fn prediction_cannon_bullet_freezes_a_tee_in_its_path_on_the_same_tick_as_the_real_world() {
    let (map, tee) = shooting_gallery();
    let mut truth = truth_world(&map, tee);
    let start = find_window(&mut truth, 4..=7, |a, b| {
        a.characters[0].unwrap().freeze_time == 0 && b.characters[0].unwrap().freeze_time > 0
    })
    .expect("the bullet reaches the tee inside a 10-tick window");
    assert_eq!(
        start.characters[0].unwrap().freeze_time,
        0,
        "not frozen at the snapshot"
    );
    assert_prediction_matches_truth(&map, &start, "freeze");
}

#[test]
fn prediction_evaluates_a_bullet_from_its_snapshot_start_tick_at_a_huge_server_tick() {
    // A bullet the snapshot says started 3 ticks ago, 8 tiles left of a tee standing on the floor:
    // it reaches the tee within a few predicted ticks and freezes it, on exactly the tick the
    // real world (same bullet, same tee, ticked from `BIG_TICK`) does. Evaluated from tick 0 (the
    // old behaviour) it would be ~8e7 px away and the tee would never freeze.
    let (map, tee) = shooting_gallery();
    let tee = Vec2::new(180.0, tee.y);
    let bullet = Projectile {
        weapon_type: ddai_physics::core::WEAPON_SHOTGUN,
        owner: -1,
        pos: Vec2::new(100.0, tee.y),
        direction: Vec2::new(1.0, 0.0),
        init_dir: Vec2::new(1.0, 0.0),
        life_span: -2,
        start_tick: BIG_TICK - 3,
        freeze: true,
        explosive: false,
        bouncing: 1,
        tune_zone: 0,
        layer: world::Layer::Game,
        number: 0,
        marked_for_destroy: false,
    };
    let mut truth = truth_world(&map, tee);
    truth.tick = BIG_TICK;
    truth.projectiles.clear();
    truth.projectiles.push(bullet);
    // Settle the tee (it spawns in mid-air) so the snapshot describes a resting tee.
    let mut settle = truth.clone();
    settle.projectiles.clear();
    for _ in 0..30 {
        settle.step(&[TickInput {
            id: 0,
            input: neutral_input(),
            kill: false,
        }]);
    }
    let rest = settle.cores.get(0).unwrap().pos;
    truth.cores.get_mut(0).unwrap().pos = rest;
    truth.characters[0].as_mut().unwrap().prev_pos = rest;
    let mut live = LiveWorld::new(Arc::clone(&map), 0, 1);
    live.on_snapshot(BIG_TICK, &[resting_tee(0, rest)], DEFAULT_TUNE_PARAMS, &[], None, None);
    live.set_projectiles(&[(7, wire_ownerless(&bullet))]);

    let mut frozen_at = None;
    for k in 1..=8 {
        truth.step(&[TickInput {
            id: 0,
            input: neutral_input(),
            kill: false,
        }]);
        let predicted = live.predict(BIG_TICK + k, &[]);
        let (got, want) = (
            predicted.characters[0].unwrap().freeze_time,
            truth.characters[0].unwrap().freeze_time,
        );
        assert_eq!(got, want, "k={k}");
        if want > 0 && frozen_at.is_none() {
            frozen_at = Some(k);
        }
    }
    assert!(
        frozen_at.is_some_and(|k| k >= 2),
        "the bullet froze the tee mid-window: {frozen_at:?}"
    );
}

#[test]
fn a_new_snapshot_replaces_projectiles_and_on_snapshot_alone_leaves_none() {
    let map = twelve_cannon_map();
    let mut truth: World<f32> = World::from_map(&map, 1);
    truth.init(std::iter::empty::<&str>()).unwrap();
    let mut live = LiveWorld::new(map, 0, 1);
    live.on_snapshot(1000, &[], DEFAULT_TUNE_PARAMS, &[], None, None);
    live.set_projectiles(&wire_projectiles(&truth, 1000));
    assert_eq!(live.base_world().projectiles.len(), 12);
    // A snapshot listing 3 of them (the rest network-clipped) leaves exactly those 3, added in
    // ascending snapshot id like the client's `SnapCollectEntities` sorts them.
    let at = |x: i32, id: i32| {
        let (_, view) = wire_projectiles(&truth, 1002)[0];
        let ProjectileView::DDNet(mut p) = view else {
            unreachable!()
        };
        p.x = x * 100;
        (id, ProjectileView::DDNet(p))
    };
    live.on_snapshot(1002, &[], DEFAULT_TUNE_PARAMS, &[], None, None);
    live.set_projectiles(&[at(300, 30), at(100, 10), at(200, 20)]);
    let xs: Vec<i32> = live.base_world().projectiles.iter().map(|p| p.pos.x as i32).collect();
    assert_eq!(xs, vec![100, 200, 300]);
    // A snapshot for which the caller supplies none: nothing is carried over.
    live.on_snapshot(1004, &[], DEFAULT_TUNE_PARAMS, &[], None, None);
    assert!(live.base_world().projectiles.is_empty());
    assert!(live.predict(1006, &[]).projectiles.is_empty());
}

#[test]
fn projectiles_of_another_ddrace_team_are_not_predicted() {
    let map = twelve_cannon_map();
    let mut live = LiveWorld::new(map, 0, 1);
    let mut teams = ddai_net::tuning::TeamsState {
        teams: [0; 128],
        received: 2,
    };
    teams.teams[1] = 5;
    live.on_snapshot(500, &[], DEFAULT_TUNE_PARAMS, &[], Some(&teams), None);
    let shot = |owner| {
        ProjectileView::DDNet(objects::DDNetProjectile {
            x: 5_000,
            y: 5_000,
            vel_x: 0,
            vel_y: 1,
            type_: ddai_physics::core::WEAPON_GRENADE,
            start_tick: 499,
            owner,
            switch_number: 0,
            tune_zone: 0,
            flags: projectileflagflag::NORMALIZE_VEL | projectileflagflag::EXPLOSIVE,
        })
    };
    live.set_projectiles(&[(1, shot(0)), (2, shot(1)), (3, shot(-1))]);
    let owners: Vec<i32> = live.base_world().projectiles.iter().map(|p| p.owner).collect();
    assert_eq!(owners, vec![0, -1], "own and ownerless shots stay, team 5's is dropped");
}

// --- tuning --------------------------------------------------------------------------------------

#[test]
fn tuning_defaults_to_ddrace_reset_values_until_the_first_sv_tune_params() {
    let map = twelve_cannon_map();
    let mut live = LiveWorld::new(map, 0, 1);
    // `Session::tuning()` before the first `Sv_TuneParams` is `DEFAULT_TUNE_PARAMS` (`received == 0`).
    assert_eq!(DEFAULT_TUNE_PARAMS.received, 0);
    live.on_snapshot(100, &[], DEFAULT_TUNE_PARAMS, &[], None, None);
    let z = *live.base_world().tuning.zone(0);
    // `CGameContext::OnInit`/`ResetTuning` (`gamecontext.cpp:4166-4190`).
    assert_eq!(z.shotgun_curvature::<f32>(), 0.0);
    assert_eq!(z.shotgun_speed::<f32>(), 500.0);
    assert_eq!(z.gun_curvature::<f32>(), 0.0);
    assert_eq!(z.gun_speed::<f32>(), 1400.0);
    assert_eq!(z.get_by_name("shotgun_speeddiff"), Some(0.0));
    assert_eq!(z.gravity::<f32>(), 0.5, "everything else is the vanilla default");

    // Once a message has arrived its values win, even if they equal the vanilla ones.
    live.on_snapshot(102, &[], received(DEFAULT_TUNE_PARAMS), &[], None, None);
    let z = *live.base_world().tuning.zone(0);
    assert_eq!(z.shotgun_curvature::<f32>(), 1.25);
    assert_eq!(z.shotgun_speed::<f32>(), 2750.0);
}

/// A 20x10 room, our tee resting on the floor in tune zone 3 (a 3x1 patch), zone 3 with its own
/// gravity from the map's settings.
fn tuned_room() -> (Arc<MapData>, Vec2<f32>) {
    let mut room = Room::new(20, 10);
    for x in 0..20 {
        room.tune_zone(x, 8, 3);
    }
    room.settings.push("tune_zone 3 gravity 0.25".to_string());
    (room.build(), Vec2::new(5.0 * 32.0, 9.0 * 32.0 - 14.0))
}

#[test]
fn observation_tuning_is_the_tuning_applied_at_the_own_tees_zone() {
    let (map, tee) = tuned_room();
    let mut live = LiveWorld::new(map, 0, 1);
    let mut msg = received(DEFAULT_TUNE_PARAMS);
    msg.gravity = 60; // 0.60, what a server would report for the tee's current zone
    live.on_snapshot(300, &[resting_tee(0, tee)], msg, &[], None, None);
    let predicted = live.predict(302, &[]).clone();
    let obs = live.build_observation(&predicted, None);
    // The tee is in zone 3 (map-position derived): the live message was applied to *that* zone.
    assert_eq!(predicted.characters[0].unwrap().tune_zone, 3);
    assert_eq!(obs.tuning.gravity::<f32>(), 0.6);
    assert_eq!(
        obs.tuning.shotgun_curvature::<f32>(),
        1.25,
        "the message's other fields too"
    );
    // Zone 0 is untouched by it: still the DDRace baseline, not the vanilla message values.
    assert_eq!(predicted.tuning.zone(0).gravity::<f32>(), 0.5);
    assert_eq!(predicted.tuning.zone(0).shotgun_curvature::<f32>(), 0.0);
    // ... and it is not just `TuningParams::default()`.
    assert_ne!(obs.tuning, ddai_physics::tuning::TuningParams::default());
}

#[test]
fn observation_tuning_before_any_message_is_the_zones_own_map_tuning() {
    let (map, tee) = tuned_room();
    let mut live = LiveWorld::new(map, 0, 1);
    live.on_snapshot(300, &[resting_tee(0, tee)], DEFAULT_TUNE_PARAMS, &[], None, None);
    let predicted = live.predict(302, &[]).clone();
    let obs = live.build_observation(&predicted, None);
    assert_eq!(
        obs.tuning.gravity::<f32>(),
        0.25,
        "map setting `tune_zone 3 gravity 0.25`"
    );
    assert_eq!(obs.tuning.shotgun_curvature::<f32>(), 0.0);
}
