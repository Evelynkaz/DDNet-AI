//! Task 1.6 stage B, acceptance criterion 3: a small *committed* golden-fixture set for the stage-B
//! mechanics, so CI (which has no multi-GB corpus) still compares the port against the real Oracle B
//! on lasers (rifle, shotgun, tele-in-weapon, telegun), ninja dashes, lights (incl. switch-gated),
//! draggers (teams/solo), turrets (`CPlasma`) and a mixed map. Mirrors `parity_oracle_b_fixtures.rs`
//! (Stage A): one FNV-1a hash per tick, but over **every character row and the whole entity list**
//! (`common/row_hash.rs`).
//!
//! Each fixture is three files in `tests/fixtures_oracle_b_stage_b/`: `NAME.rawmap` (a hand-made map,
//! `tools/ddnet-oracle/gen_stage_b.py`), `NAME.scn` (the scenario-v3 file truncated to the first
//! [`FIXTURE_TICKS`] ticks) and `NAME.json` (the hashes the real `ddai_oracle_server` produced for
//! those ticks, plus the mechanics counters the fixture is supposed to exercise — the verifier
//! asserts its own replay shows them, so a fixture cannot go vacuous).
//!
//! **Regenerating** (needs the stage-B corpus `~/aiddnet/data/traces/oracle-b/v2-stageb/`, produced by
//! `gen_stage_b.py`): `cargo test -p ddai-physics --test parity_oracle_b_stage_b_fixtures -- --ignored
//! generate_stage_b_golden_fixtures`.

use ddai_physics::core::PlayerInput;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, LaserSlot, Player, TickInput, World};
use ddai_trace::hash::Fnv1a64;
use std::path::PathBuf;

#[path = "common/oracle_b_format.rs"]
mod oracle_b_format;
use oracle_b_format::{ScenarioV3, TraceBReader, metadata_seed};
#[path = "common/row_hash.rs"]
mod row_hash;
use row_hash::{row_hash_bytes, trace_entity_hash_bytes, world_entity_hash_bytes, world_row_hash_bytes};

/// Ticks per fixture (the committed scenarios are truncated to this).
const FIXTURE_TICKS: usize = 300;

/// Corpus scenarios that become fixtures — chosen so nobody dies within [`FIXTURE_TICKS`] (a dead
/// character's reference row is a frozen copy of its last living one, which `world_row_hash_bytes`
/// deliberately does not reproduce), and each family shows its mechanic in the window:
/// tuned rifle bounces + freezes, tele-in-weapon, telegun lasers, ninja dashes with teams/solo,
/// switch-gated lights, draggers (plain and with teams), turrets with teams, the mixed map, and the
/// calm (kill-free) mixed map with five tees.
const FIXTURES: [&str; 10] = [
    "lasers_v2_s30100",
    "lasertele_v1_s31052",
    "telegun_v1_s38050",
    "ninja_v2_s32101",
    "lights_v1_s33052",
    "draggers_v0_s34002",
    "draggers_v2_s34101",
    "turrets_v2_s35101",
    "mixed_v0_s36002",
    "calm_v0_s37000",
];

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures_oracle_b_stage_b")
}

/// Same pure function as `parity_oracle_b.rs`'s (the harness's starting-loadout rule,
/// `docs/formats.md` §12.3), applied once after each character's initial spawn.
fn weapon_variety_bonus(seed: u64, id: u32) -> Option<i32> {
    let mut rng = ddai_trace::SplitMix64::new(seed.wrapping_mul(1_000_003).wrapping_add(97u64.wrapping_mul(id as u64)));
    match rng.next_u64() & 3 {
        0 => Some(ddai_physics::core::WEAPON_SHOTGUN),
        1 => Some(ddai_physics::core::WEAPON_GRENADE),
        2 => Some(ddai_physics::core::WEAPON_LASER),
        _ => None,
    }
}

/// What a replay shows, counted on *our* simulation.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct Mechanics {
    laser_ticks: u64,
    beam_ticks: u64,
    plasma_ticks: u64,
    ninja_dash_ticks: u64,
    tele_laser_ticks: u64,
    frozen_char_ticks: u64,
}

impl Mechanics {
    fn observe(&mut self, world: &World<f32>) {
        let (mut laser, mut beam, mut plasma, mut tele) = (false, false, false, false);
        for l in &world.lasers {
            match l {
                LaserSlot::Laser(l) => {
                    laser = true;
                    tele |= l.was_tele;
                }
                LaserSlot::Beam(_) => beam = true,
                LaserSlot::Plasma(_) => plasma = true,
            }
        }
        self.laser_ticks += u64::from(laser);
        self.beam_ticks += u64::from(beam);
        self.plasma_ticks += u64::from(plasma);
        self.tele_laser_ticks += u64::from(tele);
        for (id, core) in world.cores.iter() {
            let alive = world.characters[id as usize].is_some_and(|c| c.alive);
            self.ninja_dash_ticks += u64::from(alive && core.ninja.current_move_time > 0);
            self.frozen_char_ticks += u64::from(alive && world.characters[id as usize].unwrap().freeze_time > 0);
        }
    }
}

struct Replay {
    hashes: Vec<u64>,
    mechanics: Mechanics,
}

/// Replays a stage-B scenario (rawmap + scenario v3) for `ticks` ticks like `parity_oracle_b.rs`'s
/// `run_trace`, hashing every tick.
fn replay(rawmap_bytes: &[u8], scn: &ScenarioV3, seed: u64, ticks: usize) -> Replay {
    assert_eq!(
        ddai_trace::hash::sha256(rawmap_bytes),
        scn.map_sha256,
        "rawmap/scenario sha256 mismatch"
    );
    let map = ddai_trace::rawmap::read(rawmap_bytes).expect("rawmap parse failed");
    let mut world: World<f32> = World::from_map(&map, seed);
    world.init(scn.cfg_lines.iter().map(|s| s.as_str())).unwrap();
    world.apply_commands(scn.cfg_lines.iter().map(|s| s.as_str())).unwrap();
    for c in &scn.characters {
        world.players[c.id as usize] = Some(Player::new(0));
        world::spawn_character(&mut world, c.id as i32, Vec2::new(c.spawn_x as f32, c.spawn_y as f32));
        if c.team != 0 {
            world::set_force_character_team(&mut world, c.id as i32, c.team);
        }
        if let Some(bonus) = weapon_variety_bonus(seed, c.id) {
            world::give_weapon_to(&mut world, c.id as i32, bonus);
        }
    }

    let mut hashes = Vec::with_capacity(ticks);
    let mut mechanics = Mechanics::default();
    // `died_this_tick`/`respawned_this_tick`: same formula as `parity_oracle_b.rs`'s `DeathTracking`.
    let mut prev_alive = vec![true; scn.characters.len()];
    let mut rec_die_tick = vec![0i32; scn.characters.len()];
    for tick_inputs in scn.inputs.iter().take(ticks) {
        let mut inputs: Vec<TickInput> = scn
            .characters
            .iter()
            .zip(tick_inputs.iter())
            .map(|(c, i)| TickInput {
                id: c.id as u8,
                input: PlayerInput {
                    direction: i.direction,
                    target_x: i.target_x,
                    target_y: i.target_y,
                    jump: i.jump,
                    fire: i.fire,
                    hook: i.hook,
                    player_flags: i.player_flags,
                    wanted_weapon: i.wanted_weapon,
                    next_weapon: i.next_weapon,
                    prev_weapon: i.prev_weapon,
                },
                kill: i.kill != 0,
            })
            .collect();
        inputs.sort_by_key(|ti| ti.id);
        world.step(&inputs);
        mechanics.observe(&world);

        let mut h = Fnv1a64::new();
        for (slot, c) in scn.characters.iter().enumerate() {
            let alive = world.characters[c.id as usize].is_some_and(|ch| ch.alive);
            let die_tick = world.players[c.id as usize].map(|p| p.die_tick).unwrap_or(0);
            let spawn_tick = world.characters[c.id as usize].map(|ch| ch.spawn_tick).unwrap_or(-1);
            let not_first = world.tick > 1; // the harness's `Tick > 0` guard, see `parity_oracle_b.rs`
            let died = (prev_alive[slot] && !alive) || (not_first && alive && die_tick != rec_die_tick[slot]);
            let respawned = (!prev_alive[slot] && alive) || (not_first && alive && spawn_tick == world.tick);
            if alive {
                rec_die_tick[slot] = die_tick;
            }
            prev_alive[slot] = alive;
            h.update(&world_row_hash_bytes(&world, c.id as i32, died, respawned));
        }
        h.update(&world_entity_hash_bytes(&world));
        hashes.push(h.finish());
    }
    Replay { hashes, mechanics }
}

fn load_scenario(dir: &std::path::Path, name: &str) -> (Vec<u8>, ScenarioV3, u64) {
    let rawmap = std::fs::read(dir.join(format!("{name}.rawmap"))).unwrap_or_else(|e| panic!("{name}.rawmap: {e}"));
    let scn = ScenarioV3::read_bytes(&std::fs::read(dir.join(format!("{name}.scn"))).unwrap());
    let seed = scn.embedded_seed;
    (rawmap, scn, seed)
}

fn fixture_jsons() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(fixtures_dir())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixtures_dir().display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    v.sort();
    v
}

#[test]
fn every_stage_b_fixture_is_present_and_small() {
    let jsons = fixture_jsons();
    assert_eq!(jsons.len(), FIXTURES.len(), "one json per fixture");
    let total: u64 = std::fs::read_dir(fixtures_dir())
        .unwrap()
        .map(|e| std::fs::metadata(e.unwrap().path()).unwrap().len())
        .sum();
    assert!(
        total < 700 * 1024,
        "fixtures_oracle_b_stage_b/ is {total} bytes, must stay under 700 KB"
    );
}

/// The always-run CI check: every committed stage-B fixture replays to the Oracle B hashes, tick for
/// tick, and shows the mechanics it claims to.
#[test]
fn replays_every_stage_b_golden_fixture() {
    let dir = fixtures_dir();
    for path in fixture_jsons() {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = json["name"].as_str().unwrap().to_string();
        let ticks = json["ticks"].as_u64().unwrap() as usize;
        let expected: Vec<u64> = json["tick_hashes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(expected.len(), ticks, "{name}: tick_hashes/ticks length mismatch");

        let (rawmap, scn, seed) = load_scenario(&dir, &name);
        let got = replay(&rawmap, &scn, seed, ticks);
        for (i, (ours, theirs)) in got.hashes.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                ours, theirs,
                "{name}: tick {i} hash mismatch (ours={ours:#x} theirs={theirs:#x})"
            );
        }

        let claimed = &json["mechanics"];
        let m = got.mechanics;
        for (field, ours) in [
            ("laser_ticks", m.laser_ticks),
            ("beam_ticks", m.beam_ticks),
            ("plasma_ticks", m.plasma_ticks),
            ("ninja_dash_ticks", m.ninja_dash_ticks),
            ("tele_laser_ticks", m.tele_laser_ticks),
            ("frozen_char_ticks", m.frozen_char_ticks),
        ] {
            assert_eq!(
                claimed[field].as_u64().unwrap(),
                ours,
                "{name}: mechanics counter {field}"
            );
        }
    }

    // Between them the fixtures must show every stage-B mechanic at least once.
    let mut total = Mechanics::default();
    for path in fixture_jsons() {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let c = &json["mechanics"];
        total.laser_ticks += c["laser_ticks"].as_u64().unwrap();
        total.beam_ticks += c["beam_ticks"].as_u64().unwrap();
        total.plasma_ticks += c["plasma_ticks"].as_u64().unwrap();
        total.ninja_dash_ticks += c["ninja_dash_ticks"].as_u64().unwrap();
        total.tele_laser_ticks += c["tele_laser_ticks"].as_u64().unwrap();
        total.frozen_char_ticks += c["frozen_char_ticks"].as_u64().unwrap();
    }
    assert!(total.laser_ticks > 100, "{total:?}");
    assert!(total.beam_ticks > 100, "{total:?}");
    assert!(total.plasma_ticks > 100, "{total:?}");
    assert!(total.ninja_dash_ticks > 50, "{total:?}");
    assert!(total.tele_laser_ticks > 0, "{total:?}");
    assert!(total.frozen_char_ticks > 100, "{total:?}");
}

/// The `f64` instantiation (no parity claim, `docs/DECISIONS.md` D-002/D-003) must run every stage-B
/// fixture without panicking and without NaN positions/velocities — lasers, beams, plasma, lights and
/// ninja all run through the generic code, whose `to_i32_trunc`/array-index paths are the ones an
/// instantiation mistake would break.
fn run_f64(dir: &std::path::Path, name: &str) {
    let (rawmap, scn, seed) = load_scenario(dir, name);
    let map = ddai_trace::rawmap::read(&rawmap).unwrap();
    let mut world: World<f64> = World::from_map(&map, seed);
    world.init(scn.cfg_lines.iter().map(|s| s.as_str())).unwrap();
    world.apply_commands(scn.cfg_lines.iter().map(|s| s.as_str())).unwrap();
    for c in &scn.characters {
        world.players[c.id as usize] = Some(Player::new(0));
        world::spawn_character(
            &mut world,
            c.id as i32,
            Vec2::new(f64::from(c.spawn_x), f64::from(c.spawn_y)),
        );
        if c.team != 0 {
            world::set_force_character_team(&mut world, c.id as i32, c.team);
        }
    }
    for (tick, tick_inputs) in scn.inputs.iter().enumerate() {
        let mut inputs: Vec<TickInput> = scn
            .characters
            .iter()
            .zip(tick_inputs.iter())
            .map(|(c, i)| TickInput {
                id: c.id as u8,
                input: PlayerInput {
                    direction: i.direction,
                    target_x: i.target_x,
                    target_y: i.target_y,
                    jump: i.jump,
                    fire: i.fire,
                    hook: i.hook,
                    player_flags: i.player_flags,
                    wanted_weapon: i.wanted_weapon,
                    next_weapon: i.next_weapon,
                    prev_weapon: i.prev_weapon,
                },
                kill: i.kill != 0,
            })
            .collect();
        inputs.sort_by_key(|ti| ti.id);
        world.step(&inputs);
        for (id, core) in world.cores.iter() {
            assert!(
                core.pos.x.is_finite() && core.pos.y.is_finite(),
                "{name}: tee {id} position at tick {tick}"
            );
            assert!(
                !core.vel.x.is_nan() && !core.vel.y.is_nan(),
                "{name}: tee {id} velocity at tick {tick}"
            );
        }
    }
}

#[test]
fn f64_instantiation_runs_every_stage_b_fixture_without_panicking_or_nan() {
    for path in fixture_jsons() {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        run_f64(&fixtures_dir(), json["name"].as_str().unwrap());
    }
}

/// The same over the whole stage-B corpus (`#[ignore]`d: needs `~/aiddnet/data/traces/oracle-b/v2-stageb/`).
#[test]
#[ignore]
fn f64_instantiation_runs_the_stage_b_corpus_without_panicking_or_nan() {
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join("aiddnet/data/traces/oracle-b/v2-stageb");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "scn"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    names.sort();
    assert!(!names.is_empty());
    for name in &names {
        run_f64(&dir, name);
    }
    eprintln!("f64: {} scenarios ran without panic/NaN", names.len());
}

/// Truncates a scenario-v3 file to its first `ticks` ticks and points its rawmap reference at
/// `rawmap_name` (everything else — characters, cfg lines, generator id, seed — is copied).
fn truncate_scenario(bytes: &[u8], ticks: usize, rawmap_name: &str) -> Vec<u8> {
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
    let mut o = 4 + 4 + 1; // magic, version, map_ref_tag
    let path_len = u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap()) as usize;
    let header_after_path = o + 2 + path_len;
    o = header_after_path + 32 + 1 + 4; // sha256, no_weak_hook, tuning override count
    let n_chars = u32_at(o);
    o += 4 + n_chars * 16;
    let tick_count = u32_at(o);
    assert!(ticks <= tick_count, "scenario has only {tick_count} ticks");
    let inputs_start = o + 4;
    let row = n_chars * 48;
    let trailer_start = inputs_start + tick_count * row;

    let mut out = Vec::new();
    out.extend_from_slice(&bytes[..8 + 1]);
    out.extend_from_slice(&(rawmap_name.len() as u16).to_le_bytes());
    out.extend_from_slice(rawmap_name.as_bytes());
    out.extend_from_slice(&bytes[header_after_path..o]);
    out.extend_from_slice(&(ticks as u32).to_le_bytes());
    out.extend_from_slice(&bytes[inputs_start..inputs_start + ticks * row]);
    out.extend_from_slice(&bytes[trailer_start..]);
    out
}

/// Regenerates every committed stage-B fixture from the stage-B corpus. `#[ignore]`d (needs the
/// corpus, see the module doc comment).
#[test]
#[ignore]
fn generate_stage_b_golden_fixtures() {
    let corpus = PathBuf::from(std::env::var("HOME").unwrap()).join("aiddnet/data/traces/oracle-b/v2-stageb");
    assert!(corpus.is_dir(), "{} not found", corpus.display());
    std::fs::create_dir_all(fixtures_dir()).unwrap();
    for name in FIXTURES {
        let rawmap_bytes = std::fs::read(corpus.join(format!("{name}.rawmap"))).unwrap();
        let scn_bytes = std::fs::read(corpus.join(format!("{name}.scn"))).unwrap();
        let rawmap_name = format!("{name}.rawmap");
        let truncated = truncate_scenario(&scn_bytes, FIXTURE_TICKS, &rawmap_name);
        std::fs::write(fixtures_dir().join(&rawmap_name), &rawmap_bytes).unwrap();
        std::fs::write(fixtures_dir().join(format!("{name}.scn")), &truncated).unwrap();

        // Hashes of the *reference* trace for the same ticks.
        let trb_bytes = std::fs::read(corpus.join(format!("{name}.trb"))).unwrap();
        let mut trace = TraceBReader::new(&trb_bytes);
        assert_eq!(trace.version, 3, "{name}: stage-B fixtures need trace-b v3");
        let seed = metadata_seed(&trace.metadata_json);
        let mut reference_hashes = Vec::new();
        while reference_hashes.len() < FIXTURE_TICKS {
            let tick = trace.next_tick().expect("trace shorter than the fixture");
            let mut h = Fnv1a64::new();
            for row in &tick.characters {
                h.update(&row_hash_bytes(&row.core, &row.ddrace));
            }
            h.update(&trace_entity_hash_bytes(&tick));
            reference_hashes.push(h.finish());
        }

        // Our replay must already agree (the corpus parity test guarantees it; re-checked here so a
        // fixture is never written from a disagreeing pair).
        let scn = ScenarioV3::read_bytes(&truncated);
        let got = replay(&rawmap_bytes, &scn, seed, FIXTURE_TICKS);
        if let Some(i) = got.hashes.iter().zip(&reference_hashes).position(|(a, b)| a != b) {
            panic!("{name}: our replay disagrees with the reference from tick {i}");
        }

        let m = got.mechanics;
        let fixture = serde_json::json!({
            "name": name,
            "seed": seed,
            "ticks": FIXTURE_TICKS,
            "characters": scn.characters.len(),
            "tick_hashes": reference_hashes,
            "mechanics": {
                "laser_ticks": m.laser_ticks,
                "beam_ticks": m.beam_ticks,
                "plasma_ticks": m.plasma_ticks,
                "ninja_dash_ticks": m.ninja_dash_ticks,
                "tele_laser_ticks": m.tele_laser_ticks,
                "frozen_char_ticks": m.frozen_char_ticks,
            },
        });
        std::fs::write(
            fixtures_dir().join(format!("{name}.json")),
            serde_json::to_string_pretty(&fixture).unwrap(),
        )
        .unwrap();
        eprintln!("wrote fixture {name}: {m:?}");
    }
}
