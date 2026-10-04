//! Task 1.6: parity against the real Oracle B corpus (task 1.5, `~/aiddnet/data/traces/oracle-b/v1/`,
//! plus the stage-B corpus `~/aiddnet/data/traces/oracle-b/v2-stageb/`, both outside the
//! repository).
//!
//! # Stage B: every tick is compared
//!
//! Stage A compared a trace only up to a *cut*: the first tick at which the reference showed a
//! mechanic the port did not yet model (lasers, dragger beams, turret shots, lights touching a
//! character, ninja). Stage B ports all of them, so the comparison now runs over **every tick of
//! every trace** and a mismatch anywhere is a failure. Compared per tick, all through
//! `to_bits`/exact integers:
//!
//! - every character's input-independent state (`compare_character`: the 28 core fields and the
//!   54 DDRace fields, `docs/formats.md` §11.3);
//! - `GlobalTickFields.switches` (per switch, per dumped team);
//! - every live projectile, in dump order;
//! - the whole `ENTTYPE_LASER` entity list **in list order**: lasers (kind 1), dragger beams
//!   (kind 4, their `owner` = the dragged client from trace version 3 on), turret shots
//!   (kind 7, version 3 only), then the static fixtures — the positions of doors (kind 2),
//!   draggers (3), turrets (5) and lights (6), plus a light's `m_Length`/`m_AngularSpeed`/
//!   `m_Speed` which the dump carries (the last one flips sign at run time);
//! - each character's `died_this_tick`/`respawned_this_tick`.
//!
//! # The retired Stage A cut rule (diagnostic only)
//!
//! To keep reporting "how many character-ticks were previously beyond the cut" the old rule is
//! still evaluated against the *reference* trace of every tick (never stopping anything): the
//! first tick at which any of the following holds is recorded as `cut_tick`, and every compared
//! tick from there on counts as `previously_cut`:
//!
//! 1. an `EntityRecord` with `kind == 1` (`CLaser`) is present;
//! 2. an `EntityRecord` with `kind == 4` (a dragger beam) is present;
//! 3. a `kind == 5`/`6` record's position differs from that fixture's tick-0 position;
//! 4. a character's `weapon_got_mask` ninja bit (5) becomes newly set;
//! 5. an alive character is within `sv_plasma_range` (+ its own radius `28`) of a turret with a
//!    clear line of sight (`CGun::Fire`'s necessary firing precondition, `gun.cpp:53,92`);
//! 6. an alive character's radius touches a *static* (`m_AngularSpeed == 0`) light's beam.
//!
//! The rule is identical to Stage A's (`docs/formats.md` §17); only its consequence changed.

use ddai_physics::core::{PlayerInput, WEAPON_GRENADE, WEAPON_LASER, WEAPON_SHOTGUN};
use ddai_physics::vmath::{self, Vec2};
use ddai_physics::world::{self, Fixture, LaserSlot, Player, TickInput, World};
use std::path::{Path, PathBuf};

#[path = "common/oracle_b_format.rs"]
mod oracle_b_format;
use oracle_b_format::{ScenarioV3, TraceBReader, metadata_seed};

fn corpus_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set");
    PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v1")
}

/// The stage-B corpus (task 1.6 stage B): traces regenerated or crafted with the trace-b v3 harness
/// (`tools/ddnet-oracle`, `docs/formats.md` §11/§12) — maps with turrets, plus targeted scenarios
/// for lasers/shotgun bounces with tele tiles, ninja, lights, draggers and turrets. Never overlaps
/// with `v1`.
fn stage_b_corpus_dir() -> PathBuf {
    // `DDAI_STAGE_B_DIR` redirects the test at a scratch corpus while a scenario family is being
    // developed; unset, it is the real stage-B corpus.
    if let Some(dir) = std::env::var_os("DDAI_STAGE_B_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").expect("HOME must be set");
    PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v2-stageb")
}

fn resolve_rawmap_path(scn_path: &Path, rawmap_path_in_file: &str) -> PathBuf {
    let p = PathBuf::from(rawmap_path_in_file);
    if p.is_file() {
        return p;
    }
    scn_path.with_file_name(p.file_name().unwrap())
}

/// Weapon-variety bonus (`docs/formats.md` §12.3, "Разнообразие оружия"): a pure function of
/// `seed`/`id`, applied once, right after each character's *initial* spawn only (never on
/// respawn — the harness's own character-creation loop runs exactly once).
fn weapon_variety_bonus(seed: u64, id: u32) -> Option<i32> {
    let mut rng = ddai_trace::SplitMix64::new(seed.wrapping_mul(1_000_003).wrapping_add(97u64.wrapping_mul(id as u64)));
    match rng.next_u64() & 3 {
        0 => Some(WEAPON_SHOTGUN),
        1 => Some(WEAPON_GRENADE),
        2 => Some(WEAPON_LASER),
        _ => None,
    }
}

struct FixtureSnapshot {
    kind: i32,
    pos_x: f32,
    pos_y: f32,
}

/// `sv_plasma_range`'s default (`config_variables.h:689`); see this module's doc comment's
/// condition 5 for why this crate hardcodes it rather than reading a modeled config variable.
const SV_PLASMA_RANGE_DEFAULT: f32 = 700.0;

/// `CCharacterCore::PhysicalSize()`/`CEntity::m_ProximityRadius` for a `CCharacter` (`28.0`,
/// unhalved) — see [`ddai_physics::world`]'s own `character_proximity_radius`, not `pub` from
/// this test's point of view, so restated here for conditions 5/6's own range checks.
const CHARACTER_PROXIMITY_RADIUS: f32 = 28.0;

/// Returns `Some(field_name)` the first time this tick's reference row trips the Stage A cut
/// rule (see this module's doc comment); `None` if the tick is still Stage-A-comparable.
/// `world`: the running simulation (identical map/collision geometry to the reference's, since
/// both load the same rawmap, and — as of the dragger/turret/light mover-tile fix — identical
/// fixture positions too) — used by condition 5's line-of-sight check and condition 6's own
/// light-beam geometry.
fn cut_signal(
    tick: &oracle_b_format::TraceBTick,
    tick0_fixtures: &[FixtureSnapshot],
    prev_ninja_mask: &mut std::collections::HashMap<u32, bool>,
    character_ids: &[u32],
    world: &World<f32>,
) -> Option<&'static str> {
    let collision = &world.collision;
    for e in &tick.entities {
        if e.kind == 1 {
            return Some("laser (kind=1) present");
        }
        if e.kind == 4 {
            return Some("dragger beam (kind=4) present");
        }
    }
    let mut fixture_index = 0usize;
    for e in &tick.entities {
        if e.kind == 5 || e.kind == 6 {
            if let Some(snap) = tick0_fixtures.get(fixture_index)
                && snap.kind == e.kind
                && (snap.pos_x != e.pos_x || snap.pos_y != e.pos_y)
            {
                return Some("turret/light fixture moved from its tick-0 position");
            }
            fixture_index += 1;
        }
    }
    // Condition 5 (see this module's doc comment): any alive character within
    // `sv_plasma_range` of a turret, with a clear line of sight to it.
    for snap in tick0_fixtures.iter().filter(|f| f.kind == 5) {
        let turret_pos = Vec2::new(snap.pos_x, snap.pos_y);
        for row in &tick.characters {
            if row.ddrace.alive == 0 {
                continue;
            }
            let char_pos = Vec2::new(row.core.pos_x, row.core.pos_y);
            // `FindEntities(m_Pos, SvPlasmaRange, ...)` checks `distance < Radius +
            // pEnt->m_ProximityRadius` (`gameworld.cpp:66`), not `distance <= Radius` — the
            // target character's own 28-unit proximity radius is added on top (review round 1,
            // finding F11.1).
            if vmath::distance(turret_pos, char_pos) < SV_PLASMA_RANGE_DEFAULT + CHARACTER_PROXIMITY_RADIUS
                && collision.intersect_line(turret_pos, char_pos).hit == 0
            {
                return Some("character within sv_plasma_range of a turret (kind=5), line of sight clear");
            }
        }
    }
    // Condition 6 (see this module's doc comment): a *static* (`light_angular_speed == 0.0`)
    // light's own beam comes within a character's proximity radius. `world.fixtures` (not
    // `tick0_fixtures`) is used here deliberately: as of the dragger/turret/light mover-tile
    // fix, this crate's own fixture positions track the reference exactly (proven by the
    // extended fixture-position comparison, `compare_extended`), so this tick's *own* fixture
    // position is available and correct — no need to fall back to a frozen tick-0 snapshot the
    // way condition 3 (a coarser, position-only proxy) still does.
    for l in world.fixtures.iter().filter_map(|f| match f {
        Fixture::Light(l) if l.angular_speed == 0.0 => Some(l),
        _ => None,
    }) {
        // The Stage A rule's geometry: the light's fixed direction `(sin(rot), cos(rot))` times its
        // configured length (a static light never rotates, so `rotation` is its construction
        // value).
        let direction = Vec2::new(l.rotation.sin(), l.rotation.cos());
        let raw_endpoint = l.pos + direction * (l.length as f32);
        let to = collision.intersect_no_laser(l.pos, raw_endpoint).collision;
        for row in &tick.characters {
            if row.ddrace.alive == 0 {
                continue;
            }
            let char_pos = Vec2::new(row.core.pos_x, row.core.pos_y);
            let Some(closest) = vmath::closest_point_on_line(l.pos, to, char_pos) else {
                continue;
            };
            // `IntersectedCharacters(m_Pos, m_To, 0.0f, nullptr)` (`light.cpp:34`): a `0`-radius
            // beam, so only the target character's own proximity radius matters.
            if vmath::distance(char_pos, closest) < CHARACTER_PROXIMITY_RADIUS {
                return Some("character within a static light's own beam (kind=6)");
            }
        }
    }
    for (i, row) in tick.characters.iter().enumerate() {
        let has_ninja = row.ddrace.weapon_got_mask & (1 << 5) != 0;
        let id = character_ids[i];
        let was = prev_ninja_mask.get(&id).copied().unwrap_or(false);
        if has_ninja && !was {
            return Some("ninja pickup consumed (weapon_got_mask bit 5 newly set)");
        }
        prev_ninja_mask.insert(id, has_ninja);
    }
    None
}

struct RunResult {
    compared_ticks: u64,
    /// Of `compared_ticks`: character-ticks at or after the retired Stage A cut (see the module
    /// doc comment) — i.e. ticks Stage A could not compare.
    previously_cut: u64,
    total_ticks: u64,
    character_count: u64,
    mismatch: Option<String>,
    cut_reason: Option<&'static str>,
    cut_tick: Option<i32>,
    stats: MechanicStats,
}

/// How much of each stage-B mechanic a trace actually exercises, counted from the *reference*
/// dump (so a mechanic the port wrongly never produces is still counted).
#[derive(Default, Clone, Copy)]
struct MechanicStats {
    /// Ticks with at least one `CLaser` (rifle or shotgun) alive.
    laser_ticks: u64,
    /// Sum over ticks of live `CLaser`s.
    laser_records: u64,
    /// Highest `m_Bounces` seen on a laser.
    max_bounces: i32,
    /// Ticks with at least one dragger beam alive.
    beam_ticks: u64,
    /// Sum over ticks of live dragger beams.
    beam_records: u64,
    /// Ticks with at least one turret shot (kind 7, trace version 3 only) alive.
    plasma_ticks: u64,
    /// Character-ticks with ninja as the active weapon.
    ninja_ticks: u64,
    /// Character-ticks in the ninja dash (`m_CurrentMoveTime > 0`).
    ninja_dash_ticks: u64,
    /// Ticks (counted on *our* simulation, which the comparison proves equal) with a laser that has
    /// just been teleported by a `TILE_TELEINWEAPON` tile and not yet continued (`m_WasTele`).
    tele_laser_ticks: u64,
    /// Ticks with a spent rifle laser whose owner holds a telegun laser and whose teleport was not
    /// cancelled (`HasTelegunLaser`, `!m_TeleportCancelled`): the shots that arm `m_TeleGunTeleport`.
    telegun_laser_ticks: u64,
}

impl MechanicStats {
    fn add(&mut self, o: &MechanicStats) {
        self.laser_ticks += o.laser_ticks;
        self.laser_records += o.laser_records;
        self.max_bounces = self.max_bounces.max(o.max_bounces);
        self.beam_ticks += o.beam_ticks;
        self.beam_records += o.beam_records;
        self.plasma_ticks += o.plasma_ticks;
        self.ninja_ticks += o.ninja_ticks;
        self.ninja_dash_ticks += o.ninja_dash_ticks;
        self.tele_laser_ticks += o.tele_laser_ticks;
        self.telegun_laser_ticks += o.telegun_laser_ticks;
    }

    fn observe_ours(&mut self, world: &World<f32>) {
        let (mut tele, mut telegun) = (false, false);
        for slot in &world.lasers {
            if let LaserSlot::Laser(l) = slot {
                tele |= l.was_tele;
                telegun |= l.weapon_type == 4
                    && l.energy <= 0.0
                    && !l.teleport_cancelled
                    && world.cores.get(l.owner as u8).is_some_and(|c| c.has_telegun_laser);
            }
        }
        self.tele_laser_ticks += u64::from(tele);
        self.telegun_laser_ticks += u64::from(telegun);
    }

    fn observe(&mut self, tick: &oracle_b_format::TraceBTick) {
        let (mut lasers, mut beams, mut plasma) = (0u64, 0u64, 0u64);
        for e in &tick.entities {
            match e.kind {
                1 => {
                    lasers += 1;
                    self.max_bounces = self.max_bounces.max(e.extra);
                }
                4 => beams += 1,
                7 => plasma += 1,
                _ => {}
            }
        }
        self.laser_ticks += u64::from(lasers > 0);
        self.laser_records += lasers;
        self.beam_ticks += u64::from(beams > 0);
        self.beam_records += beams;
        self.plasma_ticks += u64::from(plasma > 0);
        for row in &tick.characters {
            if row.ddrace.alive != 0 && row.core.active_weapon == 5 {
                self.ninja_ticks += 1;
                if row.ddrace.ninja_current_move_time > 0 {
                    self.ninja_dash_ticks += 1;
                }
            }
        }
    }
}

/// One scenario character entry as `run_trace` needs it: `(id, spawn_x, spawn_y, team)`.
type ScenarioCharacter = (u32, i32, i32, i32);

fn run_trace(trb_path: &Path) -> RunResult {
    let trb_bytes = std::fs::read(trb_path).unwrap_or_else(|e| panic!("failed to read {}: {e}", trb_path.display()));
    let mut trace = TraceBReader::new(&trb_bytes);
    let seed = metadata_seed(&trace.metadata_json);

    let scn_path = trb_path.with_extension("scn");
    let (map, scenario_characters, embedded_seed, cfg_lines): (
        ddai_physics::map::MapData,
        Vec<ScenarioCharacter>,
        u64,
        Vec<String>,
    ) = if scn_path.is_file() {
        let scn_bytes = std::fs::read(&scn_path).unwrap();
        let scn = ScenarioV3::read_bytes(&scn_bytes);
        let rawmap_path = resolve_rawmap_path(&scn_path, &scn.rawmap_path);
        let rawmap_bytes =
            std::fs::read(&rawmap_path).unwrap_or_else(|e| panic!("failed to read {}: {e}", rawmap_path.display()));
        let actual_sha = ddai_trace::hash::sha256(&rawmap_bytes);
        assert_eq!(
            actual_sha,
            scn.map_sha256,
            "rawmap sha256 mismatch for {}",
            rawmap_path.display()
        );
        let map = ddai_trace::rawmap::read(&rawmap_bytes).expect("rawmap parse failed");
        let chars: Vec<(u32, i32, i32, i32)> = scn
            .characters
            .iter()
            .map(|c| (c.id, c.spawn_x, c.spawn_y, c.team))
            .collect();
        (map, chars, scn.embedded_seed, scn.cfg_lines)
    } else {
        panic!(
            "no companion .scn file for {} (recipe-only traces need a separate reader path, not yet exercised by this test)",
            trb_path.display()
        );
    };
    assert_eq!(
        embedded_seed,
        seed,
        "scenario/trace seed mismatch for {}",
        trb_path.display()
    );

    let mut world: World<f32> = World::from_map(&map, seed);
    // `World::init` models the *whole* `CGameContext::OnInit()` config/tuning/switcher sequence
    // (see its doc comment): the pre-init `--cfg` pass, the unconditional tune-zone reset, the
    // `sv_ddrace_tune_reset`-gated reset, the map's own embedded "Settings" strings (stored on
    // `World` by `from_map`), the `CFGFLAG_GAME` lock, and the `sv_solo_server` check — in that
    // order. The harness's `--cfg` file is executed twice in the real server (once before
    // `OnInit()`, once after) with the *same* content, so `cfg_lines` is passed to both
    // `init` (pre-init) and `apply_commands` (post-init, now locked for `CFGFLAG_GAME` vars).
    match world.init(cfg_lines.iter().map(|s| s.as_str())) {
        Ok(()) => {}
        Err(e) => panic!("{}: unrecognized command (pre-init) {:?}", trb_path.display(), e),
    }
    match world.apply_commands(cfg_lines.iter().map(|s| s.as_str())) {
        Ok(()) => {}
        Err(e) => panic!("{}: unrecognized command (post-init) {:?}", trb_path.display(), e),
    }

    for &(id, sx, sy, team) in &scenario_characters {
        world.players[id as usize] = Some(Player::new(0));
        world::spawn_character(&mut world, id as i32, Vec2::new(sx as f32, sy as f32));
        if team != 0 {
            world::set_force_character_team(&mut world, id as i32, team);
        }
        if let Some(bonus) = weapon_variety_bonus(seed, id) {
            world::give_weapon_to(&mut world, id as i32, bonus);
        }
    }

    let mut prev_ninja_mask = std::collections::HashMap::new();
    let mut death_tracking = DeathTracking::new(trace.character_ids.len());
    let mut compared_ticks: u64 = 0;
    let mut previously_cut: u64 = 0;
    let mut stats = MechanicStats::default();
    let mut mismatch: Option<String> = None;
    let mut cut_reason: Option<&'static str> = None;
    let mut cut_tick: Option<i32> = None;
    let mut tick0_fixtures: Vec<FixtureSnapshot> = Vec::new();
    let total_ticks = trace.tick_count as u64;
    let character_count = trace.character_ids.len() as u64;

    let mut tick_index: i32 = 0;
    while let Some(reference) = trace.next_tick() {
        if tick_index == 0 {
            for e in &reference.entities {
                if e.kind == 5 || e.kind == 6 {
                    tick0_fixtures.push(FixtureSnapshot {
                        kind: e.kind,
                        pos_x: e.pos_x,
                        pos_y: e.pos_y,
                    });
                }
            }
        }

        if mismatch.is_none() {
            if cut_reason.is_none()
                && let Some(reason) = cut_signal(
                    &reference,
                    &tick0_fixtures,
                    &mut prev_ninja_mask,
                    &trace.character_ids,
                    &world,
                )
            {
                // Diagnostic only (module doc comment): record the retired cut, keep comparing.
                cut_reason = Some(reason);
                cut_tick = Some(tick_index);
            }
            stats.observe(&reference);

            let inputs: Vec<TickInput> = scenario_characters
                .iter()
                .zip(reference.characters.iter())
                .map(|(&(id, ..), row)| TickInput {
                    id: id as u8,
                    input: PlayerInput {
                        direction: row.input.direction,
                        target_x: row.input.target_x,
                        target_y: row.input.target_y,
                        jump: row.input.jump,
                        fire: row.input.fire,
                        hook: row.input.hook,
                        player_flags: row.input.player_flags,
                        wanted_weapon: row.input.wanted_weapon,
                        next_weapon: row.input.next_weapon,
                        prev_weapon: row.input.prev_weapon,
                    },
                    kill: row.input.kill != 0,
                })
                .collect();
            let mut sorted = inputs.clone();
            sorted.sort_by_key(|ti| ti.id);
            world.step(&sorted);
            stats.observe_ours(&world);
            assert_eq!(
                world.tick, reference.game_tick,
                "game tick counter diverged at index {tick_index}"
            );
            if let Some((from, to)) = debug_window()
                && (from..=to).contains(&tick_index)
            {
                debug_print_tick(&world, &reference, tick_index, &scenario_characters);
            }
            for (&(id, ..), row) in scenario_characters.iter().zip(reference.characters.iter()) {
                if let Some(reason) = compare_character(&world, id as i32, row) {
                    mismatch = Some(format!(
                        "tick {tick_index} (game_tick {}) character {id}: {reason}",
                        reference.game_tick
                    ));
                    break;
                }
            }
            if mismatch.is_none()
                && let Some(reason) = compare_extended(
                    &world,
                    &reference,
                    &trace.switch_team_ids,
                    trace.switch_highest_number,
                    &trace.character_ids,
                    trace.version,
                    &mut death_tracking,
                )
            {
                mismatch = Some(format!(
                    "tick {tick_index} (game_tick {}) extended: {reason}",
                    reference.game_tick
                ));
            }
            if mismatch.is_none() {
                compared_ticks += character_count;
                if cut_reason.is_some() {
                    previously_cut += character_count;
                }
            }
        }
        tick_index += 1;
    }

    RunResult {
        compared_ticks,
        previously_cut,
        total_ticks,
        character_count,
        mismatch,
        cut_reason,
        cut_tick,
        stats,
    }
}

/// Recovers `(recipe, seed)` from a recipe trace's filename
/// (`recipe_{name}_seed{seed}.trb`, `bulk_run_server.sh`'s `recipe_${recipe//-/_}_seed${seed}`
/// — only `tele-speedup`'s hyphen is ever substituted, since it is the only recipe name
/// containing one).
fn parse_recipe_filename(trb_path: &Path) -> (String, u64) {
    let stem = trb_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("non-UTF8 filename: {}", trb_path.display()));
    let rest = stem
        .strip_prefix("recipe_")
        .unwrap_or_else(|| panic!("recipe trace filename must start with 'recipe_': {stem}"));
    let (name_part, seed_part) = rest
        .rsplit_once("_seed")
        .unwrap_or_else(|| panic!("recipe trace filename must contain '_seed': {stem}"));
    let seed: u64 = seed_part
        .parse()
        .unwrap_or_else(|e| panic!("bad seed in recipe trace filename {stem}: {e}"));
    let recipe = if name_part == "tele_speedup" {
        "tele-speedup".to_string()
    } else {
        name_part.to_string()
    };
    (recipe, seed)
}

/// Replays a synthetic-recipe trace (`arena`/`freeze`/`front`/`tele-speedup`) — unlike a
/// real-map trace, the corpus ships no companion scenario file for these (`bulk_run_server.sh`'s
/// recipe loop never passes `--emit-scenario-v3`), so the scenario itself is regenerated
/// deterministically from `(recipe, seed, ticks, characters)` via
/// `ddai_trace::generator::random_v1` — the *same* generator `trace gen-scenario` used to
/// produce the (unsaved) scenario file the real harness replayed, per that function's own
/// determinism contract. These traces were recorded via `--rawmap`/`--scenario` (`RawmapMode`),
/// not `--real-map`: no `--cfg` file exists for them at all (so [`World::init`] gets an empty
/// pre-init pass and there is no post-init pass either), and the scenario's own
/// `tuning_overrides` are a direct `NetworkArray` poke (`oracle_server.cpp`'s
/// `GlobalTuning()->NetworkArray()[Index] = O.ValueX100`) applied *after* `World::init` (whose
/// own tune-zone reset would otherwise wipe it) — mirroring `tests/common/mod.rs`'s
/// `build_tuning` (the same technique task 1.3's Oracle A parity tests already use). Every
/// recipe scenario has `no_weak_hook == false` and no per-character team (`random_v1`'s own
/// fixed defaults — see [`ddai_trace::scenario::Scenario`]'s doc comment), so neither is
/// threaded through here.
fn run_recipe_trace(trb_path: &Path) -> RunResult {
    let trb_bytes = std::fs::read(trb_path).unwrap_or_else(|e| panic!("failed to read {}: {e}", trb_path.display()));
    let mut trace = TraceBReader::new(&trb_bytes);
    let seed = metadata_seed(&trace.metadata_json);
    let (recipe, seed_from_name) = parse_recipe_filename(trb_path);
    assert_eq!(
        seed,
        seed_from_name,
        "recipe trace {}: metadata seed != filename seed",
        trb_path.display()
    );
    let ticks = trace.tick_count;
    let characters = trace.character_ids.len() as u32;

    let map = ddai_trace::synthetic::build(&recipe).unwrap_or_else(|| panic!("unknown recipe '{recipe}'"));
    let scenario = ddai_trace::generator::random_v1(
        &recipe,
        ddai_trace::generator::Params {
            seed,
            ticks,
            characters,
        },
    )
    .unwrap_or_else(|e| panic!("random_v1({recipe}, seed={seed}, ticks={ticks}, chars={characters}) failed: {e:?}"));
    assert!(!scenario.no_weak_hook, "random_v1 scenarios never set no_weak_hook");

    let mut world: World<f32> = World::from_map(&map, seed);
    // No `--cfg` file at all for a recipe/`--rawmap` trace (see this function's doc comment).
    world.init(std::iter::empty::<&str>()).unwrap();
    for o in &scenario.tuning_overrides {
        let idx = (0..ddai_physics::tuning::TuningParams::num())
            .find(|&i| ddai_physics::tuning::TuningParams::name(i).eq_ignore_ascii_case(&o.name))
            .unwrap_or_else(|| panic!("unknown tuning parameter '{}'", o.name));
        assert!(
            world.tuning.zone_mut(0).set_raw(idx, o.value_x100),
            "set_raw rejected a name `num()`/`name()` themselves just validated"
        );
    }

    for c in &scenario.characters {
        world.players[c.id as usize] = Some(Player::new(0));
        world::spawn_character(&mut world, c.id as i32, Vec2::new(c.spawn_x as f32, c.spawn_y as f32));
        if let Some(bonus) = weapon_variety_bonus(seed, c.id) {
            world::give_weapon_to(&mut world, c.id as i32, bonus);
        }
    }

    let mut prev_positions = scenario.spawn_positions();
    let mut prev_ninja_mask = std::collections::HashMap::new();
    let mut death_tracking = DeathTracking::new(trace.character_ids.len());
    let mut compared_ticks: u64 = 0;
    let mut previously_cut: u64 = 0;
    let mut stats = MechanicStats::default();
    let mut mismatch: Option<String> = None;
    let mut cut_reason: Option<&'static str> = None;
    let mut cut_tick: Option<i32> = None;
    let mut tick0_fixtures: Vec<FixtureSnapshot> = Vec::new();
    let total_ticks = trace.tick_count as u64;
    let character_count = trace.character_ids.len() as u64;

    let mut tick_index: i32 = 0;
    while let Some(reference) = trace.next_tick() {
        if tick_index == 0 {
            for e in &reference.entities {
                if e.kind == 5 || e.kind == 6 {
                    tick0_fixtures.push(FixtureSnapshot {
                        kind: e.kind,
                        pos_x: e.pos_x,
                        pos_y: e.pos_y,
                    });
                }
            }
        }

        if mismatch.is_none() {
            if cut_reason.is_none()
                && let Some(reason) = cut_signal(
                    &reference,
                    &tick0_fixtures,
                    &mut prev_ninja_mask,
                    &trace.character_ids,
                    &world,
                )
            {
                // Diagnostic only (module doc comment): record the retired cut, keep comparing.
                cut_reason = Some(reason);
                cut_tick = Some(tick_index);
            }
            stats.observe(&reference);

            let scenario_inputs = &scenario.inputs[tick_index as usize];
            let mut resolved = Vec::with_capacity(scenario.characters.len());
            for (slot, input) in scenario_inputs.iter().enumerate() {
                resolved.push(ddai_trace::scenario::resolve_input(input, slot, &prev_positions));
            }
            let inputs: Vec<TickInput> = scenario
                .characters
                .iter()
                .zip(resolved.iter())
                .map(|(c, r)| TickInput {
                    id: c.id as u8,
                    input: PlayerInput {
                        direction: r.direction,
                        target_x: r.target_x,
                        target_y: r.target_y,
                        jump: r.jump,
                        fire: r.fire,
                        hook: r.hook,
                        player_flags: r.player_flags,
                        wanted_weapon: r.wanted_weapon,
                        next_weapon: r.next_weapon,
                        prev_weapon: r.prev_weapon,
                    },
                    kill: false,
                })
                .collect();
            let mut sorted = inputs.clone();
            sorted.sort_by_key(|ti| ti.id);
            world.step(&sorted);
            stats.observe_ours(&world);
            assert_eq!(
                world.tick, reference.game_tick,
                "game tick counter diverged at index {tick_index}"
            );

            for (slot, c) in scenario.characters.iter().enumerate() {
                if let Some(row) = reference.characters.get(slot) {
                    if let Some(slot_idx) = world.cores.slot_of(c.id as u8) {
                        let core = world.cores.core_at(slot_idx);
                        prev_positions[slot] = (core.pos.x as i32, core.pos.y as i32);
                    }
                    if let Some(reason) = compare_character(&world, c.id as i32, row) {
                        mismatch = Some(format!(
                            "tick {tick_index} (game_tick {}) character {}: {reason}",
                            reference.game_tick, c.id
                        ));
                        break;
                    }
                }
            }
            if mismatch.is_none()
                && let Some(reason) = compare_extended(
                    &world,
                    &reference,
                    &trace.switch_team_ids,
                    trace.switch_highest_number,
                    &trace.character_ids,
                    trace.version,
                    &mut death_tracking,
                )
            {
                mismatch = Some(format!(
                    "tick {tick_index} (game_tick {}) extended: {reason}",
                    reference.game_tick
                ));
            }
            if mismatch.is_none() {
                compared_ticks += character_count;
                if cut_reason.is_some() {
                    previously_cut += character_count;
                }
            }
        }
        tick_index += 1;
    }

    RunResult {
        compared_ticks,
        previously_cut,
        total_ticks,
        character_count,
        mismatch,
        cut_reason,
        cut_tick,
        stats,
    }
}

/// Per-character state [`compare_extended`]'s `died_this_tick`/`respawned_this_tick` check needs
/// to track across ticks, mirroring the oracle server's own `WasAlive`/`PrevDieTick` bookkeeping
/// exactly (`oracle_server.cpp`'s dump loop: `DiedThisTickReal = (WasAlive && !Alive) || (Tick >
/// 0 && Alive && DieTick != PrevDieTick)`, `RespawnedThisTickReal = (!WasAlive && Alive) ||
/// (Tick > 0 && Alive && SpawnTick == GameTick)`). Indexed the same way `character_ids`/
/// `scenario_characters` is. Every character starts alive (from its initial spawn, which never
/// counts as a "respawn"), with `die_tick == 0` (a fresh [`Player`]'s own default).
struct DeathTracking {
    prev_alive: Vec<bool>,
    rec_die_tick: Vec<i32>,
}

impl DeathTracking {
    fn new(n: usize) -> Self {
        DeathTracking {
            prev_alive: vec![true; n],
            rec_die_tick: vec![0; n],
        }
    }
}

/// Review round 1, finding F1: everything in the trace-b schema `compare_character` doesn't
/// already cover — `GlobalTickFields.switches` (per switch, per team), every live projectile
/// (field by field, in dump order — order matters, see the crazy-shotgun/entity-order fix), the
/// full set of kind-2/3/5/6 fixture positions (`CDoor`/`CDragger`/`CGun`/`CLight` — compared as
/// an unordered multiset: real DDNet's own entity-list order for these is a leaky `InsertEntity`
/// implementation detail this crate has no independent way to re-derive for a `CDoor`
/// specifically, unlike projectiles, so requiring positional order here would just make an
/// already-fragile comparison more so for no parity benefit), and each character's
/// `died_this_tick`/`respawned_this_tick` edge-triggered flags. Returns `Some(reason)` on the
/// first mismatch found.
fn compare_extended(
    world: &World<f32>,
    reference: &oracle_b_format::TraceBTick,
    switch_team_ids: &[i32],
    switch_highest_number: u32,
    character_ids: &[u32],
    trace_version: u32,
    tracking: &mut DeathTracking,
) -> Option<String> {
    for (ti, &team) in switch_team_ids.iter().enumerate() {
        for n in 1..=switch_highest_number as usize {
            let r = reference.switches[ti * switch_highest_number as usize + (n - 1)];
            let Some(sw) = world.cores.switchers.get(n) else {
                return Some(format!("switch {n} team {team}: missing on our side"));
            };
            let ours = (
                sw.status[team as usize],
                sw.end_tick[team as usize],
                sw.kind[team as usize],
                sw.last_update_tick[team as usize],
            );
            let theirs = (r.status != 0, r.end_tick, r.kind, r.last_update_tick);
            if ours != theirs {
                return Some(format!("switch {n} team {team}: ours={ours:?} theirs={theirs:?}"));
            }
        }
    }

    let ref_projectiles: Vec<_> = reference.entities.iter().filter(|e| e.kind == 0).collect();
    if ref_projectiles.len() != world.projectiles.len() {
        return Some(format!(
            "projectile count: ours={} theirs={}",
            world.projectiles.len(),
            ref_projectiles.len()
        ));
    }
    for (k, (e, p)) in ref_projectiles.iter().zip(world.projectiles.iter()).enumerate() {
        let ours = (
            p.owner,
            p.weapon_type,
            p.pos.x.to_bits(),
            p.pos.y.to_bits(),
            p.direction.x.to_bits(),
            p.direction.y.to_bits(),
            p.start_tick,
            p.life_span,
        );
        let theirs = (
            e.owner_client_id,
            e.weapon_type,
            e.pos_x.to_bits(),
            e.pos_y.to_bits(),
            e.dir_x.to_bits(),
            e.dir_y.to_bits(),
            e.start_tick,
            e.extra,
        );
        if ours != theirs {
            return Some(format!("projectile #{k}: ours={ours:?} theirs={theirs:?}"));
        }
    }

    if let Some(reason) = compare_laser_list(world, reference, trace_version) {
        return Some(reason);
    }

    for (i, (&id, row)) in character_ids.iter().zip(reference.characters.iter()).enumerate() {
        let alive = world.characters[id as usize].is_some_and(|c| c.alive);
        let die_tick = world.players[id as usize].map(|p| p.die_tick).unwrap_or(0);
        let spawn_tick = world.characters[id as usize].map(|c| c.spawn_tick).unwrap_or(-1);
        // The harness's `Tick > 0` guards (`oracle_server.cpp`'s `DiedThisTickReal`/
        // `RespawnedThisTickReal`): the very first tick (game tick 1) never reports the same-tick
        // forms, even when a kill bit at tick 0 really did kill and respawn the tee.
        let not_first = reference.game_tick > 1;
        let died = (tracking.prev_alive[i] && !alive) || (not_first && alive && die_tick != tracking.rec_die_tick[i]);
        let respawned = (!tracking.prev_alive[i] && alive) || (not_first && alive && spawn_tick == reference.game_tick);
        let theirs = (row.ddrace.died_this_tick != 0, row.ddrace.respawned_this_tick != 0);
        if (died, respawned) != theirs {
            return Some(format!(
                "char {id} died/respawned: ours=({died},{respawned}) theirs={theirs:?}"
            ));
        }
        if alive {
            tracking.rec_die_tick[i] = die_tick;
        }
        tracking.prev_alive[i] = alive;
    }

    None
}

/// The `ENTTYPE_LASER` entity list (everything in the dump that is not a projectile), compared in
/// the reference's own order: dynamic entities first (newest first — `World::lasers` is stored
/// oldest first, so it is walked back to front), then the static fixtures. Doors, draggers,
/// turrets and lights are compared per kind in list order (their cross-kind interleaving is a
/// map-scan artefact nothing depends on: doors never tick and the ticking fixtures keep their own
/// relative order, which the per-kind comparison pins).
///
/// Trace version 2 (the v1 corpus) predates two dump fields, so for it: dragger beams are
/// compared without their `owner` (the dump carried `-1`) and turret shots (kind 7), which that
/// dump never listed, are left out of our side.
fn compare_laser_list(
    world: &World<f32>,
    reference: &oracle_b_format::TraceBTick,
    trace_version: u32,
) -> Option<String> {
    let v3 = trace_version >= 3;
    let theirs_dynamic: Vec<_> = reference
        .entities
        .iter()
        .filter(|e| matches!(e.kind, 1 | 4 | 7))
        .collect();
    let ours_dynamic: Vec<&LaserSlot<f32>> = world
        .lasers
        .iter()
        .rev()
        .filter(|l| v3 || !matches!(l, LaserSlot::Plasma(_)))
        .collect();
    if theirs_dynamic.len() != ours_dynamic.len() {
        return Some(format!(
            "laser-list length (lasers/beams/plasma): ours={} theirs={}",
            ours_dynamic.len(),
            theirs_dynamic.len()
        ));
    }
    for (k, (e, slot)) in theirs_dynamic.iter().zip(ours_dynamic.iter()).enumerate() {
        let theirs = (
            e.kind,
            e.owner_client_id,
            e.weapon_type,
            e.pos_x.to_bits(),
            e.pos_y.to_bits(),
            e.dir_x.to_bits(),
            e.dir_y.to_bits(),
            e.start_tick,
            e.extra,
        );
        let ours = match slot {
            LaserSlot::Laser(l) => (
                1,
                l.owner,
                l.weapon_type,
                l.pos.x.to_bits(),
                l.pos.y.to_bits(),
                l.dir.x.to_bits(),
                l.dir.y.to_bits(),
                l.eval_tick,
                l.bounces,
            ),
            LaserSlot::Beam(b) => (
                4,
                if v3 { b.for_client } else { -1 },
                0,
                b.pos.x.to_bits(),
                b.pos.y.to_bits(),
                0,
                0,
                0,
                0,
            ),
            LaserSlot::Plasma(p) => (
                7,
                p.for_client,
                i32::from(p.explosive) | (i32::from(p.freeze) << 1),
                p.pos.x.to_bits(),
                p.pos.y.to_bits(),
                p.core.x.to_bits(),
                p.core.y.to_bits(),
                p.eval_tick,
                p.life_time,
            ),
        };
        if ours != theirs {
            return Some(format!("laser-list entry #{k}: ours={ours:?} theirs={theirs:?}"));
        }
    }

    let theirs_of = |kind: i32| -> Vec<&oracle_b_format::EntityRecord> {
        reference.entities.iter().filter(|e| e.kind == kind).collect()
    };
    let theirs_doors = theirs_of(2);
    if theirs_doors.len() != world.doors.len()
        || theirs_doors
            .iter()
            .zip(world.doors.iter())
            .any(|(e, d)| (e.pos_x.to_bits(), e.pos_y.to_bits()) != (d.pos.x.to_bits(), d.pos.y.to_bits()))
    {
        return Some(format!(
            "door positions differ (ours {} theirs {})",
            world.doors.len(),
            theirs_doors.len()
        ));
    }
    for kind in [3, 5, 6] {
        let theirs = theirs_of(kind);
        let ours: Vec<&Fixture<f32>> = world.fixtures.iter().filter(|f| f.dump_kind() == kind).collect();
        if theirs.len() != ours.len() {
            return Some(format!(
                "kind {kind} fixture count: ours={} theirs={}",
                ours.len(),
                theirs.len()
            ));
        }
        for (k, (e, f)) in theirs.iter().zip(ours.iter()).enumerate() {
            let pos = f.pos();
            if (e.pos_x.to_bits(), e.pos_y.to_bits()) != (pos.x.to_bits(), pos.y.to_bits()) {
                return Some(format!(
                    "kind {kind} fixture #{k} position: ours=({}, {}) theirs=({}, {})",
                    pos.x, pos.y, e.pos_x, e.pos_y
                ));
            }
            if let Fixture::Light(l) = f {
                // The dump's `weapon_type` is `m_Length`, `dir_x` is `m_AngularSpeed`, `extra`
                // is `m_Speed` (`docs/formats.md` §11.2).
                let ours = (l.length, l.angular_speed.to_bits(), l.speed);
                let theirs = (e.weapon_type, e.dir_x.to_bits(), e.extra);
                if ours != theirs {
                    return Some(format!(
                        "light #{k} length/angular_speed/speed: ours={ours:?} theirs={theirs:?}"
                    ));
                }
            }
        }
    }
    None
}

/// Compares one character's simulated state against its reference row. Returns `Some(reason)` on
/// the first mismatch.
fn compare_character(world: &World<f32>, id: i32, row: &oracle_b_format::CharacterRow) -> Option<String> {
    let alive = world.characters[id as usize].is_some_and(|c| c.alive);
    if alive != (row.ddrace.alive != 0) {
        return Some(format!("alive: ours={alive} theirs={}", row.ddrace.alive));
    }
    if !alive {
        // Dead characters' fields are "frozen" at their last known value in the reference trace
        // (`docs/formats.md` §11.3) — this crate doesn't attempt to reproduce that freezing
        // (it simply stops updating a removed character's fields), so only `alive` itself (and
        // the two edge-triggered flags, not compared here) are meaningful for a dead character.
        return None;
    }
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return Some("character marked alive but has no core slot".to_string());
    };
    let core = world.cores.core_at(slot);
    macro_rules! check_f32 {
        ($ours:expr, $theirs:expr, $name:literal) => {
            if $ours.to_bits() != $theirs.to_bits() {
                return Some(format!(
                    "{}: ours={:?} ({:#x}) theirs={:?} ({:#x})",
                    $name,
                    $ours,
                    $ours.to_bits(),
                    $theirs,
                    $theirs.to_bits()
                ));
            }
        };
    }
    macro_rules! check_i32 {
        ($ours:expr, $theirs:expr, $name:literal) => {
            if $ours != $theirs {
                return Some(format!("{}: ours={} theirs={}", $name, $ours, $theirs));
            }
        };
    }
    check_f32!(core.pos.x, row.core.pos_x, "pos_x");
    check_f32!(core.pos.y, row.core.pos_y, "pos_y");
    check_f32!(core.vel.x, row.core.vel_x, "vel_x");
    check_f32!(core.vel.y, row.core.vel_y, "vel_y");
    check_f32!(core.hook_pos.x, row.core.hook_pos_x, "hook_pos_x");
    check_f32!(core.hook_pos.y, row.core.hook_pos_y, "hook_pos_y");
    check_f32!(core.hook_dir.x, row.core.hook_dir_x, "hook_dir_x");
    check_f32!(core.hook_dir.y, row.core.hook_dir_y, "hook_dir_y");
    check_f32!(core.hook_tele_base.x, row.core.hook_tele_base_x, "hook_tele_base_x");
    check_f32!(core.hook_tele_base.y, row.core.hook_tele_base_y, "hook_tele_base_y");
    check_i32!(core.hook_tick, row.core.hook_tick, "hook_tick");
    check_i32!(core.hook_state, row.core.hook_state, "hook_state");
    check_i32!(core.hooked_player(), row.core.hooked_player, "hooked_player");
    check_i32!(core.active_weapon, row.core.active_weapon, "active_weapon");
    check_i32!(core.new_hook as i32, row.core.new_hook, "new_hook");
    check_i32!(core.jumped, row.core.jumped, "jumped");
    check_i32!(core.jumped_total, row.core.jumped_total, "jumped_total");
    check_i32!(core.jumps, row.core.jumps, "jumps");
    check_i32!(core.direction, row.core.direction, "direction");
    check_i32!(core.angle, row.core.angle, "angle");
    check_i32!(core.triggered_events, row.core.triggered_events, "triggered_events");
    check_i32!(core.colliding, row.core.colliding, "colliding");
    check_i32!(core.left_wall as i32, row.core.left_wall, "left_wall");
    // Unlike task 1.3/Oracle A (core-only, switchers always empty — `docs/formats.md` §5.5),
    // Stage A's `World` has real switches/doors, so `m_MoveRestrictions`'s last-computed value
    // (`handle_tiles`'s switch-aware, anti-skip-`MapIndex`-overridden recomputation) is read
    // directly off the core rather than rederived with the switch-*unaware* simple formula.
    check_i32!(
        core.move_restrictions(),
        row.core.move_restrictions,
        "move_restrictions"
    );
    check_i32!(core.solo as i32, row.core.solo, "solo");
    check_i32!(
        core.collision_disabled as i32,
        row.core.collision_disabled,
        "collision_disabled"
    );
    check_i32!(core.endless_hook as i32, row.core.endless_hook, "endless_hook");
    check_i32!(
        core.hook_hit_disabled as i32,
        row.core.hook_hit_disabled,
        "hook_hit_disabled"
    );

    let character = world.characters[id as usize].unwrap();
    check_i32!(character.freeze_time, row.ddrace.freeze_time, "freeze_time");
    check_i32!(core.is_in_freeze as i32, row.ddrace.is_in_freeze, "is_in_freeze");
    check_i32!(core.deep_frozen as i32, row.ddrace.deep_frozen, "deep_frozen");
    check_i32!(core.live_frozen as i32, row.ddrace.live_frozen, "live_frozen");
    check_i32!(
        character.frozen_last_tick as i32,
        row.ddrace.frozen_last_tick,
        "frozen_last_tick"
    );
    check_i32!(character.reload_timer, row.ddrace.reload_timer, "reload_timer");
    check_i32!(character.attack_tick, row.ddrace.attack_tick, "attack_tick");
    check_i32!(character.queued_weapon, row.ddrace.queued_weapon, "queued_weapon");
    check_i32!(character.last_weapon, row.ddrace.last_weapon, "last_weapon");
    let got_mask: i32 = (0..6).map(|w| i32::from(core.weapons[w].got) << w).sum();
    check_i32!(got_mask, row.ddrace.weapon_got_mask, "weapon_got_mask");
    for w in 0..6 {
        check_i32!(core.weapons[w].ammo, row.ddrace.weapon_ammo[w], "weapon_ammo");
        check_i32!(
            core.weapons[w].ammo_regen_start,
            row.ddrace.weapon_ammo_regen_start[w],
            "weapon_ammo_regen_start"
        );
    }
    check_i32!(
        core.ninja.activation_tick,
        row.ddrace.ninja_activation_tick,
        "ninja_activation_tick"
    );
    check_i32!(
        core.ninja.current_move_time,
        row.ddrace.ninja_current_move_time,
        "ninja_current_move_time"
    );
    check_i32!(
        core.ninja.old_vel_amount,
        row.ddrace.ninja_old_vel_amount,
        "ninja_old_vel_amount"
    );
    check_f32!(
        core.ninja.activation_dir.x,
        row.ddrace.ninja_activation_dir_x,
        "ninja_activation_dir_x"
    );
    check_f32!(
        core.ninja.activation_dir.y,
        row.ddrace.ninja_activation_dir_y,
        "ninja_activation_dir_y"
    );
    check_i32!(character.tele_checkpoint, row.ddrace.tele_checkpoint, "tele_checkpoint");
    check_i32!(core.endless_jump as i32, row.ddrace.endless_jump, "endless_jump");
    check_i32!(core.jetpack as i32, row.ddrace.jetpack, "jetpack");
    check_i32!(core.is_super as i32, row.ddrace.is_super, "super");
    check_i32!(core.invincible as i32, row.ddrace.invincible, "invincible");
    check_i32!(
        core.hammer_hit_disabled as i32,
        row.ddrace.hammer_hit_disabled,
        "hammer_hit_disabled"
    );
    check_i32!(
        core.grenade_hit_disabled as i32,
        row.ddrace.grenade_hit_disabled,
        "grenade_hit_disabled"
    );
    check_i32!(
        core.laser_hit_disabled as i32,
        row.ddrace.laser_hit_disabled,
        "laser_hit_disabled"
    );
    check_i32!(
        core.shotgun_hit_disabled as i32,
        row.ddrace.shotgun_hit_disabled,
        "shotgun_hit_disabled"
    );
    check_i32!(
        core.has_telegun_gun as i32,
        row.ddrace.has_telegun_gun,
        "has_telegun_gun"
    );
    check_i32!(
        core.has_telegun_grenade as i32,
        row.ddrace.has_telegun_grenade,
        "has_telegun_grenade"
    );
    check_i32!(
        core.has_telegun_laser as i32,
        row.ddrace.has_telegun_laser,
        "has_telegun_laser"
    );
    check_i32!(world.teams_core.team(id), row.ddrace.team, "team");
    check_i32!(character.strong_weak_id, row.ddrace.strong_weak_id, "strong_weak_id");
    check_i32!(core.freeze_start, row.ddrace.freeze_start, "freeze_start");
    check_i32!(core.freeze_end, row.ddrace.freeze_end, "freeze_end");
    check_i32!(character.tune_zone, row.ddrace.tune_zone, "tune_zone");
    check_i32!(character.num_inputs, row.ddrace.num_inputs, "num_inputs");
    check_i32!(
        character.last_refill_jumps as i32,
        row.ddrace.last_refill_jumps,
        "last_refill_jumps"
    );
    check_i32!(character.ddrace_state, row.ddrace.ddrace_state, "ddrace_state");
    check_i32!(character.start_time, row.ddrace.start_time, "start_time");
    let die_tick = world.players[id as usize].map(|p| p.die_tick).unwrap_or(0);
    check_i32!(die_tick, row.ddrace.die_tick, "die_tick");
    let spawning = world.players[id as usize].is_some_and(|p| p.spawning);
    check_i32!(spawning as i32, row.ddrace.spawning, "spawning");
    let previous_die_tick = world.players[id as usize].map(|p| p.previous_die_tick).unwrap_or(0);
    check_i32!(previous_die_tick, row.ddrace.previous_die_tick, "previous_die_tick");

    None
}

/// `DDAI_DEBUG_TICKS=from-to` makes [`run_trace`] print, for those tick indices, a one-line
/// side-by-side of our state and the reference's for every character plus our `lasers`/`fixtures`
/// summary — the tool for chasing a mismatch the one-line report doesn't explain.
fn debug_window() -> Option<(i32, i32)> {
    let v = std::env::var("DDAI_DEBUG_TICKS").ok()?;
    let (a, b) = v.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn debug_print_tick(
    world: &World<f32>,
    reference: &oracle_b_format::TraceBTick,
    tick_index: i32,
    scenario_characters: &[ScenarioCharacter],
) {
    for (&(id, ..), row) in scenario_characters.iter().zip(reference.characters.iter()) {
        let Some(slot) = world.cores.slot_of(id as u8) else {
            continue;
        };
        let c = world.cores.core_at(slot);
        let ch = world.characters[id as usize].unwrap();
        eprintln!(
            "[dbg t{tick_index} c{id}] ours: alive={} pos=({},{}) vel=({},{}) freeze={} w={} ninja(move={},dir=({},{}),old={}) hooked={} | theirs: alive={} pos=({},{}) vel=({},{}) freeze={} w={} ninja(move={},dir=({},{}),old={}) hooked={}",
            u8::from(ch.alive),
            c.pos.x,
            c.pos.y,
            c.vel.x,
            c.vel.y,
            ch.freeze_time,
            c.active_weapon,
            c.ninja.current_move_time,
            c.ninja.activation_dir.x,
            c.ninja.activation_dir.y,
            c.ninja.old_vel_amount,
            c.hooked_player(),
            row.ddrace.alive,
            row.core.pos_x,
            row.core.pos_y,
            row.core.vel_x,
            row.core.vel_y,
            row.ddrace.freeze_time,
            row.core.active_weapon,
            row.ddrace.ninja_current_move_time,
            row.ddrace.ninja_activation_dir_x,
            row.ddrace.ninja_activation_dir_y,
            row.ddrace.ninja_old_vel_amount,
            row.core.hooked_player,
        );
    }
    for e in &reference.entities {
        if matches!(e.kind, 1 | 4 | 7) {
            eprintln!(
                "[dbg t{tick_index}]   ref entity kind={} owner={} pos=({},{}) extra={}",
                e.kind, e.owner_client_id, e.pos_x, e.pos_y, e.extra
            );
        }
    }
}

fn assert_run_result(name: &str, result: &RunResult) {
    eprintln!(
        "{name}: compared {}/{} character-ticks ({:.1}%), previously-cut={:?} at tick {:?}, mismatch={:?}",
        result.compared_ticks,
        result.total_ticks * result.character_count,
        100.0 * result.compared_ticks as f64 / (result.total_ticks * result.character_count).max(1) as f64,
        result.cut_reason,
        result.cut_tick,
        result.mismatch
    );
    assert!(
        result.mismatch.is_none(),
        "{name}: mismatch found: {:?}",
        result.mismatch
    );
}

#[test]
fn replays_one_sample_real_map_trace() {
    let dir = corpus_dir();
    if !dir.is_dir() {
        eprintln!(
            "skipping: {} not found (Oracle B corpus not present in this environment)",
            dir.display()
        );
        return;
    }
    let path = dir.join("realmap_BlmapChill__seed10001.trb");
    if !path.is_file() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let result = run_trace(&path);
    assert_run_result("realmap_BlmapChill__seed10001", &result);
}

/// Fast smoke test for the recipe reader path (item 2), same role as
/// `replays_one_sample_real_map_trace` above but for a synthetic-recipe trace.
#[test]
fn replays_one_sample_recipe_trace() {
    let dir = corpus_dir();
    if !dir.is_dir() {
        eprintln!(
            "skipping: {} not found (Oracle B corpus not present in this environment)",
            dir.display()
        );
        return;
    }
    let path = dir.join("recipe_arena_seed20001.trb");
    if !path.is_file() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let result = run_recipe_trace(&path);
    assert_run_result("recipe_arena_seed20001", &result);
}

/// Per-map/per-recipe totals: `compared` (matched character-ticks — with no cut anymore this is
/// every tick of a trace until a mismatch), `previously_cut` (of those, the character-ticks the
/// retired Stage A cut rule — see the module doc comment — would have skipped), `lost_to_mismatch`
/// (character-ticks past a genuine mismatch — a defect) and `total` (`compared + lost_to_mismatch`,
/// always).
#[derive(Default, Clone, Copy)]
struct CoverageTotals {
    compared: u64,
    previously_cut: u64,
    lost_to_mismatch: u64,
    total: u64,
    stats: MechanicStats,
}

fn run_corpus_subset(entries: &[PathBuf]) {
    let mut per_map: std::collections::BTreeMap<String, CoverageTotals> = std::collections::BTreeMap::new();
    let mut failures = Vec::new();
    let mut grand_total = CoverageTotals::default();
    let quiet = std::env::var_os("DDAI_PARITY_QUIET").is_some();

    for path in entries {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let map_key = name.split("_seed").next().unwrap_or(&name).to_string();
        // Recipe traces (`recipe_*.trb`) have no companion `.scn` — see `run_recipe_trace`'s
        // doc comment for why they need their own reader path.
        let result = if path.with_extension("scn").is_file() {
            run_trace(path)
        } else {
            run_recipe_trace(path)
        };
        let total_possible = result.total_ticks * result.character_count;
        let lost_to_mismatch = total_possible - result.compared_ticks;

        for t in [per_map.entry(map_key).or_default(), &mut grand_total] {
            t.compared += result.compared_ticks;
            t.previously_cut += result.previously_cut;
            t.lost_to_mismatch += lost_to_mismatch;
            t.total += total_possible;
            t.stats.add(&result.stats);
        }

        if !quiet || result.mismatch.is_some() {
            eprintln!(
                "{name}: compared {}/{total_possible} ({:.1}%), previously beyond the cut {} (cut={:?}@{:?}), mismatch={:?}",
                result.compared_ticks,
                100.0 * result.compared_ticks as f64 / total_possible.max(1) as f64,
                result.previously_cut,
                result.cut_reason,
                result.cut_tick,
                result.mismatch
            );
        }
        if let Some(m) = result.mismatch {
            failures.push(format!("{name}: {m}"));
        }
    }

    eprintln!("=== per-map parity (compared / previously beyond the Stage A cut / lost to mismatch) ===");
    let line = |label: &str, t: &CoverageTotals| {
        eprintln!(
            "{label}: compared {}/{} ({:.1}%), previously-beyond-cut {} ({:.1}%), lost-to-mismatch {}; \
             laser-ticks {} (max bounces {}, tele-laser-ticks {}, telegun-laser-ticks {}), beam-ticks {} ({} beam-records), plasma-ticks {}, ninja-char-ticks {} (dash {})",
            t.compared,
            t.total,
            100.0 * t.compared as f64 / t.total.max(1) as f64,
            t.previously_cut,
            100.0 * t.previously_cut as f64 / t.total.max(1) as f64,
            t.lost_to_mismatch,
            t.stats.laser_ticks,
            t.stats.max_bounces,
            t.stats.tele_laser_ticks,
            t.stats.telegun_laser_ticks,
            t.stats.beam_ticks,
            t.stats.beam_records,
            t.stats.plasma_ticks,
            t.stats.ninja_ticks,
            t.stats.ninja_dash_ticks,
        );
    };
    for (map, t) in &per_map {
        line(map, t);
    }
    line(&format!("TOTAL across {} traces", entries.len()), &grand_total);

    assert!(
        failures.is_empty(),
        "{} trace(s) had a mismatch:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A quick, curated cross-section (7 maps x 11 seeds, incl. the `sv_hit 0`/`sv_solo_server`/
/// `--team` cfg slices `bulk_run_server.sh` produces every 5th/10th/7th seed) — cheap enough to
/// run un-ignored, giving fast, broad-but-not-exhaustive evidence between full-corpus runs.
///
/// Seeds `26`/`34`/`38`/`39`/`70` (added on top of the original `1,5,7,14,20,25`) pin the exact
/// evidence traces the diagnosis behind this crate's 5 `move_restrictions`/pickup-movement/
/// team-death-loop/explosion-position/solo-respawn fixes named (see this crate's `BUILD REPORT`):
/// `10026`/`10034` (Fix 1, `jumped`), `10038`/`10039` (Fix 2, freeze-pickup movement), `10007`/
/// `10014`/`10070` (Fix 3, teamed-character-death loop cutoff — `70` doubles as Fix 5's own
/// evidence seed), `10070` (Fix 5, `sv_solo_server` respawn). Fix 4's evidence
/// (`recipe_arena_seed20001`) and Fix 2's `Blockdale`/`active_weapon` evidence (seed `1`) were
/// already in range.
#[test]
fn replays_a_curated_cross_section() {
    let dir = corpus_dir();
    if !dir.is_dir() {
        eprintln!("skipping: {} not found", dir.display());
        return;
    }
    const MAPS: [&str; 7] = [
        "BlmapChill",
        "blmapV3multistarbox",
        "blmapV5_ddpp",
        "Blockdale",
        "BlockField",
        "ChillBlock5",
        "Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25",
    ];
    const SEEDS: [u32; 11] = [1, 5, 7, 14, 20, 25, 26, 34, 38, 39, 70];
    let mut entries = Vec::new();
    for map in MAPS {
        for seed in SEEDS {
            let p = dir.join(format!("realmap_{map}__seed{}.trb", 10000 + seed));
            if p.is_file() {
                entries.push(p);
            }
        }
    }
    // A small slice of every synthetic recipe too (item 2: "recipe traces are part of stage A").
    const RECIPES: [&str; 4] = ["arena", "freeze", "front", "tele_speedup"];
    for recipe in RECIPES {
        for seed in 1..=6u32 {
            let p = dir.join(format!("recipe_{recipe}_seed{}.trb", 20000 + seed));
            if p.is_file() {
                entries.push(p);
            }
        }
    }
    if entries.is_empty() {
        eprintln!("skipping: no curated files found under {}", dir.display());
        return;
    }
    run_corpus_subset(&entries);
}

/// Perf smoke test (not a correctness check — `run_trace`'s own comparisons still run, but this
/// only reports timing): `World::step()`'s throughput on one full, mechanically rich real-map
/// trace (3000 ticks x 3 characters), release mode. `#[ignore]`d like the corpus tests above
/// since it needs the corpus and is meant to be run explicitly, not on every `cargo test`.
#[test]
#[ignore]
fn measures_world_step_throughput_on_one_real_map_trace() {
    let dir = corpus_dir();
    if !dir.is_dir() {
        eprintln!(
            "skipping: {} not found (Oracle B corpus not present in this environment)",
            dir.display()
        );
        return;
    }
    let path = dir.join("realmap_BlmapChill__seed10001.trb");
    if !path.is_file() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let start = std::time::Instant::now();
    let result = run_trace(&path);
    let elapsed = start.elapsed();
    let character_ticks = result.total_ticks * result.character_count;
    eprintln!(
        "measures_world_step_throughput_on_one_real_map_trace: {} ticks x {} characters = {character_ticks} character-ticks in {:.3}ms ({:.0} character-ticks/s, {:.0} ticks/s)",
        result.total_ticks,
        result.character_count,
        elapsed.as_secs_f64() * 1000.0,
        character_ticks as f64 / elapsed.as_secs_f64(),
        result.total_ticks as f64 / elapsed.as_secs_f64(),
    );
}

#[test]
#[ignore]
fn replays_full_oracle_b_corpus() {
    let dir = corpus_dir();
    assert!(dir.is_dir(), "{} not found", dir.display());
    // Every real-map trace (`.trb` + companion `.scn`) plus every synthetic-recipe trace
    // (`recipe_*.trb`, no companion — `run_recipe_trace` regenerates its scenario instead):
    // the full 650-trace corpus, compared over every tick (no cut).
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "trb")
                && (p.with_extension("scn").is_file()
                    || p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("recipe_")))
        })
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no .trb traces found under {}", dir.display());
    run_corpus_subset(&entries);
}

/// Every `.trb` of the stage-B corpus (all of them have a companion `.scn` + rawmap; the crafted
/// ones are replayed through the same scenario-v3 path as the real-map traces).
#[test]
#[ignore]
fn replays_stage_b_corpus() {
    let dir = stage_b_corpus_dir();
    assert!(dir.is_dir(), "{} not found", dir.display());
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "trb") && p.with_extension("scn").is_file())
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no .trb traces found under {}", dir.display());
    run_corpus_subset(&entries);
}

/// opus-reviewer's own regression traces, crafted against an instrumented Oracle B across two
/// review rounds:
/// - Round 1: `ks_*` team/switch/respawn scenarios, `br_*` brawl scenarios incl.
///   `sv_no_weak_hook 1` — F2's evidence, `gr_*` grenade-owner-death scenarios — F4's evidence,
///   `sp_*` speedway/speedup scenarios (the `Force == 255`/`TeeSpeed` sub-cases) — F5's evidence,
///   `cfg_*` a `sv_solo_server`-tuned brawl variant.
/// - Round 2: `r2-st/speedtune_*` (`tune velramp_range`/`velramp_curvature` overrides exercising
///   `MaxRampSpeed`'s `log`-in-`f64` sub-case — F5 round 2's own evidence,
///   `speedtune_v0.trb` specifically), `r2-adv/*` (75 adversarial scenarios: `sv_no_weak_hook`
///   with hooks and death tiles, grenades with `sv_destroy_bullets_on_death` 0/1 and teams,
///   team-empty switch reset and forced solo, draggers on mover tiles, airborne old-type
///   speedups).
///
/// Copied out of the reviewer's own scratchpad into
/// `~/aiddnet/data/traces/oracle-b/reviewer-crafted/` (outside the repository, like the main
/// corpus — never committed) so this test survives that scratchpad being cleaned up.
/// `#[ignore]`d like the corpus tests above (needs this exact local directory, not something a
/// fresh checkout has).
#[test]
#[ignore]
fn replays_opus_reviewer_crafted_regression_traces() {
    let home = std::env::var("HOME").expect("HOME must be set");
    let dir = PathBuf::from(home).join("aiddnet/data/traces/oracle-b/reviewer-crafted");
    if !dir.is_dir() {
        eprintln!("skipping: {} not found", dir.display());
        return;
    }
    let mut entries: Vec<PathBuf> = Vec::new();
    for sub in ["ks", "br", "gr", "sp", "cfg", "r2-st", "r2-adv"] {
        let subdir = dir.join(sub);
        if !subdir.is_dir() {
            continue;
        }
        for e in std::fs::read_dir(&subdir).unwrap().filter_map(|e| e.ok()) {
            let p = e.path();
            if p.extension().is_some_and(|e| e == "trb") {
                entries.push(p);
            }
        }
    }
    entries.sort();
    assert!(!entries.is_empty(), "no .trb traces found under {}", dir.display());
    run_corpus_subset(&entries);
}
