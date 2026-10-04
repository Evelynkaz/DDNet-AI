//! Task 1.6 stage B: behaviour tests for the entities stage B ported (`CLaser` rifle/shotgun,
//! `CDragger`/`CDraggerBeam`, `CGun`/`CPlasma`, `CLight`, ninja), on tiny hand-built maps. The
//! bit-exact evidence is `tests/parity_oracle_b.rs` (the Oracle B corpora); these tests pin the
//! *semantics* each port step is supposed to have — written from the C++ source (cited per test),
//! not from this crate's output — so a regression is explained by a failing sentence rather than
//! by a corpus diff, and so CI (which has no corpus) still exercises every stage-B code path.
//!
//! The no-allocation test at the bottom drives all of them at once and asserts `World::step`
//! never touches the heap in steady state (acceptance criterion "Keep the allocation-free hot
//! path of stage A").

use allocation_counter::measure;
use ddai_physics::core::{PlayerInput, WEAPON_GRENADE, WEAPON_LASER, WEAPON_NINJA, WEAPON_SHOTGUN};
use ddai_physics::map::{
    self, ENTITY_OFFSET, MapData, SwitchTile, TILE_FREEZE, TILE_SOLID, TILE_TELEINWEAPON, TILE_TELEOUT, TeleTile, Tile,
    TuneTile,
};
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Fixture, LaserSlot, Player, TickInput, World};

const T: i32 = 32;

/// A `w` x `h` tile room with a solid border (everything else air).
fn room(w: u32, h: u32) -> MapData {
    let mut game = vec![Tile::default(); (w * h) as usize];
    for x in 0..w {
        game[x as usize].index = TILE_SOLID;
        game[((h - 1) * w + x) as usize].index = TILE_SOLID;
    }
    for y in 0..h {
        game[(y * w) as usize].index = TILE_SOLID;
        game[(y * w + w - 1) as usize].index = TILE_SOLID;
    }
    MapData {
        width: w,
        height: h,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

fn set_tile(map: &mut MapData, x: u32, y: u32, index: u8) {
    map.game[(y * map.width + x) as usize].index = index;
}

fn set_entity(map: &mut MapData, x: u32, y: u32, entity: u8) {
    set_tile(map, x, y, ENTITY_OFFSET + entity);
}

fn set_switch_entity(map: &mut MapData, x: u32, y: u32, number: u8, entity: u8) {
    let n = (map.width * map.height) as usize;
    let sw = map.switch.get_or_insert_with(|| vec![SwitchTile::default(); n]);
    sw[(y * map.width + x) as usize] = SwitchTile {
        number,
        kind: ENTITY_OFFSET + entity,
        flags: 0,
        delay: 0,
    };
}

fn set_tele(map: &mut MapData, x: u32, y: u32, number: u8, kind: u8) {
    let n = (map.width * map.height) as usize;
    let t = map.tele.get_or_insert_with(|| vec![TeleTile::default(); n]);
    t[(y * map.width + x) as usize] = TeleTile { number, kind };
}

fn new_world(map: &MapData) -> World<f32> {
    let mut world: World<f32> = World::from_map(map, 1);
    world.init(std::iter::empty::<&str>()).unwrap();
    world
}

/// Spawns client `id` standing on the floor of the room at pixel column `x`: the tee is 28 px tall,
/// so its centre rests 15 px (not 14: the box's bottom edge must stay on the last air pixel row,
/// `TestBox` counts the first solid row) above the floor tile row's top edge.
fn spawn_on_floor(world: &mut World<f32>, id: i32, x: i32, floor_row: i32) {
    world.players[id as usize] = Some(Player::new(0));
    world::spawn_character(world, id, Vec2::new(x as f32, (floor_row * T - 15) as f32));
}

fn set_active_weapon(world: &mut World<f32>, id: i32, weapon: i32) {
    world::give_weapon_to(world, id, weapon);
    let slot = world.cores.slot_of(id as u8).unwrap();
    world.cores.core_at_mut(slot).active_weapon = weapon;
}

fn idle(id: u8) -> TickInput {
    TickInput {
        id,
        input: PlayerInput {
            target_x: 100,
            ..Default::default()
        },
        kill: false,
    }
}

/// Steps once with `id` holding `fire` (the wire counter, odd = pressed) aimed at `(tx, ty)` and
/// everybody else idle.
fn step_fire(world: &mut World<f32>, ids: &[u8], shooter: u8, fire: i32, tx: i32, ty: i32) {
    let inputs: Vec<TickInput> = ids
        .iter()
        .map(|&id| {
            if id == shooter {
                TickInput {
                    id,
                    input: PlayerInput {
                        target_x: tx,
                        target_y: ty,
                        fire,
                        ..Default::default()
                    },
                    kill: false,
                }
            } else {
                idle(id)
            }
        })
        .collect();
    world.step(&inputs);
}

fn step_idle(world: &mut World<f32>, ids: &[u8], n: usize) {
    let inputs: Vec<TickInput> = ids.iter().map(|&id| idle(id)).collect();
    for _ in 0..n {
        world.step(&inputs);
    }
}

fn freeze(world: &mut World<f32>, id: i32) {
    let slot = world.cores.slot_of(id as u8).unwrap();
    let mut ch = world.characters[id as usize].unwrap();
    let mut core = *world.cores.core_at(slot);
    assert!(world::freeze_default(&mut ch, &mut core, world.tick, 3));
    world.characters[id as usize] = Some(ch);
    *world.cores.core_at_mut(slot) = core;
}

fn lasers(world: &World<f32>) -> Vec<world::Laser<f32>> {
    world
        .lasers
        .iter()
        .filter_map(|l| match l {
            LaserSlot::Laser(l) => Some(*l),
            _ => None,
        })
        .collect()
}

// --- CLaser ------------------------------------------------------------------------------------

/// `CLaser::HitCharacter` (`laser.cpp:96-98`): a rifle hit unfreezes the target — in the very tick
/// the shot is fired (`CLaser`'s constructor runs the first `DoBounce`, `laser.cpp:42`).
#[test]
fn rifle_shot_unfreezes_a_frozen_tee_in_the_line_of_fire() {
    let map = room(30, 12);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 100, 11);
    spawn_on_floor(&mut world, 1, 400, 11);
    set_active_weapon(&mut world, 0, WEAPON_LASER);
    step_idle(&mut world, &[0, 1], 2);
    freeze(&mut world, 1);
    assert!(world.characters[1].unwrap().freeze_time > 0);

    step_fire(&mut world, &[0, 1], 0, 1, 100, 0);
    assert_eq!(
        world.characters[1].unwrap().freeze_time,
        0,
        "the rifle must unfreeze the tee it hits"
    );
    let ls = lasers(&world);
    assert_eq!(
        ls.len(),
        1,
        "the laser entity lives on (it is only destroyed by a later DoBounce)"
    );
    assert!(ls[0].energy < 0.0, "a hit sets m_Energy = -1");
}

/// `CLaser::HitCharacter` (`laser.cpp:64-78`): a shotgun hit adds `normalize(m_PrevPos - HitPos) *
/// ShotgunStrength` to the target's velocity — toward where the shot started.
#[test]
fn shotgun_pulls_the_hit_tee_toward_the_shooter() {
    let map = room(30, 12);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 100, 11);
    spawn_on_floor(&mut world, 1, 400, 11);
    set_active_weapon(&mut world, 0, WEAPON_SHOTGUN);
    step_idle(&mut world, &[0, 1], 2);
    let before = world.cores.get(1).unwrap().vel.x;
    step_fire(&mut world, &[0, 1], 0, 1, 100, 0);
    let after = world.cores.get(1).unwrap().vel.x;
    // ShotgunStrength defaults to 10; one tick of ground friction (0.5) follows the hit.
    assert!(
        before - after > 3.0,
        "vel.x went {before} -> {after}, expected a pull toward -x"
    );
}

/// A shot that misses everything reflects off the far wall (`laser.cpp:130-186`): one bounce, the
/// energy reduced by the segment length, and a second segment only after `laser_bounce_delay`.
#[test]
fn a_laser_bounces_off_a_wall_and_continues_after_the_bounce_delay() {
    let map = room(30, 12);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 300, 11);
    set_active_weapon(&mut world, 0, WEAPON_LASER);
    step_idle(&mut world, &[0], 2);
    step_fire(&mut world, &[0], 0, 1, 100, 0);
    step_fire(&mut world, &[0], 0, 2, 100, 0);
    let ls = lasers(&world);
    assert_eq!(ls.len(), 1);
    let l = ls[0];
    assert_eq!(l.bounces, 1, "one bounce off the right wall");
    assert!(l.dir.x < 0.0, "reflected back toward the shooter");
    // First segment: from x = 300 to the last air pixel before the right wall (tile column 29 starts
    // at x = 928, so about 627 px); `m_Energy -= Distance + LaserBounceCost (0)` leaves roughly
    // 800 - 627 = 173 for the next segment.
    assert!((l.energy - 173.0).abs() < 3.0, "energy {}", l.energy);
    let first_eval = l.eval_tick;
    // Created at tick 2; the world is at tick 4 now. `CLaser::Tick` bounces again once `Tick -
    // m_EvalTick > TickSpeed * 150 / 1000 = 7.5`, i.e. at tick 10, not at tick 9.
    step_idle(&mut world, &[0], 5);
    assert_eq!(
        lasers(&world)[0].eval_tick,
        first_eval,
        "no new segment inside the 150 ms delay"
    );
    step_idle(&mut world, &[0], 2);
    let l2 = lasers(&world)[0];
    assert!(
        l2.eval_tick > first_eval,
        "the second segment starts after laser_bounce_delay"
    );
}

/// `IntersectLineTeleWeapon` + `laser.cpp:167-172`: a laser that crosses a `TILE_TELEINWEAPON` tile
/// with a matching `TELEOUT` is teleported there, without counting a bounce.
#[test]
fn a_laser_through_a_tele_in_weapon_tile_continues_from_the_tele_out() {
    let mut map = room(30, 12);
    set_tele(&mut map, 10, 10, 1, TILE_TELEINWEAPON);
    set_tele(&mut map, 20, 3, 1, TILE_TELEOUT);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 100, 11);
    set_active_weapon(&mut world, 0, WEAPON_LASER);
    step_idle(&mut world, &[0], 2);
    // Aim straight along the row the tele-in tile sits in: the tee stands at y = 11 * 32 - 15 = 337
    // and the tile (row 10) spans y 320..352, so a horizontal shot crosses it.
    step_fire(&mut world, &[0], 0, 1, 100, 0);
    let l = lasers(&world)[0];
    assert!(l.was_tele, "the shot must register the teleport");
    assert_eq!(l.tele_pos, Vec2::new(20.0 * 32.0 + 16.0, 3.0 * 32.0 + 16.0));
    assert_eq!(l.bounces, 0, "a teleport is not a bounce");
}

/// `CLaser::Tick` (`laser.cpp:265-272`): with `sv_destroy_lasers_on_death` a laser whose owner is
/// no longer alive is destroyed, with the default (off) it keeps flying.
#[test]
fn lasers_outlive_their_owner_unless_sv_destroy_lasers_on_death() {
    for destroy in [false, true] {
        let map = room(30, 12);
        let mut world = new_world(&map);
        world.config.sv_destroy_lasers_on_death = destroy;
        world.players[0] = Some(Player::new(0));
        std::sync::Arc::make_mut(&mut world.spawn_points).push(Vec2::new(200.0, 200.0));
        spawn_on_floor(&mut world, 0, 100, 11);
        set_active_weapon(&mut world, 0, WEAPON_LASER);
        step_idle(&mut world, &[0], 2);
        step_fire(&mut world, &[0], 0, 1, 100, 0);
        assert_eq!(lasers(&world).len(), 1);
        // Kill the owner (the kill bit, like a client's `/kill`).
        world.step(&[TickInput {
            id: 0,
            input: PlayerInput::default(),
            kill: true,
        }]);
        step_idle(&mut world, &[0], 1);
        assert_eq!(
            lasers(&world).len(),
            usize::from(!destroy),
            "sv_destroy_lasers_on_death={destroy}"
        );
    }
}

/// `CGameWorld::RemoveEntitiesFromPlayer` (`teams.cpp:497`): changing a tee's DDRace team removes
/// the lasers it owns.
#[test]
fn a_team_change_removes_the_owners_lasers() {
    let map = room(30, 12);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 100, 11);
    set_active_weapon(&mut world, 0, WEAPON_LASER);
    step_idle(&mut world, &[0], 2);
    step_fire(&mut world, &[0], 0, 1, 100, 0);
    assert_eq!(lasers(&world).len(), 1);
    world::set_force_character_team(&mut world, 0, 5);
    step_idle(&mut world, &[0], 1);
    assert!(lasers(&world).is_empty());
}

/// `RemoveEntitiesFromPlayer` removes the laser *at once* (review 1.6b F2): an owner in team 3 who is
/// killed right before the laser's next segment (death moves it to team 0, which makes it collidable
/// with the team-0 victim; a dead owner's laser also has `NoHitOthers = sv_hit = 0`) must not see that
/// segment run in the same step. Before the fix the marked laser still bounced and pulled the victim.
#[test]
fn a_laser_removed_by_the_owners_team_change_does_not_tick_in_the_same_step() {
    let mut map = room(30, 12);
    map.settings.push("sv_hit 0".to_string());
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 700, 11);
    spawn_on_floor(&mut world, 1, 400, 11);
    world::set_force_character_team(&mut world, 0, 3);
    set_active_weapon(&mut world, 0, WEAPON_SHOTGUN);
    step_idle(&mut world, &[0, 1], 2);
    step_fire(&mut world, &[0, 1], 0, 1, 100, 0); // laser created at tick 2, next segment at tick 10
    step_fire(&mut world, &[0, 1], 0, 2, 100, 0);
    step_idle(&mut world, &[0, 1], 5);
    assert_eq!(world.tick, 9);
    assert_eq!(lasers(&world).len(), 1);
    world.step(&[
        TickInput {
            id: 0,
            input: PlayerInput {
                target_x: 100,
                ..Default::default()
            },
            kill: true,
        },
        idle(1),
    ]);
    assert!(
        lasers(&world).is_empty(),
        "the owner's death removes its lasers immediately"
    );
    assert_eq!(
        world.cores.get(1).unwrap().vel.x,
        0.0,
        "the removed laser's bounce must not run"
    );
}

/// `TryRespawn` sets `m_ViewPos = SpawnPos` and `CPlayer::Tick` recomputes `m_TuneZone` from it in the same
/// call (review 1.6b F3): the player's tune zone is the spawn zone right after the respawn.
#[test]
fn a_respawn_moves_the_players_tune_zone_to_the_spawn_zone_at_once() {
    let mut map = room(20, 12);
    set_entity(&mut map, 10, 8, map::ENTITY_SPAWN);
    let mut tune = vec![TuneTile::default(); (map.width * map.height) as usize];
    tune[(8 * map.width + 10) as usize] = TuneTile { number: 5, kind: 1 };
    map.tune = Some(tune);
    let mut world = new_world(&map);
    world.players[0] = Some(Player::new(0));
    assert_eq!(world.players[0].as_ref().unwrap().tune_zone, 0);
    assert!(world::try_respawn(&mut world, 0));
    assert_eq!(world.players[0].as_ref().unwrap().tune_zone, 5);
}

/// `CInteractions::CanHit` (`interactions.cpp:99-114`): a laser fired by a tee in DDRace team 5
/// cannot hit a tee in team 0 (and `CanCollide` already hides it from the intersection).
#[test]
fn a_laser_does_not_hit_a_tee_of_another_team() {
    let map = room(30, 12);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 100, 11);
    spawn_on_floor(&mut world, 1, 400, 11);
    world::set_force_character_team(&mut world, 0, 5);
    set_active_weapon(&mut world, 0, WEAPON_LASER);
    step_idle(&mut world, &[0, 1], 2);
    freeze(&mut world, 1);
    step_fire(&mut world, &[0, 1], 0, 1, 100, 0);
    assert!(
        world.characters[1].unwrap().freeze_time > 0,
        "the other team's tee stays frozen"
    );
}

/// `sv_old_laser` (`laser.cpp:52,317`): the shot can hit its own shooter right after it left.
#[test]
fn stage_b_config_variables_are_recognized_and_clamped() {
    let map = room(12, 8);
    let mut world = new_world(&map);
    // After `init` the CFGFLAG_GAME variables are locked, so apply them as pre-init lines.
    let mut fresh: World<f32> = World::from_map(&map, 1);
    fresh
        .init([
            "sv_dragger_range 123456",
            "sv_plasma_range 0",
            "sv_plasma_per_sec 99",
            "sv_destroy_lasers_on_death 1",
        ])
        .unwrap();
    // The `sv_ddrace_tune_reset` block does not touch these four, so they survive `init`.
    assert_eq!(fresh.config.sv_dragger_range, 99999, "clamped to the variable's max");
    assert_eq!(fresh.config.sv_plasma_range, 1, "clamped to the variable's min");
    assert_eq!(fresh.config.sv_plasma_per_sec, 50);
    assert!(fresh.config.sv_destroy_lasers_on_death);
    assert!(world.apply_commands(["sv_plasma_per_sec 5"]).is_ok());
    assert_eq!(
        world.config.sv_plasma_per_sec, 3,
        "locked after init, like every CFGFLAG_GAME variable"
    );
}

// --- ninja -------------------------------------------------------------------------------------

/// `FireWeapon`'s `WEAPON_NINJA` case + `HandleNinja` (`character.cpp:296-394,630-642`): firing
/// starts a 10-tick dash at velocity 50 along the aim, then the old speed comes back.
#[test]
fn ninja_dashes_ten_ticks_and_hits_tees_on_the_way() {
    let map = room(40, 12);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 100, 11);
    spawn_on_floor(&mut world, 1, 300, 11);
    {
        let slot = world.cores.slot_of(0).unwrap();
        let mut ch = world.characters[0].unwrap();
        let mut core = *world.cores.core_at(slot);
        world::give_ninja(&mut ch, &mut core, world.tick);
        world.characters[0] = Some(ch);
        *world.cores.core_at_mut(slot) = core;
    }
    assert_eq!(world.cores.get(0).unwrap().active_weapon, WEAPON_NINJA);
    step_idle(&mut world, &[0, 1], 2);
    let start_x = world.cores.get(0).unwrap().pos.x;
    step_fire(&mut world, &[0, 1], 0, 1, 100, 0);
    assert_eq!(
        world.cores.get(0).unwrap().ninja.current_move_time,
        9,
        "10 ticks, one already spent"
    );
    let vy_before = world.cores.get(1).unwrap().vel.y;
    step_fire(&mut world, &[0, 1], 0, 2, 100, 0);
    for _ in 0..8 {
        step_fire(&mut world, &[0, 1], 0, 2, 100, 0);
    }
    let moved = world.cores.get(0).unwrap().pos.x - start_x;
    assert!(moved > 150.0, "the dash must cover ground: moved {moved}");
    // Tee 1 stood 200 px away, inside the dash path (radius 56 around each step's start): it must
    // have been hit exactly once — `TakeDamage((0, -10))` pushes it upward.
    assert!(
        world.cores.get(1).unwrap().vel.y < vy_before - 5.0 || world.characters[1].unwrap().pos.y < 11.0 * 32.0 - 14.0,
        "ninja hit should have kicked tee 1 upward"
    );
    assert_eq!(world.characters[0].unwrap().num_objects_hit, 1);
}

/// `CCharacter::RemoveNinja` (`character.cpp:691-702`): the weapon before the ninja comes back,
/// and `m_LastWeapon` is left alone (`SetWeapon(W == active)` returns at once).
#[test]
fn ninja_expiry_restores_the_previous_weapon_without_touching_last_weapon() {
    let map = room(20, 8);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 100, 7);
    set_active_weapon(&mut world, 0, WEAPON_GRENADE);
    {
        let slot = world.cores.slot_of(0).unwrap();
        let mut ch = world.characters[0].unwrap();
        let mut core = *world.cores.core_at(slot);
        world::give_ninja(&mut ch, &mut core, world.tick);
        assert_eq!(ch.last_weapon, WEAPON_GRENADE);
        world.characters[0] = Some(ch);
        *world.cores.core_at_mut(slot) = core;
    }
    // 15 s * 50 ticks + 1: the tick after the duration, `HandleNinja` removes it.
    step_idle(&mut world, &[0], 752);
    let core = world.cores.get(0).unwrap();
    assert_eq!(core.active_weapon, WEAPON_GRENADE);
    assert!(!core.weapons[WEAPON_NINJA as usize].got);
    assert_eq!(world.characters[0].unwrap().last_weapon, WEAPON_GRENADE);
}

// --- CLight ------------------------------------------------------------------------------------

/// `CLight` (`light.cpp`): a static `ENTITY_LASER_STOP` light with a length marker freezes a tee
/// standing in its beam — from the first `Step()` on (tick 7, `m_To` is the empty segment before).
#[test]
fn a_static_light_freezes_a_tee_in_its_beam_once_it_has_stepped() {
    let mut map = room(30, 12);
    // Light source at (5, 10) with a long beam east (direction index 2: E), marker at (6, 10).
    set_entity(&mut map, 5, 10, map::ENTITY_LASER_STOP);
    set_entity(&mut map, 6, 10, map::ENTITY_LASER_LONG);
    let mut world = new_world(&map);
    assert!(matches!(world.fixtures[0], Fixture::Light(_)));
    world.players[0] = Some(Player::new(0));
    // Stand in the beam: x between 6.x and 15 tiles, y on the row of the source (10 * 32 + 16).
    world::spawn_character(&mut world, 0, Vec2::new(11.0 * 32.0, 10.0 * 32.0 + 16.0));
    step_idle(&mut world, &[0], 6);
    assert_eq!(
        world.characters[0].unwrap().freeze_time,
        0,
        "m_To == m_Pos until the first Step()"
    );
    step_idle(&mut world, &[0], 2);
    assert!(
        world.characters[0].unwrap().freeze_time > 0,
        "the beam is live after tick 7"
    );
}

/// A light behind a switch (`m_Layer == LAYER_SWITCH && m_Number > 0`) only acts while that switch
/// is on for the tee's team (`light.cpp:39`).
#[test]
fn a_switch_gated_light_ignores_tees_while_its_switch_is_off() {
    let mut map = room(30, 12);
    set_switch_entity(&mut map, 5, 10, 3, map::ENTITY_LASER_STOP);
    set_switch_entity(&mut map, 6, 10, 3, map::ENTITY_LASER_LONG);
    let mut world = new_world(&map);
    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, Vec2::new(11.0 * 32.0, 10.0 * 32.0 + 16.0));
    // Close the switch for team 0.
    world.cores.switchers[3].status[0] = false;
    step_idle(&mut world, &[0], 20);
    assert_eq!(world.characters[0].unwrap().freeze_time, 0);
    world.cores.switchers[3].status[0] = true;
    step_idle(&mut world, &[0], 2);
    assert!(world.characters[0].unwrap().freeze_time > 0);
}

// --- CDragger / CDraggerBeam -------------------------------------------------------------------

/// `CDragger::LookForPlayersToDrag` + `CDraggerBeam::Tick` (`dragger.cpp`, `dragger_beam.cpp`): a
/// dragger within range and line of sight creates one beam per team for the closest tee and the
/// beam adds `strength` toward the dragger each tick.
#[test]
fn a_dragger_pulls_the_closest_tee_toward_itself() {
    let mut map = room(30, 12);
    set_entity(&mut map, 3, 4, map::ENTITY_DRAGGER_STRONG);
    let mut world = new_world(&map);
    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, Vec2::new(12.0 * 32.0, 4.0 * 32.0 + 16.0));
    // `LookForPlayersToDrag` runs on tick % 7 == 0.
    step_idle(&mut world, &[0], 8);
    let beams: Vec<_> = world
        .lasers
        .iter()
        .filter_map(|l| match l {
            LaserSlot::Beam(b) => Some(*b),
            _ => None,
        })
        .collect();
    assert_eq!(beams.len(), 1);
    assert_eq!(beams[0].for_client, 0);
    assert!(beams[0].active);
    let x0 = world.cores.get(0).unwrap().pos.x;
    step_idle(&mut world, &[0], 10);
    let x1 = world.cores.get(0).unwrap().pos.x;
    assert!(
        x1 < x0 - 3.0,
        "the strong dragger (3 units/tick) must have pulled the tee left: {x0} -> {x1}"
    );
}

/// A solid wall blocks a normal dragger (`IntersectNoLaser`), an `_NW` (ignore-walls) one
/// ignores it (`IntersectNoLaserNoWalls`, only `TILE_NOLASER` blocks).
#[test]
fn dragger_beams_respect_walls_unless_ignore_walls() {
    for (entity, expect_beam) in [
        (map::ENTITY_DRAGGER_NORMAL, false),
        (map::ENTITY_DRAGGER_NORMAL_NW, true),
    ] {
        let mut map = room(30, 12);
        set_entity(&mut map, 3, 4, entity);
        for y in 1..11 {
            set_tile(&mut map, 8, y, TILE_SOLID);
        }
        let mut world = new_world(&map);
        world.players[0] = Some(Player::new(0));
        world::spawn_character(&mut world, 0, Vec2::new(12.0 * 32.0, 4.0 * 32.0 + 16.0));
        step_idle(&mut world, &[0], 8);
        let n = world.lasers.iter().filter(|l| matches!(l, LaserSlot::Beam(_))).count();
        assert_eq!(n, usize::from(expect_beam), "entity {entity}");
    }
}

/// `CDraggerBeam::Tick` (`dragger_beam.cpp:62-70`): the beam dissolves when the tee leaves the
/// range (`sv_dragger_range`, default 700) — and the dragger may then make a new one.
#[test]
fn a_dragger_beam_dissolves_when_the_tee_leaves_the_range() {
    let mut map = room(60, 12);
    set_entity(&mut map, 3, 4, map::ENTITY_DRAGGER_WEAK);
    let mut world = new_world(&map);
    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, Vec2::new(10.0 * 32.0, 4.0 * 32.0 + 16.0));
    step_idle(&mut world, &[0], 8);
    assert_eq!(
        world.lasers.iter().filter(|l| matches!(l, LaserSlot::Beam(_))).count(),
        1
    );
    // Teleport the tee far away (a harness move, like a tele tile would do).
    let slot = world.cores.slot_of(0).unwrap();
    world.cores.core_at_mut(slot).pos = Vec2::new(55.0 * 32.0, 10.0 * 32.0);
    step_idle(&mut world, &[0], 2);
    assert_eq!(
        world.lasers.iter().filter(|l| matches!(l, LaserSlot::Beam(_))).count(),
        0
    );
}

// --- CGun / CPlasma ----------------------------------------------------------------------------

/// `CGun::Fire` + `CPlasma` (`gun.cpp`, `plasma.cpp`): a freezing turret shoots the closest tee
/// in range at `sv_plasma_per_sec` and the shot freezes it on contact.
#[test]
fn a_freeze_turret_shoots_a_tee_in_range_and_the_plasma_freezes_it() {
    let mut map = room(30, 12);
    set_entity(&mut map, 3, 5, map::ENTITY_PLASMAF);
    let mut world = new_world(&map);
    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, Vec2::new(12.0 * 32.0, 5.0 * 32.0 + 16.0));
    let mut saw_plasma = false;
    let mut frozen_at = None;
    for t in 0..80 {
        step_idle(&mut world, &[0], 1);
        saw_plasma |= world.lasers.iter().any(|l| matches!(l, LaserSlot::Plasma(_)));
        if frozen_at.is_none() && world.characters[0].unwrap().freeze_time > 0 {
            frozen_at = Some(t);
        }
    }
    assert!(saw_plasma, "the turret must have fired");
    assert!(frozen_at.is_some(), "the plasma must have frozen the tee");
}

/// `sv_plasma_per_sec 0` disables every turret (`gun.cpp:41`).
#[test]
fn a_turret_does_not_fire_with_sv_plasma_per_sec_zero() {
    let mut map = room(30, 12);
    set_entity(&mut map, 3, 5, map::ENTITY_PLASMAF);
    let mut world: World<f32> = World::from_map(&map, 1);
    world.init(["sv_plasma_per_sec 0"]).unwrap();
    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, Vec2::new(12.0 * 32.0, 5.0 * 32.0 + 16.0));
    for _ in 0..60 {
        step_idle(&mut world, &[0], 1);
        assert!(world.lasers.is_empty());
    }
    assert_eq!(world.characters[0].unwrap().freeze_time, 0);
}

// --- bookkeeping -------------------------------------------------------------------------------

/// `World::restore_from` must reproduce lasers, beams, plasma and fixture state exactly: a restored
/// world and the original step in lock-step afterwards.
#[test]
fn restore_from_reproduces_stage_b_state() {
    let mut map = room(40, 14);
    set_entity(&mut map, 3, 5, map::ENTITY_PLASMA);
    set_entity(&mut map, 36, 5, map::ENTITY_DRAGGER_NORMAL);
    set_entity(&mut map, 20, 2, map::ENTITY_LASER_NORMAL_CW);
    set_entity(&mut map, 21, 2, map::ENTITY_LASER_MEDIUM);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 400, 13);
    spawn_on_floor(&mut world, 1, 700, 13);
    set_active_weapon(&mut world, 0, WEAPON_LASER);
    step_idle(&mut world, &[0, 1], 2);
    step_fire(&mut world, &[0, 1], 0, 1, 300, -50);
    step_idle(&mut world, &[0, 1], 25);
    assert!(!world.lasers.is_empty());

    let mut copy = world.clone();
    let mut restored = new_world(&map);
    restored.restore_from(&world);
    for _ in 0..40 {
        step_idle(&mut world, &[0, 1], 1);
        step_idle(&mut copy, &[0, 1], 1);
        step_idle(&mut restored, &[0, 1], 1);
        assert_eq!(world.lasers, copy.lasers);
        assert_eq!(world.lasers, restored.lasers);
        assert_eq!(world.fixtures, restored.fixtures);
        assert_eq!(world.cores.get(0).unwrap().pos, restored.cores.get(0).unwrap().pos);
        assert_eq!(world.cores.get(1).unwrap().vel, restored.cores.get(1).unwrap().vel);
    }
}

/// The allocation-free hot path (`tests/world_no_alloc.rs`'s contract) with every stage-B entity
/// active at once: a rifle/shotgun tee firing at a wall, a ninja dashing, a turret, a dragger and
/// a rotating light. `World::step` must not allocate in steady state.
#[test]
fn world_step_performs_zero_heap_allocations_with_every_stage_b_entity_active() {
    let mut map = room(60, 16);
    set_entity(&mut map, 6, 6, map::ENTITY_PLASMA);
    set_entity(&mut map, 50, 6, map::ENTITY_PLASMAE);
    set_entity(&mut map, 30, 3, map::ENTITY_DRAGGER_STRONG);
    set_entity(&mut map, 30, 8, map::ENTITY_LASER_NORMAL_CCW);
    set_entity(&mut map, 31, 8, map::ENTITY_LASER_LONG);
    set_entity(&mut map, 33, 8, map::ENTITY_LASER_C_NORMAL);
    set_tile(&mut map, 25, 14, TILE_FREEZE);
    let mut world = new_world(&map);
    spawn_on_floor(&mut world, 0, 400, 15);
    spawn_on_floor(&mut world, 1, 900, 15);
    spawn_on_floor(&mut world, 2, 1300, 15);
    set_active_weapon(&mut world, 0, WEAPON_LASER);
    world::give_weapon_to(&mut world, 0, WEAPON_SHOTGUN);
    world::give_weapon_to(&mut world, 1, WEAPON_SHOTGUN);
    let slot = world.cores.slot_of(2).unwrap();
    {
        let mut ch = world.characters[2].unwrap();
        let mut core = *world.cores.core_at(slot);
        world::give_ninja(&mut ch, &mut core, 0);
        world.characters[2] = Some(ch);
        *world.cores.core_at_mut(slot) = core;
    }

    let input = |tick: u32, id: u8| -> TickInput {
        let phase = tick % 90;
        TickInput {
            id,
            input: PlayerInput {
                direction: if (phase / 30).is_multiple_of(2) { 1 } else { -1 },
                target_x: if id == 1 { -300 } else { 300 },
                target_y: -((phase % 7) as i32) * 40,
                fire: i32::from(phase % 4 < 2) + 2 * (tick as i32 % 20),
                jump: i32::from(phase == 10),
                wanted_weapon: if phase == 0 { 2 + i32::from(id == 0) } else { 0 },
                ..Default::default()
            },
            kill: false,
        }
    };
    let step = |world: &mut World<f32>, tick: u32| {
        world.step(&[input(tick, 0), input(tick, 1), input(tick, 2)]);
    };
    for tick in 0..200 {
        step(&mut world, tick);
    }
    let info = measure(|| {
        for tick in 200..700 {
            step(&mut world, tick);
        }
    });
    assert_eq!(info.count_total, 0, "World::step allocated: {info:?}");
}
