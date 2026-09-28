//! Task 1.6, Stage A: end-to-end smoke tests for `World<R>`. Build a `World` on a synthetic
//! recipe (`ddai_trace::synthetic`, task 1.2), spawn characters, run many ticks with varied
//! input, and check for panics/NaN — the same "does it run at all" bar `core_world`'s own tests
//! hold task 1.3's code to, before any bit-exactness claim. Real corpus parity is
//! `tests/parity_oracle_b.rs`'s job. Lives here (an integration test, not a `src/world.rs` unit
//! test) because it needs `ddai_trace::synthetic`, and `ddai-trace` itself depends on
//! `ddai-physics` — a unit test inside `ddai-physics`'s own lib target would see two distinct
//! (test-cfg vs normal) compilations of `ddai_physics::map::MapData`, which Rust treats as
//! different types.

use ddai_physics::core::PlayerInput;
use ddai_physics::real::Real;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Player, TickInput, World};
use ddai_trace::synthetic;

fn spawn_two<R: Real>(world: &mut World<R>) {
    let a = world
        .spawn_points
        .first()
        .copied()
        .unwrap_or(Vec2::new(R::from_i32(200), R::from_i32(200)));
    let b = world
        .spawn_points
        .get(1)
        .copied()
        .unwrap_or(Vec2::new(R::from_i32(260), R::from_i32(200)));
    world.players[0] = Some(Player::new(0));
    world.players[1] = Some(Player::new(0));
    world::spawn_character(world, 0, a);
    world::spawn_character(world, 1, b);
}

#[test]
fn runs_many_ticks_on_freeze_recipe_without_panicking_or_nan() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 12345);
    spawn_two(&mut world);
    // `sv_no_weak_hook` is a `CFGFLAG_GAME` config variable `World::step` reads itself now (not
    // a per-call argument — see its own doc comment), so exercise its code path by setting it
    // once, up front, rather than toggling it mid-run (which real DDNet can never do either).
    world.config.sv_no_weak_hook = true;

    for tick in 0..2000u32 {
        let phase = tick % 97;
        let inputs = [
            TickInput {
                id: 0,
                input: PlayerInput {
                    direction: if phase < 40 { 1 } else { -1 },
                    target_x: 1,
                    target_y: -1,
                    jump: i32::from(phase == 0),
                    hook: i32::from(phase < 30),
                    fire: i32::from(phase % 5 == 0),
                    ..Default::default()
                },
                kill: false,
            },
            TickInput {
                id: 1,
                input: PlayerInput {
                    direction: if phase < 50 { -1 } else { 1 },
                    target_x: -1,
                    target_y: -1,
                    jump: i32::from(phase == 20),
                    hook: i32::from((30..60).contains(&phase)),
                    fire: i32::from(phase % 7 == 0),
                    ..Default::default()
                },
                kill: tick == 500,
            },
        ];
        world.step(&inputs);

        for (id, core) in world.cores.iter() {
            assert!(
                !core.pos.x.is_nan() && !core.pos.y.is_nan(),
                "NaN position for id {id} at tick {tick}"
            );
            assert!(
                !core.vel.x.is_nan() && !core.vel.y.is_nan(),
                "NaN velocity for id {id} at tick {tick}"
            );
        }
    }
}

#[test]
fn f64_instantiation_runs_without_panicking() {
    let map = synthetic::build("tele-speedup").expect("tele-speedup recipe must exist");
    let mut world: World<f64> = World::from_map(&map, 999);
    let a = world.spawn_points.first().copied().unwrap_or(Vec2::new(200.0, 200.0));
    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, a);

    for tick in 0..500u32 {
        let inputs = [TickInput {
            id: 0,
            input: PlayerInput {
                direction: 1,
                target_x: 1,
                target_y: 0,
                hook: i32::from(tick % 10 < 5),
                ..Default::default()
            },
            kill: false,
        }];
        world.step(&inputs);
    }
    let core = world.cores.get(0).unwrap();
    assert!(!core.pos.x.is_nan());
}

#[test]
fn kill_bit_respawns_within_a_few_ticks() {
    let map = synthetic::build("arena").expect("arena recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 7);
    // The synthetic recipes (task 1.2) are Oracle-A-only maps: characters get their positions
    // straight from the scenario, never from an `ENTITY_SPAWN` map tile, so `World::from_map`
    // finds no spawn points here. Populate `spawn_points` directly (a `pub` field) so
    // `try_respawn`'s `CanSpawn` has somewhere to put a respawning character — a real map always
    // has at least one `ENTITY_SPAWN` tile (`docs/formats.md`'s corpus notes).
    world.spawn_points.push(Vec2::new(200.0, 200.0));
    spawn_two(&mut world);

    // Kill id 1 on tick 0 (input applied on the tick-1 step); it holds fire the whole time so
    // the very next tick's early-input spawn trigger fires immediately once the character is
    // gone (`docs/formats.md`/this crate's `BUILD REPORT`, `Player::has_character`'s doc
    // comment).
    for tick in 0..10u32 {
        let inputs = [
            TickInput {
                id: 0,
                input: PlayerInput::default(),
                kill: false,
            },
            TickInput {
                id: 1,
                input: PlayerInput {
                    fire: 1,
                    ..Default::default()
                },
                kill: tick == 0,
            },
        ];
        world.step(&inputs);
    }
    assert!(
        world.characters[1].unwrap().alive,
        "id 1 should have respawned by tick 10"
    );
}

#[test]
fn no_weak_hook_runs_without_panicking() {
    let map = synthetic::build("front").expect("front recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 42);
    world.config.sv_no_weak_hook = true;
    spawn_two(&mut world);
    for _tick in 0..300u32 {
        let inputs = [
            TickInput {
                id: 0,
                input: PlayerInput {
                    direction: 1,
                    target_x: 1,
                    target_y: 0,
                    hook: 1,
                    ..Default::default()
                },
                kill: false,
            },
            TickInput {
                id: 1,
                input: PlayerInput {
                    direction: -1,
                    target_x: -1,
                    target_y: 0,
                    hook: 1,
                    ..Default::default()
                },
                kill: false,
            },
        ];
        world.step(&inputs);
    }
}

/// Pins the full [`world::World::init`] rule end to end, replaying exactly the corpus's own
/// `nohit` shape (a pre-init `--cfg` of `"sv_hit 0"`, no map settings, then a post-init re-apply
/// of the same line): `sv_hit` ends up `true` (the `sv_ddrace_tune_reset` block wipes the
/// pre-init `0`, and the post-init pass is rejected because it is now locked), with exactly one
/// warning recorded — this is the *rule*, not a hardcoded no-op, and it reproduces the exact same
/// observable outcome the Oracle B corpus's "nohit" traces show.
#[test]
fn world_init_models_sv_hit_ending_up_true_for_a_pre_init_only_nohit_cfg() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 1);
    world.init(["sv_hit 0"]).unwrap();
    assert!(
        world.config.sv_hit,
        "sv_ddrace_tune_reset must wipe the pre-init sv_hit=0 back to true"
    );
    assert!(world.command_log.is_empty(), "init()'s own passes run before the lock");
    world.apply_commands(["sv_hit 0"]).unwrap();
    assert!(world.config.sv_hit, "the post-init re-apply must be rejected (locked)");
    assert_eq!(world.command_log.len(), 1);
    assert!(world.command_log[0].contains("sv_hit"));
}

/// A map setting (applied inside [`world::World::init`], *after* the `sv_ddrace_tune_reset` wipe
/// but *before* the lock) is exactly how a map can make `sv_hit 0` actually stick — unlike a
/// `--cfg` line, which the previous test shows gets wiped and then locked out.
#[test]
fn world_init_lets_a_map_setting_make_sv_hit_stick() {
    let mut map = synthetic::build("freeze").expect("freeze recipe must exist");
    map.settings.push("sv_hit 0".to_string());
    let mut world: World<f32> = World::from_map(&map, 1);
    world.init(std::iter::empty()).unwrap();
    assert!(
        !world.config.sv_hit,
        "a map setting runs after the wipe, so it should stick"
    );
    assert!(world.config.game_settings_locked);
}

/// Real DDNet's `IConsole::ExecuteLine` dispatches one line at a time and keeps going past a bad
/// one; `World::init` must report *every* unrecognized command from a multi-line pass, not just
/// the first, and must still apply every *recognized* line around it (task spec acceptance
/// criterion 2: "unknown commands -> explicit error listing them").
#[test]
fn init_reports_every_unknown_command_not_just_the_first() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 1);
    // `sv_freeze_delay` (unlike `sv_hit`) isn't reset by step 3's `sv_ddrace_tune_reset` block,
    // so its value here can only have come from actually applying this pass's own middle line.
    let err = world
        .init(["not_a_real_command 1", "sv_freeze_delay 9", "also_not_real 2"])
        .unwrap_err();
    assert_eq!(err.len(), 2, "both unknown lines must be reported: {err:?}");
    assert_eq!(err[0].line, "not_a_real_command 1");
    assert_eq!(err[1].line, "also_not_real 2");
    assert_eq!(
        world.config.sv_freeze_delay, 9,
        "the recognized sv_freeze_delay line in between the two bad ones must still apply"
    );
}

// --- Regression tests for the 5 Oracle B diagnosis fixes + 2 "other deviations" (see this
// crate's `BUILD REPORT`). `tests/parity_oracle_b.rs`'s curated cross-section pins the exact
// evidence *traces* end to end; these pin the specific *mechanism* each fix changed, in
// isolation. --------------------------------------------------------------------------------

/// Fix 3 (the world `Tick` loop being cut short when a teamed character dies,
/// `teams.cpp:486,497`/`gameworld.cpp:160-182`): a death that changes the dying character's own
/// team must immediately mark every one of its own live projectiles for removal
/// (`RemoveEntitiesFromPlayer`), regardless of weapon type — not merely rely on the unrelated,
/// weapon-type-gated "owner not alive" lazy check `projectile_tick` already had
/// (`projectile.cpp:121-129`, which explicitly excludes grenades). Also pins
/// [`World::team_changed_this_pass`] itself getting set, the signal `World::world_tick`'s
/// character loop uses to stop early.
#[test]
fn death_that_changes_team_marks_own_projectiles_for_removal() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 1);
    spawn_two(&mut world);
    // Move id 0 off the default `TEAM_FLOCK` (`0`) so its later death is a genuine team change
    // (`teams.cpp:486`'s `OldTeam != Team` gate) — a `TEAM_FLOCK` character dying again would
    // never trip this fix at all (`SetForceCharacterTeam` is a no-op when `Team == OldTeam`).
    world::set_force_character_team(&mut world, 0, 1);
    // That call above is itself a team change (`0` -> `1`), so it already set this; clear it so
    // the assertion below actually reflects `die()`'s own effect, not a stale leftover.
    world.team_changed_this_pass = false;

    // A grenade specifically, since it's the one weapon type the *other*, pre-existing,
    // lazy "owner not alive" check in `projectile_tick` deliberately never marks for destroy —
    // proving this removal is the *new*, unconditional `RemoveEntitiesFromPlayer` path, not that
    // older mechanism firing instead.
    world.projectiles.push(world::Projectile {
        weapon_type: ddai_physics::core::WEAPON_GRENADE,
        owner: 0,
        pos: Vec2::new(300.0, 300.0),
        direction: Vec2::new(1.0, 0.0),
        init_dir: Vec2::new(1.0, 0.0),
        life_span: -1,
        start_tick: 0,
        freeze: false,
        explosive: true,
        bouncing: 0,
        tune_zone: 0,
        layer: world::Layer::Game,
        number: 0,
        marked_for_destroy: false,
    });
    assert!(!world.projectiles[0].marked_for_destroy);

    world::die(&mut world, 0, 0, world::WEAPON_WORLD);

    assert!(
        world.projectiles[0].marked_for_destroy,
        "a death that changes the dying character's team must immediately mark its own \
         projectiles for removal, including a grenade"
    );
    assert!(
        world.team_changed_this_pass,
        "World::team_changed_this_pass must be set so World::world_tick's character loop can \
         stop ticking the remaining characters this pass"
    );
}

/// A same-team death (`Team == OldTeam` — here, both characters default to `TEAM_FLOCK`) must
/// *not* touch unrelated projectiles: the fix is specifically gated on a genuine team change,
/// not on every death.
#[test]
fn death_without_a_team_change_does_not_touch_projectiles() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 1);
    spawn_two(&mut world);
    world.projectiles.push(world::Projectile {
        weapon_type: ddai_physics::core::WEAPON_GUN,
        owner: 0,
        pos: Vec2::new(300.0, 300.0),
        direction: Vec2::new(1.0, 0.0),
        init_dir: Vec2::new(1.0, 0.0),
        life_span: -1,
        start_tick: 0,
        freeze: false,
        explosive: false,
        bouncing: 0,
        tune_zone: 0,
        layer: world::Layer::Game,
        number: 0,
        marked_for_destroy: false,
    });

    world::die(&mut world, 0, 0, world::WEAPON_WORLD);

    assert!(
        !world.projectiles[0].marked_for_destroy,
        "id 0 was already TEAM_FLOCK, so dying back into TEAM_FLOCK is not a team change"
    );
    assert!(!world.team_changed_this_pass);
}

/// Fix 5 (`CanSpawn`/`EvaluateSpawnType`'s per-type early return,
/// `gamecontroller.cpp:110-111,161-173`): under `player_collision == 0`, once `DEFAULT`'s own
/// pass has found *any* spot, `RED`/`BLUE` must never be evaluated at all — even when a `RED`
/// point would score far better (lower) than every occupied `DEFAULT` point. An earlier revision
/// of this port flattened `[DEFAULT, RED, BLUE]` into one pool and scored every point together,
/// which would have picked the far-better-scoring `RED` point here instead.
#[test]
fn can_spawn_stops_at_default_type_when_player_collision_is_off() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 1);
    assert!(world.tuning.zone_mut(0).set_by_name("player_collision", 0.0));

    let default_point = Vec2::new(100.0, 100.0);
    world.spawn_points = vec![default_point];
    world.spawn_points_red = vec![Vec2::new(9000.0, 9000.0)];
    world.spawn_points_blue = vec![];

    // A character sitting exactly on the default point gives it a huge (`1e9`) score —
    // `EvaluateSpawnPos`'s `d == 0` case — while the (empty-of-neighbors) red point scores `0`,
    // far better if the two types were ever pooled and compared together.
    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, default_point);

    let pos = world.can_spawn(1).expect("must find a spawn point");
    assert_eq!(
        pos, default_point,
        "DEFAULT's own result must win outright once found, never displaced by a better-scoring \
         RED point"
    );
}

/// With `player_collision` at its normal nonzero default, `DEFAULT`'s only point is occupied
/// (`j == 0` finds no empty slot in any of its 5 offsets) and its `j == 1` fallback still wins
/// the *type*, but a much-better-scoring, unoccupied `RED` point is available too. `CanSpawn`
/// still evaluates `RED` in this case (`PlayerCollision` is on, so `EvaluateSpawnType`'s early
/// return never gates it) — pinning that this crate's fix isn't "RED is never used", only "RED is
/// skipped exactly when real DDNet would skip it".
#[test]
fn can_spawn_still_considers_red_type_when_player_collision_is_on() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 1);

    let default_point = Vec2::new(100.0, 100.0);
    let red_point = Vec2::new(9000.0, 9000.0);
    world.spawn_points = vec![default_point];
    world.spawn_points_red = vec![red_point];
    world.spawn_points_blue = vec![];

    world.players[0] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, default_point);

    let pos = world.can_spawn(1).expect("must find a spawn point");
    assert_eq!(
        pos, red_point,
        "with player_collision on, DEFAULT's only point is occupied by id 0 (j==0 finds no free \
         offset) and RED's own unoccupied, better-scoring point must win"
    );
}

/// "Other deviation" A (`pickup.cpp:62`'s armor-strip loop bound is `j < NUM_WEAPONS`, inclusive
/// of `WEAPON_NINJA`): an armor pickup must strip an active ninja pickup's `got`/`ammo` state
/// too, not stop one weapon short of it.
#[test]
fn armor_pickup_strips_ninja_too() {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 1);
    spawn_two(&mut world);
    world::give_weapon_to(&mut world, 0, ddai_physics::core::WEAPON_NINJA);
    assert!(world.cores.get(0).unwrap().weapons[ddai_physics::core::WEAPON_NINJA as usize].got);

    world.pickups.push(world::Pickup {
        pos: world.cores.get(0).unwrap().pos,
        kind: world::PickupKind::Armor,
        layer: world::Layer::Game,
        number: 0,
        mcore: Vec2::new(0.0, 0.0),
    });
    let idx = world.pickups.len() - 1;
    world::pickup_tick(&mut world, idx);

    let ninja = world.cores.get(0).unwrap().weapons[ddai_physics::core::WEAPON_NINJA as usize];
    assert!(!ninja.got, "an armor pickup must strip WEAPON_NINJA's `got` flag too");
    assert_eq!(
        ninja.ammo, 0,
        "and its ammo, matching pickup.cpp:62's `j < NUM_WEAPONS` bound"
    );
}
