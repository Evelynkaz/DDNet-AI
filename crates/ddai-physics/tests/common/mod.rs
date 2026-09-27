//! Shared helper for the parity tests (`parity_fixtures.rs`, `parity_bulk.rs`): runs a
//! `ddai_trace::scenario::Scenario` through the Rust `f32` physics port
//! (`ddai_physics::core`/`core_world`) and returns one `ddai_trace::trace::TraceRow` per (tick,
//! character) — the same shape `ddai_trace::trace::Trace` uses, so a caller can build a `Trace`
//! directly from the result and reuse `Trace::tick_hashes()`/`ddai_trace::trace::diff` unchanged.
//!
//! This file is *not* itself a test binary (it lives under `tests/common/`, which Cargo excludes
//! from test auto-discovery) — both parity test binaries `include!`/`mod` it.

use ddai_physics::collision::Collision;
use ddai_physics::core::{CharacterCore, PlayerInput as PhysInput, TeamsCore, WorldCore};
use ddai_physics::core_world;
use ddai_physics::map::MapData;
use ddai_physics::real::Real;
use ddai_physics::tuning::TuningParams;
use ddai_physics::vmath::Vec2;
use ddai_trace::scenario::{PlayerInput as TraceInput, Scenario, resolve_input};
use ddai_trace::trace::{CharacterCoreState, TraceRow};

/// Compact `WorldCore` capacity for these tests — the generator caps scenarios at 8 characters
/// (see `ddai_trace::generator::random_v1`), so `CAP` matches that exactly; `WorldCore`'s array
/// size only matters for the performance benchmark, which picks its own `CAP` (see
/// `benches/physics.rs`), not for correctness here.
pub const CAP: usize = 8;

fn to_phys_input(i: TraceInput) -> PhysInput {
    PhysInput {
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
    }
}

/// Builds the `state_schema`-ordered `CharacterCoreState` (`docs/formats.md` §6.2) for one
/// character's current state — every field the trace format/canonical hash cares about.
pub fn char_state(core: &CharacterCore<f32>) -> CharacterCoreState {
    CharacterCoreState {
        pos_x: core.pos.x,
        pos_y: core.pos.y,
        vel_x: core.vel.x,
        vel_y: core.vel.y,
        hook_pos_x: core.hook_pos.x,
        hook_pos_y: core.hook_pos.y,
        hook_dir_x: core.hook_dir.x,
        hook_dir_y: core.hook_dir.y,
        hook_tele_base_x: core.hook_tele_base.x,
        hook_tele_base_y: core.hook_tele_base.y,
        hook_tick: core.hook_tick,
        hook_state: core.hook_state,
        hooked_player: core.hooked_player(),
        active_weapon: core.active_weapon,
        new_hook: core.new_hook as i32,
        jumped: core.jumped,
        jumped_total: core.jumped_total,
        jumps: core.jumps,
        direction: core.direction,
        angle: core.angle,
        triggered_events: core.triggered_events,
        colliding: core.colliding,
        // `move_restrictions` mirrors Oracle A recomputing it via
        // `Collision::get_move_restrictions` rather than reading a private field (docs/formats.md
        // §5.5) — this crate's `CharacterCore::move_restrictions()` getter holds the identical
        // value for the same reason (computed once, at the top of `tick`, from the same `m_Pos`).
        left_wall: core.left_wall as i32,
        move_restrictions: core.move_restrictions(),
        solo: core.solo as i32,
        collision_disabled: core.collision_disabled as i32,
        endless_hook: core.endless_hook as i32,
        hook_hit_disabled: core.hook_hit_disabled as i32,
    }
}

/// Applies a scenario's tuning overrides on top of `CTuningParams::DEFAULT`, exactly like Oracle A
/// (`tools/ddnet-oracle/oracle_core.cpp`'s `ApplyTuningOverrides`: writes the raw fixed-point
/// `value_x100` directly, never through the `f32`-round-tripping `set`).
pub fn build_tuning(scenario: &Scenario) -> TuningParams {
    let mut tuning = TuningParams::default();
    for o in &scenario.tuning_overrides {
        let idx = (0..TuningParams::num())
            .find(|&i| TuningParams::name(i).eq_ignore_ascii_case(&o.name))
            .unwrap_or_else(|| panic!("unknown tuning parameter '{}'", o.name));
        assert!(tuning.set_raw(idx, o.value_x100));
    }
    tuning
}

/// Runs `scenario` on `map` through the Rust `f32` physics port, reproducing Oracle A's per-tick
/// loop (`ddai_physics::core_world::step`) tick-for-tick, resolving each tick's aim-mode inputs
/// with the same shared `ddai_trace::scenario::resolve_input` Oracle A's own reimplementation
/// mirrors (see `docs/formats.md` §2.1 and the task spec's "Updates after the 1.2 review" note).
///
/// Returns `rows[tick][character_slot]`, `character_slot` indexing into `scenario.characters`
/// (matching `ddai_trace::trace::Trace::rows`'s shape exactly).
pub fn run_scenario(map: &MapData, scenario: &Scenario) -> Vec<Vec<TraceRow>> {
    assert!(
        scenario.characters.len() <= CAP,
        "scenario has {} characters, more than this test harness's CAP={CAP}",
        scenario.characters.len()
    );

    let collision: Collision<f32> = Collision::new(map);
    let teams = TeamsCore::new();
    let tuning = build_tuning(scenario);

    let mut entries = Vec::with_capacity(scenario.characters.len());
    for c in &scenario.characters {
        let mut core = CharacterCore::<f32>::default();
        core.reset();
        core.init();
        core.id = c.id as i32;
        core.pos = Vec2::new(c.spawn_x as f32, c.spawn_y as f32);
        core.tuning = tuning;
        entries.push((c.id as u8, core));
    }
    let mut world: WorldCore<f32, CAP> = WorldCore::from_characters(&entries);
    let ids_in_scenario_order: Vec<u8> = scenario.characters.iter().map(|c| c.id as u8).collect();
    let tick_order = core_world::tick_order_newest_first(&world, &ids_in_scenario_order);

    let mut prev_positions: Vec<(i32, i32)> = scenario.spawn_positions();
    let mut rows: Vec<Vec<TraceRow>> = Vec::with_capacity(scenario.ticks());

    for tick_inputs in &scenario.inputs {
        let mut resolved_inputs = Vec::with_capacity(scenario.characters.len());
        for (slot, input) in tick_inputs.iter().enumerate() {
            let resolved = resolve_input(input, slot, &prev_positions);
            let id = scenario.characters[slot].id as u8;
            world.get_mut(id).expect("character id must be present").input = to_phys_input(resolved);
            resolved_inputs.push(resolved);
        }

        core_world::step(&mut world, &collision, &teams, &tick_order, scenario.no_weak_hook);

        let mut tick_rows = Vec::with_capacity(scenario.characters.len());
        for (slot, c) in scenario.characters.iter().enumerate() {
            let core = world.get(c.id as u8).expect("character id must be present");
            prev_positions[slot] = (core.pos.x as i32, core.pos.y as i32);
            tick_rows.push(TraceRow {
                input: resolved_inputs[slot],
                state: char_state(core),
            });
        }
        rows.push(tick_rows);
    }

    rows
}

/// Like [`run_scenario`], but generic over [`Real`] instead of hardwired to `f32` — used only by
/// the `f64` smoke test (acceptance criterion 2.c: a non-`f32` instantiation compiles and runs
/// without panicking; no bit-exactness claim for it, see the task spec, so this returns just each
/// tick's positions rather than a fixture-comparable [`TraceRow`]) over the *golden fixture*
/// scenarios instead of one hand-written scenario (review round 2, finding F5).
///
/// Position-tracking for aim-mode resolution (`prev_positions`) uses a plain saturating
/// `as i32` cast here rather than [`Real::to_i32_trunc`]: unlike the physics core itself (where
/// every float→int conversion must match C++'s `cvttss2si` bit-for-bit, see finding F1), this is
/// test-harness bookkeeping with no oracle on the other side to match — it only needs to never
/// panic, which a saturating cast never does.
///
/// `#[allow(dead_code)]`: this file is `mod`-included separately into *both* `parity_fixtures.rs`
/// and `parity_bulk.rs` (see the module doc comment above), but only the former calls this
/// function — from that binary's point of view it's dead code, even though it's very much used.
#[allow(dead_code)]
pub fn run_scenario_generic<R: Real>(map: &MapData, scenario: &Scenario) -> Vec<Vec<(R, R)>> {
    assert!(
        scenario.characters.len() <= CAP,
        "scenario has {} characters, more than this test harness's CAP={CAP}",
        scenario.characters.len()
    );

    let collision: Collision<R> = Collision::new(map);
    let teams = TeamsCore::new();
    let tuning = build_tuning(scenario);

    let mut entries = Vec::with_capacity(scenario.characters.len());
    for c in &scenario.characters {
        let mut core = CharacterCore::<R>::default();
        core.reset();
        core.init();
        core.id = c.id as i32;
        core.pos = Vec2::new(R::from_i32(c.spawn_x), R::from_i32(c.spawn_y));
        core.tuning = tuning;
        entries.push((c.id as u8, core));
    }
    let mut world: WorldCore<R, CAP> = WorldCore::from_characters(&entries);
    let ids_in_scenario_order: Vec<u8> = scenario.characters.iter().map(|c| c.id as u8).collect();
    let tick_order = core_world::tick_order_newest_first(&world, &ids_in_scenario_order);

    let mut prev_positions: Vec<(i32, i32)> = scenario.spawn_positions();
    let mut rows: Vec<Vec<(R, R)>> = Vec::with_capacity(scenario.ticks());

    for tick_inputs in &scenario.inputs {
        for (slot, input) in tick_inputs.iter().enumerate() {
            let resolved = resolve_input(input, slot, &prev_positions);
            let id = scenario.characters[slot].id as u8;
            world.get_mut(id).expect("character id must be present").input = to_phys_input(resolved);
        }

        core_world::step(&mut world, &collision, &teams, &tick_order, scenario.no_weak_hook);

        let mut tick_row = Vec::with_capacity(scenario.characters.len());
        for (slot, c) in scenario.characters.iter().enumerate() {
            let core = world.get(c.id as u8).expect("character id must be present");
            prev_positions[slot] = (core.pos.x.to_f64() as i32, core.pos.y.to_f64() as i32);
            tick_row.push((core.pos.x, core.pos.y));
        }
        rows.push(tick_row);
    }

    rows
}
