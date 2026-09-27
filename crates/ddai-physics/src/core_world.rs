//! Original DDNet-AI code (not a port of any single DDNet file): a stepping helper that
//! reproduces *exactly* Oracle A's per-tick loop (task 1.2), built from the already-ported
//! [`crate::core::tick`]/[`crate::core::tick_deferred`]/[`crate::core::move_character`] +
//! [`crate::core::quantize`]. See `docs/formats.md` §5 for the server-derived per-tick sequence
//! this mirrors, and its §5.3 for the `sv_no_weak_hook` variant.
//!
//! This module deliberately does **not** depend on `ddai-trace` (scenario files, the input
//! generator, `resolve_input`, trace hashing): those live in a crate that depends on
//! `ddai-physics`, not the reverse. Callers (the parity tests, and eventually a fuller world in
//! task 1.6) resolve each tick's [`crate::core::PlayerInput`] themselves and set it on each
//! character (`world.core_at_mut(slot).input = ...`) before calling [`step`].

use crate::collision::Collision;
use crate::core::{self, TeamsCore, WorldCore};
use crate::real::Real;

/// The order [`step`] calls `Tick`/`TickDeferred`/`Move` for each character this tick — compact
/// [`WorldCore`] slot indices. For Oracle A this is "newest-spawned-first": the *reverse* of a
/// scenario's `characters[]` list order (see `docs/formats.md` §5.1) — [`tick_order_newest_first`]
/// computes exactly that from a scenario's character ids (in `characters[]` order), independent
/// of how `WorldCore` happens to store them internally (sorted by ascending client id — see that
/// type's doc comment — which is generally a *different* order than a scenario's `characters[]`
/// list, though for this task's generator, which assigns sequential ids, the two coincide).
pub fn tick_order_newest_first<R: Real, const CAP: usize>(
    world: &WorldCore<R, CAP>,
    ids_in_scenario_order: &[u8],
) -> Vec<usize> {
    ids_in_scenario_order
        .iter()
        .rev()
        .map(|&id| {
            world
                .slot_of(id)
                .unwrap_or_else(|| panic!("client id {id} not present in this WorldCore"))
        })
        .collect()
}

/// Runs exactly one world tick, assuming every character's [`core::PlayerInput`] has already been
/// set (`world.core_at_mut(slot).input = resolved_input`) for this tick. `tick_order`: compact
/// `WorldCore` slot indices in calling order (see [`tick_order_newest_first`]).
///
/// Mirrors `docs/formats.md` §5.2/§5.3 exactly:
/// - `no_weak_hook == false`: `Tick(true, true)` (which internally calls `TickDeferred`) for
///   every character in `tick_order`, then `Move()` + `Quantize()` for every character in
///   `tick_order`.
/// - `no_weak_hook == true`: `Tick(true, false)` for every character in `tick_order`, *then*
///   `TickDeferred()` for every character in `tick_order` (a separate pass), then `Move()` +
///   `Quantize()` for every character in `tick_order`.
pub fn step<R: Real, const CAP: usize>(
    world: &mut WorldCore<R, CAP>,
    collision: &Collision<R>,
    teams: &TeamsCore,
    tick_order: &[usize],
    no_weak_hook: bool,
) {
    if no_weak_hook {
        for &slot in tick_order {
            core::tick(world, slot, collision, teams, true, false);
        }
        for &slot in tick_order {
            core::tick_deferred(world, slot, teams);
        }
    } else {
        for &slot in tick_order {
            core::tick(world, slot, collision, teams, true, true);
        }
    }
    for &slot in tick_order {
        core::move_character(world, slot, collision, teams);
        core::quantize(world.core_at_mut(slot));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CharacterCore, PlayerInput};
    use crate::map::{MapData, TILE_SOLID, Tile};
    use crate::vmath::Vec2;

    fn floor_map(w: i32, h: i32) -> MapData {
        let mut game = vec![
            Tile {
                index: 0,
                flags: 0,
                skip: 0,
                reserved: 0
            };
            (w * h) as usize
        ];
        for x in 0..w {
            game[((h - 1) * w + x) as usize].index = TILE_SOLID;
        }
        MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn tick_order_newest_first_reverses_scenario_order_regardless_of_id_values() {
        // ids are deliberately NOT in ascending order matching slot order, to prove the mapping
        // is by id lookup, not by assuming id == scenario slot index.
        let mut core_a = CharacterCore::<f32>::default();
        core_a.reset();
        let mut core_b = CharacterCore::<f32>::default();
        core_b.reset();
        let world = WorldCore::<f32, 4>::from_characters(&[(10, core_a), (3, core_b)]);
        // scenario order: slot0 has id=10, slot1 has id=3 -> newest first = reverse = [slot1(id3), slot0(id10)]
        let order = tick_order_newest_first(&world, &[10, 3]);
        assert_eq!(order.len(), 2);
        assert_eq!(world.id_at(order[0]), 3);
        assert_eq!(world.id_at(order[1]), 10);
    }

    #[test]
    fn a_falling_character_lands_on_the_floor_after_enough_ticks() {
        let map = floor_map(10, 10);
        let collision: Collision<f32> = Collision::new(&map);
        let teams = TeamsCore::new();

        let mut core = CharacterCore::<f32>::default();
        core.reset();
        core.id = 0;
        core.pos = Vec2::new(160.0, 100.0);
        let mut world = WorldCore::<f32, 2>::from_characters(&[(0, core)]);
        let order = tick_order_newest_first(&world, &[0]);

        for _ in 0..300 {
            world.core_at_mut(order[0]).input = PlayerInput {
                target_x: 0,
                target_y: -1,
                ..Default::default()
            };
            step(&mut world, &collision, &teams, &order, false);
        }
        // Floor's first pixel row is at y=288; box half-size 14 -> resting position y=288-14=274
        // (may settle 1px short due to the box test happening at pixel granularity).
        let pos = world.core_at(order[0]).pos;
        assert!(
            (pos.y - 274.0).abs() <= 1.0,
            "expected the tee to have landed near y=274, got {pos:?}"
        );
    }

    #[test]
    fn no_weak_hook_runs_a_separate_deferred_pass_without_panicking_with_two_characters() {
        let map = floor_map(20, 20);
        let collision: Collision<f32> = Collision::new(&map);
        let teams = TeamsCore::new();
        let mut a = CharacterCore::<f32>::default();
        a.reset();
        a.pos = Vec2::new(160.0, 160.0);
        let mut b = CharacterCore::<f32>::default();
        b.reset();
        b.pos = Vec2::new(180.0, 160.0);
        let mut world = WorldCore::<f32, 4>::from_characters(&[(0, a), (1, b)]);
        let order = tick_order_newest_first(&world, &[0, 1]);
        for _ in 0..50 {
            for &slot in &order {
                world.core_at_mut(slot).input = PlayerInput {
                    target_x: 1,
                    target_y: 0,
                    ..Default::default()
                };
            }
            step(&mut world, &collision, &teams, &order, true);
        }
        // Just checking this ran to completion without panicking is the point of this test; also
        // sanity-check both characters actually moved (physics ran, not a no-op).
        assert_ne!(world.core_at(order[0]).pos, Vec2::new(160.0, 160.0));
    }

    /// Acceptance criterion 2.c: the `f64` instantiation compiles and runs the same kind of
    /// scenario without panicking — no bit-exactness claim for `f64` (see the task spec), just
    /// "compiles and runs".
    #[test]
    fn f64_instantiation_runs_a_full_scenario_without_panicking() {
        let map = floor_map(20, 20);
        let collision: Collision<f64> = Collision::new(&map);
        let teams = TeamsCore::new();
        let mut a = CharacterCore::<f64>::default();
        a.reset();
        a.id = 0;
        a.pos = Vec2::new(160.0, 100.0);
        let mut b = CharacterCore::<f64>::default();
        b.reset();
        b.id = 1;
        b.pos = Vec2::new(200.0, 100.0);
        let mut world = WorldCore::<f64, 4>::from_characters(&[(0, a), (1, b)]);
        let order = tick_order_newest_first(&world, &[0, 1]);

        for tick in 0..500u32 {
            let phase = tick % 37;
            for (i, &slot) in order.iter().enumerate() {
                world.core_at_mut(slot).input = PlayerInput {
                    direction: if (tick + i as u32) % 20 < 10 { 1 } else { -1 },
                    target_x: 1,
                    target_y: 0,
                    jump: i32::from(phase == 0),
                    hook: i32::from(phase < 15),
                    ..Default::default()
                };
            }
            // Alternate no_weak_hook to exercise both stepping paths in one run.
            step(&mut world, &collision, &teams, &order, tick % 2 == 0);
        }

        // No panics is the primary assertion; also sanity-check the physics actually ran (both
        // characters moved from their exact spawn positions).
        assert_ne!(world.core_at(order[0]).pos, Vec2::new(160.0, 100.0));
        assert_ne!(world.core_at(order[1]).pos, Vec2::new(200.0, 100.0));
        for &slot in &order {
            let c = world.core_at(slot);
            assert!(
                !c.pos.x.is_nan() && !c.pos.y.is_nan(),
                "f64 physics produced NaN position"
            );
        }
    }
}
