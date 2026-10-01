//! The bot's bridge to the planner's helpers (`ddai-planner`): one private [`PhysicsWorld`] that is
//! re-synced from the world at hand (`sync_from`, allocation-free once warm) for the three things the
//! bot asks of the planner outside the brain:
//!
//! - **`sealed_in`** (`seal.ts:83-107`) for the target filter: would this tee, alone, still end up
//!   frozen in freeze after 90 ticks whatever it tries? (Such a tee is "settled" and not worth
//!   hitting.) Answers are cached 6 ticks per id by the caller.
//! - the **guard** (`bot.ts:2801-2820`): for a brain without its own shield (scripted, fly) and for
//!   wandering — keep the wanted input only if an escape from freeze still exists after holding it
//!   two ticks, else the shield's `saferInput`.
//! - **`rope_catches`** (`bot.ts:4817-4826`) for the hook veto: would a hook thrown along this aim
//!   catch a spared tee before the target or a wall?
//!
//! `ddai-planner`'s helpers allocate (a few small `Vec`s per call); they run only for frozen
//! candidates, for guarded inputs and for vetoed hooks — the bot's own steady-state path (no
//! frozen candidate, brain with its own shield) never enters them, which is what the allocation test
//! in `bot.rs` measures.

use std::collections::HashMap;
use std::sync::Arc;

use ddai_brain::{Action, IVec2};
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::World;
use ddai_planner::brains::{action_from_input, enemy_input_from_tee, input_from_action};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::seal::sealed_in;
use ddai_planner::shield::{escape_exists, safer_input};
use ddai_planner::types::PlayerInput as PlannerInput;

use crate::consts::*;
use crate::tees::{HOOK_IDLE, Tee, rope_catch_along};

pub struct PlanScratch {
    world: PhysicsWorld,
}

impl PlanScratch {
    pub fn new(map: Arc<MapData>) -> Self {
        PlanScratch {
            world: PhysicsWorld::new(map, 1),
        }
    }

    /// `isSealed` without the cache (`bot.ts:3030-3050`): `sealedIn(sim, id, state, held)` on a copy
    /// of `base` holding only that tee. `held` is what the tee's snapshot shows it doing.
    pub fn sealed(&mut self, base: &World<f32>, id: i32) -> bool {
        self.world.sync_from(base);
        let Some(state) = self.world.get_tee(id) else {
            return false;
        };
        let held = enemy_input_from_tee(&state);
        sealed_in(&mut self.world, id, &state, held)
    }

    /// `guard` (`bot.ts:2801-2820`). `predicted` is the world the wanted input takes effect in (our
    /// own in-flight inputs already applied — the TS rolled `lag` ticks with `prevInput` for that);
    /// `prev` is the input we last sent (the shield's "sent aim"). Returns the wanted action when
    /// an escape from freeze exists after holding it 2 ticks, else the shield's safer input, else the
    /// wanted one (`?? input`). Nothing to protect from when frozen or dead.
    pub fn guard(&mut self, predicted: &World<f32>, self_id: i32, wanted: Action, prev: &PlannerInput) -> Action {
        self.world.sync_from(predicted);
        let Some(me) = self.world.get_tee(self_id) else {
            return wanted;
        };
        if me.frozen || !me.alive {
            return wanted;
        }
        for other in self.world.all_tees() {
            if other.id != self_id {
                self.world.remove_tee(other.id);
            }
        }
        let input = input_from_action(&wanted, prev);
        let none: HashMap<i32, PlannerInput> = HashMap::new();
        if escape_exists(&mut self.world, self_id, &input, GUARD_HOLD_TICKS, &none) {
            return wanted;
        }
        match safer_input(&mut self.world, self_id, &input, GUARD_HOLD_TICKS, &none, Some(prev)) {
            Some(safe) => {
                let mut out = action_from_input(&safe);
                // The shield proposes direction/jump/hook/aim; fire and weapon stay the wanted ones.
                out.fire = wanted.fire;
                out.wanted_weapon = wanted.wanted_weapon;
                out
            }
            None => wanted,
        }
    }

    /// `ropeCatches(self, input, tees, before)` (`bot.ts:4817-4826`): a hook thrown along `aim` from
    /// `from` is stopped by the first wall (`intersect_line_hook`) or by `before` (the target) —
    /// does it catch any of `tees` earlier than that?
    pub fn rope_catches<'a>(
        &self,
        from: Vec2<f32>,
        aim: IVec2,
        tees: impl IntoIterator<Item = &'a Tee>,
        before: Option<&Tee>,
    ) -> bool {
        let n = f32::hypot(aim.x as f32, aim.y as f32);
        if n == 0.0 {
            return false;
        }
        let dir = Vec2::new(aim.x as f32 / n, aim.y as f32 / n);
        let to = ddai_planner::vmath::Vec2 {
            x: f64::from(from.x + dir.x * HOOK_LENGTH_PX),
            y: f64::from(from.y + dir.y * HOOK_LENGTH_PX),
        };
        let start = ddai_planner::vmath::Vec2 {
            x: f64::from(from.x),
            y: f64::from(from.y),
        };
        let hit = self.world.collision().intersect_line_hook(start, to);
        let mut stop = if hit.collision != 0 {
            ((hit.out_pos.x - start.x).hypot(hit.out_pos.y - start.y)) as f32
        } else {
            HOOK_LENGTH_PX
        };
        if let Some(b) = before {
            stop = stop.min(rope_catch_along(from, dir, b.pos));
        }
        tees.into_iter().any(|t| rope_catch_along(from, dir, t.pos) < stop)
    }
}

/// Is `t`'s hook out ([`HOOK_IDLE`] is the only "not out" state)?
pub fn hook_is_out(t: &Tee) -> bool {
    t.hook_state != HOOK_IDLE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapgrid::test_maps::{FREEZE, room};
    /// A world with our tee (id 0) at `(x, y)`, built the way the bot builds it: from a snapshot.
    fn world_with_tee(extra: &[(u32, u32, u8)], x: f32, y: f32) -> (Arc<MapData>, World<f32>) {
        use ddai_net::generated::objects;
        use ddai_net::view::CharacterView;
        let map = Arc::new(room(60, 30, extra));
        let cv = CharacterView {
            id: 0,
            character: objects::Character {
                tick: 0,
                x: x as i32,
                y: y as i32,
                vel_x: 0,
                vel_y: 0,
                angle: 0,
                direction: 0,
                jumped: 0,
                hooked_player: -1,
                hook_state: 0,
                hook_tick: 0,
                hook_x: x as i32,
                hook_y: y as i32,
                hook_dx: 0,
                hook_dy: 0,
                player_flags: 1,
                health: 10,
                armor: 0,
                ammo_count: -1,
                weapon: 0,
                emote: 0,
                attack_tick: 0,
            },
            ddnet: Some(objects::DDNetCharacter {
                flags: 0,
                freeze_end: 0,
                jumps: 2,
                tele_checkpoint: -1,
                strong_weak_id: 0,
                jumped_total: -1,
                ninja_activation_tick: -1,
                freeze_start: -1,
                target_x: 0,
                target_y: 0,
                tune_zone_override: -1,
            }),
        };
        let mut live = ddai_world::LiveWorld::new(Arc::clone(&map), 0, 1);
        live.on_snapshot(ddai_world::SnapshotInput::new(
            1000,
            &[cv],
            ddai_net::tuning::DEFAULT_TUNE_PARAMS,
        ));
        (map, live.base_world().clone())
    }

    /// Floor at row 29 (y = 928); a freeze tile at (20, 28) right next to the tee. Freezing needs the
    /// tee's *centre* inside the tile, so it stands 1 px short of it.
    #[test]
    fn guard_replaces_an_input_that_freezes_us_within_two_ticks_and_keeps_a_safe_one() {
        let (map, w) = world_with_tee(&[(20, 28, FREEZE)], 20.0 * 32.0 - 1.0, 29.0 * 32.0 - 15.0);
        let mut plan = PlanScratch::new(Arc::clone(&map));
        let prev = ddai_planner::types::empty_input();
        let run_right = Action {
            direction: 1,
            ..Action::neutral()
        };
        let g = plan.guard(&w, 0, run_right, &prev);
        assert_ne!(g, run_right, "running into the freeze tile must be replaced");
        let run_left = Action {
            direction: -1,
            ..Action::neutral()
        };
        assert_eq!(plan.guard(&w, 0, run_left, &prev), run_left, "walking away is fine");
    }

    #[test]
    fn rope_catches_a_tee_on_the_line_before_a_wall_and_not_one_behind_it() {
        let (map, _w) = world_with_tee(&[], 300.0, 29.0 * 32.0 - 15.0);
        let plan = PlanScratch::new(map);
        let from = Vec2::new(300.0, 500.0);
        let on_line = Tee {
            id: 1,
            alive: true,
            pos: Vec2::new(500.0, 500.0),
            ..Tee::DEAD
        };
        let off_line = Tee {
            id: 2,
            alive: true,
            pos: Vec2::new(500.0, 560.0),
            ..Tee::DEAD
        };
        let aim = IVec2::new(100, 0);
        assert!(plan.rope_catches(from, aim, [&on_line], None));
        assert!(!plan.rope_catches(from, aim, [&off_line], None));
        assert!(
            !plan.rope_catches(
                from,
                aim,
                [&on_line],
                Some(&Tee {
                    pos: Vec2::new(400.0, 500.0),
                    ..on_line
                })
            ),
            "the target is nearer"
        );
        assert!(
            !plan.rope_catches(from, IVec2::new(0, 0), [&on_line], None),
            "no aim, no rope"
        );
        // A hook aimed at a far wall stops there: a tee beyond the wall's hit point is not caught.
        let wall_map = Arc::new(room(60, 30, &[(14, 15, crate::mapgrid::test_maps::SOLID)]));
        let plan2 = PlanScratch::new(wall_map);
        let behind = Tee {
            id: 3,
            alive: true,
            pos: Vec2::new(14.0 * 32.0 + 100.0, 15.0 * 32.0 + 16.0),
            ..Tee::DEAD
        };
        assert!(!plan2.rope_catches(
            Vec2::new(10.0 * 32.0, 15.0 * 32.0 + 16.0),
            IVec2::new(100, 0),
            [&behind],
            None
        ));
    }

    #[test]
    fn sealed_is_true_for_a_frozen_tee_in_a_pit_and_false_for_one_that_can_get_out() {
        let pit: Vec<(u32, u32, u8)> = (10..20).flat_map(|x| (20..29).map(move |y| (x, y, FREEZE))).collect();
        let (map, mut w) = world_with_tee(&pit, 15.0 * 32.0, 25.0 * 32.0);
        {
            let ch = w.characters[0].as_mut().unwrap();
            ch.freeze_time = 140;
        }
        let mut plan = PlanScratch::new(Arc::clone(&map));
        assert!(plan.sealed(&w, 0), "frozen in the middle of a freeze pit");
        let (map2, mut w2) = world_with_tee(&[], 15.0 * 32.0, 29.0 * 32.0 - 15.0);
        w2.characters[0].as_mut().unwrap().freeze_time = 140;
        let mut plan2 = PlanScratch::new(map2);
        assert!(!plan2.sealed(&w2, 0), "frozen on safe ground: it thaws");
    }
}
