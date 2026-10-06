//! Task 3.10 ("held block"): what becomes of a frozen tee if nobody touches it again.
//!
//! A block is only worth something if it **lasts**: DDNet keeps a tee frozen while it touches a freeze tile (the timer is renewed every
//! second) and for `sv_freeze_delay` = 3 s (150 ticks) after it left the tile; then it thaws and walks away. The planner's rollouts look
//! 27 ticks ahead, so a victim that is frozen *off* the freeze zone and will thaw in 150 ticks looks the same to it as one that is sealed
//! in. [`passive_forecast`] plays the victim alone on the real physics (no input: a frozen tee cannot act, and a thawed one is not modelled)
//! and answers "in how many ticks is it free again", with an exit as soon as the answer is certain (the tee rests: then it is held
//! for good on a freeze tile, or free after its remaining freeze time off it).
//!
//! The function **mutates the world it is given** (it removes every other tee, like [`crate::seal::sealed_in`]): give it a scratch copy,
//! or a rollout's world just before the state is restored. Allocation-free once the world is warm (it reuses one event buffer by taking
//! the world's own `step_into`).

use crate::plan_world::PlanWorld;
use crate::seal::touches_freeze;
use crate::types::empty_input;

/// The horizon of the held-block metric: 250 ticks (5 s, more than `sv_freeze_delay`).
pub const HELD_HORIZON_TICKS: i32 = 250;

/// The answer of [`passive_forecast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Forecast {
    /// Ticks until the tee is free again (alive and not frozen), counted from the world's tick; `None` = it is still out when the horizon
    /// ends (or for good).
    pub free_in: Option<i32>,
    /// The tee dies (a kill tile) inside the horizon: out for good.
    pub died: bool,
    /// Physics ticks of the lone tee that were simulated (the work meter charges them).
    pub steps: i32,
}

impl Forecast {
    /// Out of the game for the whole horizon (frozen the whole time, or dead).
    pub fn held(&self) -> bool {
        self.free_in.is_none()
    }

    /// Ticks the tee stays out, capped at `horizon` (`horizon` when held).
    pub fn out_ticks(&self, horizon: i32) -> i32 {
        self.free_in.map_or(horizon, |t| t.min(horizon))
    }
}

/// How many tees the world holds (no allocation: probes the client ids).
pub fn tee_count<W: PlanWorld>(world: &W) -> usize {
    (0..ddai_physics::core::MAX_CLIENTS as i32)
        .filter(|&i| world.get_tee(i).is_some())
        .count()
}

/// Forecast of tee `id` for `horizon` ticks with every other tee removed and the tee itself on neutral input (see the module docs).
pub fn passive_forecast<W: PlanWorld>(world: &mut W, id: i32, horizon: i32) -> Forecast {
    // No allocation: probe the client ids instead of listing the tees.
    for other in 0..ddai_physics::core::MAX_CLIENTS as i32 {
        if other != id && world.get_tee(other).is_some() {
            world.remove_tee(other);
        }
    }
    let Some(start) = world.get_tee(id) else {
        return Forecast {
            free_in: None,
            died: true,
            steps: 0,
        };
    };
    if !start.alive {
        return Forecast {
            free_in: None,
            died: true,
            steps: 0,
        };
    }
    if !start.frozen {
        return Forecast {
            free_in: Some(0),
            died: false,
            steps: 0,
        };
    }
    if start.deep_frozen == Some(true) {
        return Forecast {
            free_in: None,
            died: false,
            steps: 0,
        };
    }
    world.set_input(id, empty_input());
    let mut events = Vec::new();
    let (mut last_pos, mut last_vel) = (start.pos, start.vel);
    for t in 0..horizon {
        events.clear();
        world.step_into(&mut events);
        let Some(now) = world.get_tee(id) else {
            return Forecast {
                free_in: None,
                died: true,
                steps: t + 1,
            };
        };
        if !now.alive {
            return Forecast {
                free_in: None,
                died: true,
                steps: t + 1,
            };
        }
        if !now.frozen {
            return Forecast {
                free_in: Some(t + 1),
                died: false,
                steps: t + 1,
            };
        }
        // A fixed point of the physics (nobody else is in the world): the tee will not move again, so its fate is a countdown.
        if now.pos == last_pos && now.vel == last_vel && now.hook_state <= 0 {
            if touches_freeze(world.collision(), now.pos.x, now.pos.y) {
                return Forecast {
                    free_in: None,
                    died: false,
                    steps: t + 1,
                };
            }
            let free = i64::from(t) + 1 + now.freeze_ticks_left.max(0);
            return Forecast {
                free_in: (free < i64::from(horizon)).then_some(free as i32),
                died: false,
                steps: t + 1,
            };
        }
        last_pos = now.pos;
        last_vel = now.vel;
    }
    Forecast {
        free_in: None,
        died: false,
        steps: horizon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics_adapter::PhysicsWorld;
    use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
    use std::sync::Arc;

    /// A room 40 wide and 16 tall with a solid floor at row 10 (the tee stands on row 9) and, optionally, a freeze strip in the floor row
    /// `strip` (tiles `x0..=x1`, replacing the floor there: a pit).
    fn room(strip: Option<(usize, usize)>) -> Arc<MapData> {
        let (w, h) = (40usize, 16usize);
        let mut game = vec![Tile::default(); w * h];
        let tile = |index| Tile {
            index,
            ..Tile::default()
        };
        for x in 0..w {
            for y in 10..13 {
                game[y * w + x] = tile(TILE_SOLID);
            }
        }
        if let Some((x0, x1)) = strip {
            for x in x0..=x1 {
                for y in 10..12 {
                    game[y * w + x] = tile(TILE_FREEZE);
                }
            }
        }
        Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    fn frozen_world(map: &Arc<MapData>, tile_x: f64, left: i64) -> PhysicsWorld {
        let mut w = PhysicsWorld::new(Arc::clone(map), 1);
        w.add_tee(
            0,
            crate::vmath::Vec2 {
                x: tile_x * 32.0 + 16.0,
                y: 9.0 * 32.0 + 16.0,
            },
        );
        let mut st = w.get_tee(0).unwrap();
        st.frozen = true;
        st.freeze_ticks_left = left;
        w.apply_tee_state(0, &st);
        w
    }

    #[test]
    fn a_frozen_tee_on_open_ground_thaws_when_its_freeze_runs_out() {
        let map = room(None);
        let mut w = frozen_world(&map, 20.0, 120);
        let f = passive_forecast(&mut w, 0, HELD_HORIZON_TICKS);
        let t = f.free_in.expect("it thaws");
        assert!((115..=130).contains(&t), "about the remaining 120 ticks, got {t}");
        assert!(!f.died && !f.held());
        assert_eq!(f.out_ticks(HELD_HORIZON_TICKS), t);
    }

    #[test]
    fn a_frozen_tee_resting_in_freeze_is_held_for_the_whole_horizon() {
        let map = room(Some((18, 22)));
        let mut w = frozen_world(&map, 20.0, 40);
        // Put the tee into the pit: the strip is 2 tiles deep from row 10.
        let mut st = w.get_tee(0).unwrap();
        st.pos.y = 10.0 * 32.0 + 16.0;
        w.apply_tee_state(0, &st);
        let f = passive_forecast(&mut w, 0, HELD_HORIZON_TICKS);
        assert!(f.held(), "{f:?}");
        assert_eq!(f.out_ticks(HELD_HORIZON_TICKS), HELD_HORIZON_TICKS);
    }

    #[test]
    fn a_tee_that_is_not_frozen_is_free_at_once_and_a_missing_one_is_held() {
        let map = room(None);
        let mut w = PhysicsWorld::new(Arc::clone(&map), 1);
        w.add_tee(
            0,
            crate::vmath::Vec2 {
                x: 20.0 * 32.0,
                y: 9.0 * 32.0 + 16.0,
            },
        );
        assert_eq!(passive_forecast(&mut w, 0, 250).free_in, Some(0));
        let f = passive_forecast(&mut w, 7, 250);
        assert!(f.held() && f.died, "{f:?}");
    }

    #[test]
    fn a_long_freeze_off_the_zone_is_held_when_it_outlasts_the_horizon() {
        let map = room(None);
        let mut w = frozen_world(&map, 20.0, 400);
        assert!(passive_forecast(&mut w, 0, 250).held());
    }
}
