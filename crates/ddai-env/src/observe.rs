//! Building the plain-data [`Observation`] a [`ddai_brain::Brain`] sees from the arena's
//! `World<f32>`. The field mapping is the one the live path uses
//! (`ddai-world::live_world::character_observation`), so a brain trained or scored in the arena
//! is fed the same features it will get from `LiveWorld`.

use std::sync::Arc;

use ddai_brain::{CharacterObservation, Observation, jumps_left};
use ddai_physics::core::{self, MAX_CLIENTS};
use ddai_physics::map::MapData;
use ddai_physics::world::World;

/// `true` when `id` has a live character (a core exists and `alive` is set).
pub fn is_alive(world: &World<f32>, id: i32) -> bool {
    id_index(id).is_some_and(|i| {
        world.cores.slot_of(i as u8).is_some() && world.characters[i].as_ref().is_some_and(|c| c.alive)
    })
}

/// "Out" in the rules sense: dead, or frozen (`freeze_time > 0`).
pub fn is_out(world: &World<f32>, id: i32) -> bool {
    match id_index(id) {
        Some(i) if is_alive(world, id) => world.characters[i].as_ref().is_some_and(|c| c.freeze_time > 0),
        _ => true,
    }
}

/// `-1` when `id` has no live character or is not hooking anybody.
pub fn hooked_player(world: &World<f32>, id: i32) -> i32 {
    if !is_alive(world, id) {
        return -1;
    }
    world.cores.get(id as u8).map_or(-1, |c| c.hooked_player())
}

fn id_index(id: i32) -> Option<usize> {
    (0..MAX_CLIENTS as i32).contains(&id).then_some(id as usize)
}

/// One live character's observable state; `None` for a dead/absent one.
pub fn character_observation(world: &World<f32>, id: i32) -> Option<CharacterObservation> {
    let i = id_index(id)?;
    let core = world.cores.get(id as u8)?;
    let character = world.characters[i].as_ref().filter(|c| c.alive)?;
    let grounded = world.collision.is_on_ground(core.pos, core::physical_size::<f32>());
    Some(CharacterObservation {
        id,
        team: world.teams_core.team(id),
        pos: core.pos,
        vel: core.vel,
        hook_state: core.hook_state,
        hook_pos: core.hook_pos,
        hooked_player: core.hooked_player(),
        is_frozen: character.freeze_time > 0,
        is_deep_frozen: core.deep_frozen,
        is_live_frozen: core.live_frozen,
        freeze_ticks_remaining: character.freeze_time,
        jumps_left: jumps_left(core.jumps, core.jumped, core.jumped_total, core.endless_jump, grounded),
        jumps_used: core.jumped_total,
        grounded,
        weapon: core.active_weapon,
        direction: core.direction,
    })
}

/// The observation for `self_id`: every other live character in `others`, `target_id` as chosen
/// by the caller, tick and tuning from the world (the tuning of the tee's own zone).
/// `None` when `self_id` itself is not alive.
pub fn observation(
    world: &World<f32>,
    map: &Arc<MapData>,
    self_id: i32,
    player_ids: &[i32],
    target_id: Option<i32>,
) -> Option<Observation> {
    let self_state = character_observation(world, self_id)?;
    let others = player_ids
        .iter()
        .filter(|&&id| id != self_id)
        .filter_map(|&id| character_observation(world, id))
        .collect();
    let tuning = world.cores.get(self_id as u8)?.tuning;
    Some(Observation {
        map: map.clone(),
        tick: world.tick,
        self_state,
        others,
        target_id,
        tuning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tall room with a floor: tee 0 stands, jumps, then double-jumps. `jumps_left` must follow
    /// the HUD: 2 on the ground, 1 airborne after the ground jump, 0 after the air jump (it used to
    /// report the constant jump capacity `core.jumps`).
    #[test]
    fn jumps_left_counts_jumps_not_capacity() {
        use ddai_physics::core::PlayerInput;
        use ddai_physics::map::{MapData, TILE_SOLID, Tile};
        use ddai_physics::world::{TickInput, spawn_character};
        let (w, h) = (12usize, 20usize);
        let mut game = vec![Tile::default(); w * h];
        for x in 0..w {
            game[(h - 1) * w + x] = Tile {
                index: TILE_SOLID,
                ..Default::default()
            };
        }
        let map = MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let mut world = World::<f32>::from_map(&map, 1);
        let _ = world.init(std::iter::empty::<&str>());
        spawn_character(
            &mut world,
            0,
            ddai_physics::vmath::Vec2::new(5.0 * 32.0, 19.0 * 32.0 - 14.0),
        );
        let input = |jump: i32| PlayerInput {
            jump,
            target_y: -1,
            ..PlayerInput::default()
        };
        let left = |world: &World<f32>| character_observation(world, 0).unwrap().jumps_left;
        for _ in 0..3 {
            world.step(&[TickInput {
                id: 0,
                input: input(0),
                kill: false,
            }]);
        }
        assert_eq!(left(&world), 2, "standing");
        for _ in 0..4 {
            world.step(&[TickInput {
                id: 0,
                input: input(1),
                kill: false,
            }]);
        }
        assert_eq!(left(&world), 1, "airborne after the ground jump");
        world.step(&[TickInput {
            id: 0,
            input: input(0),
            kill: false,
        }]);
        for _ in 0..3 {
            world.step(&[TickInput {
                id: 0,
                input: input(1),
                kill: false,
            }]);
        }
        assert_eq!(left(&world), 0, "air jump spent");
    }
}
