// Ported from DDNet 20.1 `src/game/server/entities/character.cpp` (`CCharacter::HandleNinja`,
// `FireWeapon`'s `WEAPON_NINJA` case) and `datasrc/content.py` (the ninja weapon spec). DDNet's
// zlib-style license notice for the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real`, no `unsafe`, cosmetic parts
// (`CreateSound`, `CreateDamageInd`, `SetArmorProgress`) dropped.

//! Ninja: `HandleNinja()` (the dash, its hit pass and the 15-second expiry) and the dash
//! activation `FireWeapon()` performs for `WEAPON_NINJA`.

use super::{World, find_characters_in_range_into, remove_ninja, take_damage};
use crate::core::{self, CharacterCore, WEAPON_NINJA};
use crate::real::Real;
use crate::vmath::{self, Vec2};

/// `g_pData->m_Weapons.m_Ninja.m_Duration * Server()->TickSpeed() / 1000` with
/// `m_Duration == 15000` (`datasrc/content.py`, `Weapon_Ninja`), integer arithmetic.
pub const NINJA_DURATION_TICKS: i32 = 15000 * core::SERVER_TICK_SPEED / 1000;
/// `g_pData->m_Weapons.m_Ninja.m_Movetime * Server()->TickSpeed() / 1000` with `m_Movetime ==
/// 200`.
pub const NINJA_MOVE_TICKS: i32 = 200 * core::SERVER_TICK_SPEED / 1000;
/// `g_pData->m_Weapons.m_Ninja.m_Velocity` (`50`).
pub const NINJA_VELOCITY: i32 = 50;

/// `FireWeapon`'s `WEAPON_NINJA` case (`character.cpp:630-642`) on the firing character's working
/// copies: resets the hit list and starts the dash.
pub fn ninja_activate<R: Real>(character: &mut super::Character<R>, core: &mut CharacterCore<R>, direction: Vec2<R>) {
    // reset Hit objects
    character.num_objects_hit = 0;
    character.hit_objects = [0; 2];

    core.ninja.activation_dir = direction;
    core.ninja.current_move_time = NINJA_MOVE_TICKS;

    // clamp to prevent massive MoveBox calculation lag with SG bug
    core.ninja.old_vel_amount = vmath::length(core.vel)
        .clamp(R::ZERO, R::from_f64(6000.0))
        .to_i32_trunc();
}

/// `CCharacter::HandleNinja()` (`character.cpp:296-394`).
pub fn handle_ninja<R: Real>(world: &mut World<R>, id: i32, slot: usize) {
    if world.cores.core_at(slot).active_weapon != WEAPON_NINJA {
        return;
    }

    let tick = world.tick;
    if (tick - world.cores.core_at(slot).ninja.activation_tick) > NINJA_DURATION_TICKS {
        // time's up, return
        let mut character = world.characters[id as usize].expect("a ticking character exists");
        let mut core = *world.cores.core_at(slot);
        remove_ninja(&mut character, &mut core);
        world.characters[id as usize] = Some(character);
        *world.cores.core_at_mut(slot) = core;
        return;
    }

    // `GameServer()->CreateDamageInd(...)`/`SetArmorProgress` are cosmetic; `SetWeapon(WEAPON_NINJA)`
    // returns at once (`W == m_Core.m_ActiveWeapon`, checked above).

    world.cores.core_at_mut(slot).ninja.current_move_time -= 1;

    if world.cores.core_at(slot).ninja.current_move_time == 0 {
        // reset velocity
        let c = world.cores.core_at_mut(slot);
        c.vel = c.ninja.activation_dir * R::from_i32(c.ninja.old_vel_amount);
    }

    if world.cores.core_at(slot).ninja.current_move_time > 0 {
        // Set velocity
        let character = world.characters[id as usize].expect("a ticking character exists");
        let (old_pos, new_pos, new_vel) = {
            let c = world.cores.core_at(slot);
            let vel = c.ninja.activation_dir * R::from_i32(NINJA_VELOCITY);
            // `OldPos = m_Pos`: the entity position, which only `TickDeferred` updates — still the
            // pre-move position here.
            let old_pos = character.pos;
            let tuning = world.tuning.zone(character.tune_zone);
            let ground_elasticity = Vec2::new(tuning.ground_elasticity_x::<R>(), tuning.ground_elasticity_y::<R>());
            let size = Vec2::new(core::physical_size::<R>(), core::physical_size::<R>());
            let (new_pos, new_vel, _grounded) = world.collision.move_box(c.pos, vel, size, ground_elasticity);
            (old_pos, new_pos, new_vel)
        };
        {
            let c = world.cores.core_at_mut(slot);
            c.pos = new_pos;
            c.vel = new_vel;
            // reset velocity so the client doesn't predict stuff
            c.vel = Vec2::zero();
        }

        // check if we Hit anything along the way
        let radius = core::physical_size::<R>() * R::from_f64(2.0);
        let mut targets = std::mem::take(&mut world.range_scratch);
        find_characters_in_range_into(world, old_pos, radius, &mut targets);

        // check that we're not in solo part
        if !world.teams_core.get_solo(id) {
            for &target_id in &targets {
                if target_id == id {
                    continue;
                }
                // Don't hit players in other teams
                if world.teams_core.team(id) != world.teams_core.team(target_id) {
                    continue;
                }
                // Don't hit players in solo parts
                if world.teams_core.get_solo(target_id) {
                    continue;
                }
                // make sure we haven't Hit this object before
                if world.characters[id as usize]
                    .expect("a ticking character exists")
                    .hit_objects[(target_id / 64) as usize]
                    & (1u64 << (target_id % 64))
                    != 0
                {
                    continue;
                }
                // check so we are sufficiently close
                let target_slot = world
                    .cores
                    .slot_of(target_id as u8)
                    .expect("in-range characters are alive");
                let target_entity_pos = world.characters[target_id as usize].expect("alive").pos;
                if vmath::distance(target_entity_pos, old_pos) > radius {
                    continue;
                }
                // Hit a player, give them damage and stuffs... set their velocity to fast upward
                // (for now)
                {
                    let ch = world.characters[id as usize]
                        .as_mut()
                        .expect("a ticking character exists");
                    ch.hit_objects[(target_id / 64) as usize] |= 1u64 << (target_id % 64);
                    ch.num_objects_hit += 1;
                }
                let target_move_restrictions = world.characters[target_id as usize].expect("alive").move_restrictions;
                take_damage(
                    world.cores.core_at_mut(target_slot),
                    Vec2::new(R::ZERO, R::from_f64(-10.0)),
                    target_move_restrictions,
                );
            }
        }
        targets.clear();
        world.range_scratch = targets;
    }
}
