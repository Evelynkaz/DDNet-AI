//! The Oracle-B golden-fixture hash layout, shared by `parity_oracle_b_fixtures.rs` (Stage A, hash
//! per tick over every character's rows) and `parity_oracle_b_stage_b_fixtures.rs` (Stage B, the
//! same plus the entity list). Lives under `tests/common/` so both integration-test crates
//! `#[path]`-include one copy instead of two diverging ones.
#![allow(dead_code)]

use crate::oracle_b_format::{CoreState, DDRaceState};
use ddai_physics::world::{self, World};

/// The single source of truth for "which fields, in which order" a golden-fixture hash covers —
/// see this file's module doc comment. `f32`s go in as their raw `to_le_bytes()` (matching
/// `Trace::tick_hashes()`'s own convention of hashing the exact bit pattern, not a
/// re-normalized value — `NaN`/`-0.0` hash differently from their "equal" counterparts, which is
/// exactly what a bit-exactness fixture wants).
pub fn row_hash_bytes(core: &CoreState, ddrace: &DDRaceState) -> Vec<u8> {
    let mut b = Vec::with_capacity(4 * (10 + 18) + 4 * 54);
    for v in [
        core.pos_x,
        core.pos_y,
        core.vel_x,
        core.vel_y,
        core.hook_pos_x,
        core.hook_pos_y,
        core.hook_dir_x,
        core.hook_dir_y,
        core.hook_tele_base_x,
        core.hook_tele_base_y,
    ] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [
        core.hook_tick,
        core.hook_state,
        core.hooked_player,
        core.active_weapon,
        core.new_hook,
        core.jumped,
        core.jumped_total,
        core.jumps,
        core.direction,
        core.angle,
        core.triggered_events,
        core.colliding,
        core.left_wall,
        core.move_restrictions,
        core.solo,
        core.collision_disabled,
        core.endless_hook,
        core.hook_hit_disabled,
    ] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&ddrace.alive.to_le_bytes());
    b.extend_from_slice(&ddrace.died_this_tick.to_le_bytes());
    b.extend_from_slice(&ddrace.respawned_this_tick.to_le_bytes());
    b.extend_from_slice(&ddrace.freeze_time.to_le_bytes());
    b.extend_from_slice(&ddrace.is_in_freeze.to_le_bytes());
    b.extend_from_slice(&ddrace.deep_frozen.to_le_bytes());
    b.extend_from_slice(&ddrace.live_frozen.to_le_bytes());
    b.extend_from_slice(&ddrace.frozen_last_tick.to_le_bytes());
    b.extend_from_slice(&ddrace.reload_timer.to_le_bytes());
    b.extend_from_slice(&ddrace.attack_tick.to_le_bytes());
    b.extend_from_slice(&ddrace.queued_weapon.to_le_bytes());
    b.extend_from_slice(&ddrace.last_weapon.to_le_bytes());
    b.extend_from_slice(&ddrace.weapon_got_mask.to_le_bytes());
    for v in ddrace.weapon_ammo {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in ddrace.weapon_ammo_regen_start {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&ddrace.ninja_activation_tick.to_le_bytes());
    b.extend_from_slice(&ddrace.ninja_current_move_time.to_le_bytes());
    b.extend_from_slice(&ddrace.ninja_old_vel_amount.to_le_bytes());
    b.extend_from_slice(&ddrace.ninja_activation_dir_x.to_le_bytes());
    b.extend_from_slice(&ddrace.ninja_activation_dir_y.to_le_bytes());
    b.extend_from_slice(&ddrace.tele_checkpoint.to_le_bytes());
    b.extend_from_slice(&ddrace.endless_jump.to_le_bytes());
    b.extend_from_slice(&ddrace.jetpack.to_le_bytes());
    b.extend_from_slice(&ddrace.is_super.to_le_bytes());
    b.extend_from_slice(&ddrace.invincible.to_le_bytes());
    b.extend_from_slice(&ddrace.hammer_hit_disabled.to_le_bytes());
    b.extend_from_slice(&ddrace.grenade_hit_disabled.to_le_bytes());
    b.extend_from_slice(&ddrace.laser_hit_disabled.to_le_bytes());
    b.extend_from_slice(&ddrace.shotgun_hit_disabled.to_le_bytes());
    b.extend_from_slice(&ddrace.has_telegun_gun.to_le_bytes());
    b.extend_from_slice(&ddrace.has_telegun_grenade.to_le_bytes());
    b.extend_from_slice(&ddrace.has_telegun_laser.to_le_bytes());
    b.extend_from_slice(&ddrace.team.to_le_bytes());
    b.extend_from_slice(&ddrace.strong_weak_id.to_le_bytes());
    b.extend_from_slice(&ddrace.freeze_start.to_le_bytes());
    b.extend_from_slice(&ddrace.freeze_end.to_le_bytes());
    b.extend_from_slice(&ddrace.tune_zone.to_le_bytes());
    b.extend_from_slice(&ddrace.num_inputs.to_le_bytes());
    b.extend_from_slice(&ddrace.last_refill_jumps.to_le_bytes());
    b.extend_from_slice(&ddrace.ddrace_state.to_le_bytes());
    b.extend_from_slice(&ddrace.start_time.to_le_bytes());
    b.extend_from_slice(&ddrace.die_tick.to_le_bytes());
    b.extend_from_slice(&ddrace.spawning.to_le_bytes());
    b.extend_from_slice(&ddrace.previous_die_tick.to_le_bytes());
    b
}

/// [`row_hash_bytes`], but reading straight off a live [`World`] instead of a trace-b
/// [`CoreState`]/[`DDRaceState`] pair — this is what makes the verifier able to hash *our own*
/// simulated state with the exact same byte layout the fixture's `tick_hashes` were built from.
/// A dead character's fields are NOT "frozen" the way the reference trace's are (see
/// `compare_character`'s doc comment in `parity_oracle_b.rs`) — none of this crate's fixtures'
/// characters ever die, so this is never exercised, but is written to match the reference's own
/// convention (all zero) rather than panic, in case a future fixture does include a death.
/// `died_this_tick`/`respawned_this_tick` are edge-triggered flags the oracle server derives
/// itself, not stored fields (`oracle_server.cpp`'s `DiedThisTickReal`/`RespawnedThisTickReal` —
/// see `parity_oracle_b.rs`'s `compare_extended`/`DeathTracking` for the exact same formula and
/// citation, duplicated here for the same reason `weapon_variety_bonus` is). `died`/`respawned`
/// are the caller's own already-computed values (this function has no tick-to-tick memory of
/// its own to derive them from).
pub fn world_row_hash_bytes(world: &World<f32>, id: i32, died: bool, respawned: bool) -> Vec<u8> {
    let alive = world.characters[id as usize].is_some_and(|c| c.alive);
    let Some(slot) = world.cores.slot_of(id as u8).filter(|_| alive) else {
        return vec![0u8; 4 * (10 + 18) + 4 * 54];
    };
    let core = world.cores.core_at(slot);
    let character = world.characters[id as usize].unwrap();
    let core_state = CoreState {
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
        left_wall: core.left_wall as i32,
        move_restrictions: core.move_restrictions(),
        solo: core.solo as i32,
        collision_disabled: core.collision_disabled as i32,
        endless_hook: core.endless_hook as i32,
        hook_hit_disabled: core.hook_hit_disabled as i32,
    };
    let got_mask: i32 = (0..6).map(|w| i32::from(core.weapons[w].got) << w).sum();
    let mut weapon_ammo = [0i32; 6];
    let mut weapon_ammo_regen_start = [0i32; 6];
    for w in 0..6 {
        weapon_ammo[w] = core.weapons[w].ammo;
        weapon_ammo_regen_start[w] = core.weapons[w].ammo_regen_start;
    }
    let die_tick = world.players[id as usize].map(|p| p.die_tick).unwrap_or(0);
    let spawning = world.players[id as usize].is_some_and(|p| p.spawning);
    let previous_die_tick = world.players[id as usize].map(|p| p.previous_die_tick).unwrap_or(0);
    let ddrace_state = DDRaceState {
        alive: 1,
        died_this_tick: died as i32,
        respawned_this_tick: respawned as i32,
        freeze_time: character.freeze_time,
        is_in_freeze: core.is_in_freeze as i32,
        deep_frozen: core.deep_frozen as i32,
        live_frozen: core.live_frozen as i32,
        frozen_last_tick: character.frozen_last_tick as i32,
        reload_timer: character.reload_timer,
        attack_tick: character.attack_tick,
        queued_weapon: character.queued_weapon,
        last_weapon: character.last_weapon,
        weapon_got_mask: got_mask,
        weapon_ammo,
        weapon_ammo_regen_start,
        ninja_activation_tick: core.ninja.activation_tick,
        ninja_current_move_time: core.ninja.current_move_time,
        ninja_old_vel_amount: core.ninja.old_vel_amount,
        ninja_activation_dir_x: core.ninja.activation_dir.x,
        ninja_activation_dir_y: core.ninja.activation_dir.y,
        tele_checkpoint: character.tele_checkpoint,
        endless_jump: core.endless_jump as i32,
        jetpack: core.jetpack as i32,
        is_super: core.is_super as i32,
        invincible: core.invincible as i32,
        hammer_hit_disabled: core.hammer_hit_disabled as i32,
        grenade_hit_disabled: core.grenade_hit_disabled as i32,
        laser_hit_disabled: core.laser_hit_disabled as i32,
        shotgun_hit_disabled: core.shotgun_hit_disabled as i32,
        has_telegun_gun: core.has_telegun_gun as i32,
        has_telegun_grenade: core.has_telegun_grenade as i32,
        has_telegun_laser: core.has_telegun_laser as i32,
        team: world::character_team(&world.teams_core, id),
        strong_weak_id: character.strong_weak_id,
        freeze_start: core.freeze_start,
        freeze_end: core.freeze_end,
        tune_zone: character.tune_zone,
        num_inputs: character.num_inputs,
        last_refill_jumps: character.last_refill_jumps as i32,
        ddrace_state: character.ddrace_state,
        start_time: character.start_time,
        die_tick,
        spawning: spawning as i32,
        previous_die_tick,
    };
    row_hash_bytes(&core_state, &ddrace_state)
}

/// Appends one trace-b `EntityRecord`-shaped record (9 little-endian words, `docs/formats.md` §11.2):
/// `kind, owner, weapon_type, pos_x, pos_y, dir_x, dir_y, start_tick, extra`.
#[allow(clippy::too_many_arguments)]
fn push_record(
    b: &mut Vec<u8>,
    kind: i32,
    owner: i32,
    weapon: i32,
    pos: [u32; 2],
    dir: [u32; 2],
    start: i32,
    extra: i32,
) {
    for w in [
        kind as u32,
        owner as u32,
        weapon as u32,
        pos[0],
        pos[1],
        dir[0],
        dir[1],
        start as u32,
        extra as u32,
    ] {
        b.extend_from_slice(&w.to_le_bytes());
    }
}

/// The entity part of a stage-B golden-fixture tick hash, built from *our* world: projectiles in list
/// order, then the dynamic `ENTTYPE_LASER` entities (lasers, beams, plasma) head first, then the
/// static fixtures grouped by dump kind (doors, draggers, turrets, lights) in list order — the
/// canonical order [`trace_entity_hash_bytes`] reproduces from a reference tick (the cross-kind
/// interleaving of the statics in the real list is a map-scan artefact, see `compare_laser_list`
/// in `parity_oracle_b.rs`).
pub fn world_entity_hash_bytes(world: &World<f32>) -> Vec<u8> {
    use ddai_physics::world::{Fixture, LaserSlot};
    let bits = |v: f32| v.to_bits();
    let mut b = Vec::new();
    for p in &world.projectiles {
        push_record(
            &mut b,
            0,
            p.owner,
            p.weapon_type,
            [bits(p.pos.x), bits(p.pos.y)],
            [bits(p.direction.x), bits(p.direction.y)],
            p.start_tick,
            p.life_span,
        );
    }
    for slot in world.lasers.iter().rev() {
        match slot {
            LaserSlot::Laser(l) => push_record(
                &mut b,
                1,
                l.owner,
                l.weapon_type,
                [bits(l.pos.x), bits(l.pos.y)],
                [bits(l.dir.x), bits(l.dir.y)],
                l.eval_tick,
                l.bounces,
            ),
            LaserSlot::Beam(e) => push_record(&mut b, 4, e.for_client, 0, [bits(e.pos.x), bits(e.pos.y)], [0, 0], 0, 0),
            LaserSlot::Plasma(p) => push_record(
                &mut b,
                7,
                p.for_client,
                i32::from(p.explosive) | (i32::from(p.freeze) << 1),
                [bits(p.pos.x), bits(p.pos.y)],
                [bits(p.core.x), bits(p.core.y)],
                p.eval_tick,
                p.life_time,
            ),
        }
    }
    for d in world.doors.iter() {
        push_record(&mut b, 2, -1, 0, [bits(d.pos.x), bits(d.pos.y)], [0, 0], 0, 0);
    }
    for kind in [3, 5, 6] {
        for f in world.fixtures.iter().filter(|f| f.dump_kind() == kind) {
            let pos = f.pos();
            match f {
                Fixture::Light(l) => push_record(
                    &mut b,
                    6,
                    -1,
                    l.length,
                    [bits(pos.x), bits(pos.y)],
                    [bits(l.angular_speed), 0],
                    0,
                    l.speed,
                ),
                _ => push_record(&mut b, kind, -1, 0, [bits(pos.x), bits(pos.y)], [0, 0], 0, 0),
            }
        }
    }
    b
}

/// [`world_entity_hash_bytes`]' counterpart for a reference tick: the same canonical order, the
/// records' own fields (the dump already holds them in the 9-word layout).
pub fn trace_entity_hash_bytes(tick: &crate::oracle_b_format::TraceBTick) -> Vec<u8> {
    let mut b = Vec::new();
    for kind_group in [&[0][..], &[1, 4, 7][..], &[2][..], &[3][..], &[5][..], &[6][..]] {
        for e in tick.entities.iter().filter(|e| kind_group.contains(&e.kind)) {
            push_record(
                &mut b,
                e.kind,
                e.owner_client_id,
                e.weapon_type,
                [e.pos_x.to_bits(), e.pos_y.to_bits()],
                [e.dir_x.to_bits(), e.dir_y.to_bits()],
                e.start_tick,
                e.extra,
            );
        }
    }
    b
}
