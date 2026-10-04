// Ported from DDNet 20.1 `src/game/server/entities/laser.cpp` (`CLaser`) and
// `src/game/server/interactions.cpp` (`CInteractions::CanHit`). DDNet's zlib-style license notice
// for the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real`, no `unsafe`, `CEntity`'s
// intrusive linked list replaced by [`World::lasers`] (see its doc comment), cosmetic parts
// (`CreateSound`, `Snap`) dropped.

//! `CLaser`: the laser rifle *and* the shotgun (DDNet 20.1's shotgun fires a `CLaser` of
//! `WEAPON_SHOTGUN` type, not a projectile), plus the entity list they share with plasma bullets
//! and dragger beams ([`LaserSlot`], DDNet's `ENTTYPE_LASER` slot minus the static map fixtures,
//! which live in [`World::fixtures`]).

use super::{
    Character, World, get_nearest_air_pos, get_nearest_air_pos_player, intersect_character_ex, take_damage, unfreeze,
};
use crate::core::{self, CharacterCore, MAX_CLIENTS, WEAPON_LASER, WEAPON_SHOTGUN};
use crate::map;
use crate::real::Real;
use crate::vmath::{self, Vec2};

use super::fixtures::{DraggerBeam, Plasma};

/// `CCharacter::m_Pos`: the entity position (see [`Character::pos`]).
fn char_pos<R: Real>(world: &World<R>, id: i32) -> Vec2<R> {
    world.characters[id as usize].expect("an alive character exists").pos
}

/// One entry of [`World::lasers`]: the *dynamic* members of DDNet's `ENTTYPE_LASER` list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaserSlot<R: Real> {
    /// `CLaser` (dump kind 1).
    Laser(Laser<R>),
    /// `CDraggerBeam` (dump kind 4).
    Beam(DraggerBeam<R>),
    /// `CPlasma` (dump kind 7).
    Plasma(Plasma<R>),
}

impl<R: Real> LaserSlot<R> {
    /// `m_MarkedForDestroy`.
    pub fn marked_for_destroy(&self) -> bool {
        match self {
            LaserSlot::Laser(l) => l.marked_for_destroy,
            LaserSlot::Beam(b) => b.marked_for_destroy,
            LaserSlot::Plasma(p) => p.marked_for_destroy,
        }
    }
}

/// [`World::lasers`]: a `Vec<LaserSlot>` whose capacity is reserved once ([`super::LASER_CAPACITY`])
/// and *stays* reserved across `Clone` — a plain `Vec::clone` shrinks to the length, so the first
/// shot after every `World::clone()` would allocate; this clone keeps at least the reservation, and
/// `clone_from` (what [`World::restore_from`] uses) reuses the destination's buffer. `Deref`s to the
/// `Vec`, so every slice/`Vec` method (`push`, `iter`, `retain`, indexing, ...) works unchanged.
#[derive(Debug, PartialEq)]
pub struct LaserList<R: Real>(Vec<LaserSlot<R>>);

impl<R: Real> LaserList<R> {
    /// An empty list with the standard reservation.
    pub fn new() -> Self {
        LaserList(Vec::with_capacity(super::LASER_CAPACITY))
    }
}

impl<R: Real> Default for LaserList<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Real> Clone for LaserList<R> {
    fn clone(&self) -> Self {
        let mut v = Vec::with_capacity(self.0.len().max(super::LASER_CAPACITY));
        v.extend_from_slice(&self.0);
        LaserList(v)
    }

    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0);
    }
}

impl<'a, R: Real> IntoIterator for &'a LaserList<R> {
    type Item = &'a LaserSlot<R>;
    type IntoIter = std::slice::Iter<'a, LaserSlot<R>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<R: Real> std::ops::Deref for LaserList<R> {
    type Target = Vec<LaserSlot<R>>;
    fn deref(&self) -> &Vec<LaserSlot<R>> {
        &self.0
    }
}

impl<R: Real> std::ops::DerefMut for LaserList<R> {
    fn deref_mut(&mut self) -> &mut Vec<LaserSlot<R>> {
        &mut self.0
    }
}

/// `CLaser` (`laser.h`). Interaction state (`CInteractions m_InteractState`) is not stored: every
/// `SyncInteractState()` call in the C++ source (constructor, `HitCharacter`, `Tick`) refills it
/// from the owner's *current* state right before the only reader (`CanHit`, inside
/// `HitCharacter`, which syncs first), so evaluating it on demand (`can_hit`) is equivalent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Laser<R: Real> {
    /// `m_Pos`.
    pub pos: Vec2<R>,
    /// `m_From`.
    pub from: Vec2<R>,
    /// `m_Dir`.
    pub dir: Vec2<R>,
    /// `m_TelePos`.
    pub tele_pos: Vec2<R>,
    /// `m_WasTele`.
    pub was_tele: bool,
    /// `m_Energy`.
    pub energy: R,
    /// `m_Bounces`.
    pub bounces: i32,
    /// `m_EvalTick`.
    pub eval_tick: i32,
    /// `m_Owner`.
    pub owner: i32,
    /// `m_ZeroEnergyBounceInLastTick`.
    pub zero_energy_bounce_in_last_tick: bool,
    /// `m_PrevPos`.
    pub prev_pos: Vec2<R>,
    /// `m_Type` (`WEAPON_LASER` or `WEAPON_SHOTGUN`).
    pub weapon_type: i32,
    /// `m_TuneZone`, fixed at creation.
    pub tune_zone: i32,
    /// `m_TeleportCancelled`.
    pub teleport_cancelled: bool,
    /// `m_IsBlueTeleport`.
    pub is_blue_teleport: bool,
    /// `m_MarkedForDestroy`.
    pub marked_for_destroy: bool,
}

/// `static const vec2 StackedLaserShotgunBugSpeed` (`laser.cpp:47`).
fn stacked_laser_shotgun_bug_speed<R: Real>() -> Vec2<R> {
    Vec2::new(R::from_f64(-2147483648.0), R::from_f64(-2147483648.0))
}

/// `GetPlayerChar(id)` (`gamecontext.cpp:242`): the character, if its player exists and it is
/// alive (`CPlayer::GetCharacter`, `player.cpp:698`).
fn alive_char<R: Real>(world: &World<R>, id: i32) -> Option<&Character<R>> {
    if !(0..MAX_CLIENTS as i32).contains(&id) {
        return None;
    }
    world.characters[id as usize].as_ref().filter(|c| c.alive)
}

/// `CInteractions::CanHit(pServer, hit_id)` (`interactions.cpp:99-114`) with the state
/// `SyncInteractState()` (`laser.cpp:304-329`) would have just filled in (see [`Laser`]'s doc
/// comment). `hit_id` is the hit character's client id; `CPlayer::GetUniqueCid()` equality is
/// client-id equality in this world (players are never replaced, so unique ids never diverge from
/// client ids).
fn can_hit<R: Real>(world: &World<R>, l: &Laser<R>, hit_id: i32) -> bool {
    let owner_player = (0..MAX_CLIENTS as i32).contains(&l.owner) && world.players[l.owner as usize].is_some();
    let owner_alive = alive_char(world, l.owner).is_some();
    let (ddrace_team, solo, no_hit_others) = if owner_player {
        let owner_core = if owner_alive {
            world.cores.get(l.owner as u8)
        } else {
            None
        };
        // `bool NoHitOthers = g_Config.m_SvHit; if(pOwnerChar) NoHitOthers = ...` (`laser.cpp:314-316`)
        let no_hit_others = match owner_core {
            Some(c) => {
                (l.weapon_type == WEAPON_LASER && c.laser_hit_disabled)
                    || (l.weapon_type == WEAPON_SHOTGUN && c.shotgun_hit_disabled)
            }
            None => world.config.sv_hit,
        };
        (
            world.teams_core.team(l.owner),
            owner_core.is_some_and(|c| c.solo),
            no_hit_others,
        )
    } else {
        // `FillOwnerDisconnected()` is unreachable here (players are never removed); the
        // constructor's zero-initialized state is the closest faithful reading.
        (0, false, false)
    };
    let no_hit_self = world.config.sv_old_laser || (l.bounces == 0 && !l.was_tele);
    let same = hit_id == l.owner;
    if ddrace_team != 0 && world.teams_core.team(hit_id) != ddrace_team {
        return false;
    }
    if solo && !same {
        return false;
    }
    if no_hit_others && !same {
        return false;
    }
    if no_hit_self && same {
        return false;
    }
    true
}

/// `CLaser::CLaser(...)` (`laser.cpp:17-43`): creates the entity, inserts it at the head of the
/// list and runs the first `DoBounce()` right away (the shot's first segment is traced within
/// the very tick it is fired).
pub fn laser_new<R: Real>(
    world: &mut World<R>,
    pos: Vec2<R>,
    dir: Vec2<R>,
    start_energy: R,
    owner: i32,
    weapon_type: i32,
) {
    let tune_zone = world.collision.is_tune(world.collision.get_map_index(pos));
    world.lasers.push(LaserSlot::Laser(Laser {
        pos,
        from: Vec2::zero(),
        dir,
        tele_pos: Vec2::zero(),
        was_tele: false,
        energy: start_energy,
        bounces: 0,
        eval_tick: 0,
        owner,
        zero_energy_bounce_in_last_tick: false,
        prev_pos: Vec2::zero(),
        weapon_type,
        tune_zone,
        teleport_cancelled: false,
        is_blue_teleport: false,
        marked_for_destroy: false,
    }));
    let index = world.lasers.len() - 1;
    laser_do_bounce(world, index);
}

fn laser_at<R: Real>(world: &World<R>, index: usize) -> Laser<R> {
    match world.lasers[index] {
        LaserSlot::Laser(l) => l,
        _ => unreachable!("laser_* called with a non-laser slot"),
    }
}

/// `CLaser::HitCharacter(vec2 From, vec2 To)` (`laser.cpp:45-102`). `l` is the laser's working
/// copy ([`laser_do_bounce`] writes it back); returns whether a character was hit.
fn laser_hit_character<R: Real>(world: &mut World<R>, l: &mut Laser<R>, from: Vec2<R>, to: Vec2<R>) -> bool {
    let owner_alive = alive_char(world, l.owner).is_some();
    let owner_core = if owner_alive {
        world.cores.get(l.owner as u8).copied()
    } else {
        None
    };
    let dont_hit_self = world.config.sv_old_laser || (l.bounces == 0 && !l.was_tele);
    let not_this = if dont_hit_self && owner_alive { l.owner } else { -1 };
    let hit_enabled = match owner_core {
        Some(c) => {
            (!c.laser_hit_disabled && l.weapon_type == WEAPON_LASER)
                || (!c.shotgun_hit_disabled && l.weapon_type == WEAPON_SHOTGUN)
        }
        None => world.config.sv_hit,
    };
    let this_only = if hit_enabled || !owner_alive { -1 } else { l.owner };
    let Some((hit_id, at)) = intersect_character_ex(world, l.pos, to, R::ZERO, not_this, l.owner, this_only) else {
        return false;
    };
    if !can_hit(world, l, hit_id) {
        return false;
    }
    l.from = from;
    l.pos = at;
    l.energy = -R::ONE;
    let hit_slot = world.cores.slot_of(hit_id as u8).expect("a hit character has a core");
    let hit_move_restrictions = world.characters[hit_id as usize]
        .expect("a hit character exists")
        .move_restrictions;
    if l.weapon_type == WEAPON_SHOTGUN {
        let strength: R = world.tuning.zone(l.tune_zone).shotgun_strength();
        let hit_pos = world.cores.core_at(hit_slot).pos;
        if !world.config.sv_old_laser {
            if l.prev_pos != hit_pos {
                let add = vmath::normalize(l.prev_pos - hit_pos) * strength;
                add_velocity(world.cores.core_at_mut(hit_slot), add, hit_move_restrictions);
            } else {
                world.cores.core_at_mut(hit_slot).vel = stacked_laser_shotgun_bug_speed();
            }
        } else if let Some(owner_core) = owner_core {
            if owner_core.pos != hit_pos {
                let add = vmath::normalize(owner_core.pos - hit_pos) * strength;
                add_velocity(world.cores.core_at_mut(hit_slot), add, hit_move_restrictions);
            } else {
                world.cores.core_at_mut(hit_slot).vel = stacked_laser_shotgun_bug_speed();
            }
        } else {
            // "Re-apply move restrictions as a part of 'shotgun bug' reproduction"
            super::apply_move_restrictions(world.cores.core_at_mut(hit_slot), hit_move_restrictions);
        }
    } else if l.weapon_type == WEAPON_LASER {
        let mut hit_char = world.characters[hit_id as usize].expect("a hit character exists");
        let mut hit_core = *world.cores.core_at(hit_slot);
        unfreeze(&mut hit_char, &mut hit_core);
        world.characters[hit_id as usize] = Some(hit_char);
        *world.cores.core_at_mut(hit_slot) = hit_core;
    }
    take_damage(world.cores.core_at_mut(hit_slot), Vec2::zero(), hit_move_restrictions);
    true
}

/// `CCharacter::AddVelocity(vec2 Addition)` (`character.cpp:2605-2608`): `SetVelocity(m_Core.m_Vel
/// + Addition)`, i.e. `ClampVel(m_MoveRestrictions, ...)` with the character-level field.
pub fn add_velocity<R: Real>(core: &mut CharacterCore<R>, addition: Vec2<R>, move_restrictions: i32) {
    core.vel = crate::collision::clamp_vel(move_restrictions, core.vel + addition);
}

/// `CLaser::DoBounce()` (`laser.cpp:104-255`).
pub fn laser_do_bounce<R: Real>(world: &mut World<R>, index: usize) {
    let mut l = laser_at(world, index);
    l.eval_tick = world.tick;

    if l.energy < R::ZERO {
        l.marked_for_destroy = true;
        world.lasers[index] = LaserSlot::Laser(l);
        return;
    }
    l.prev_pos = l.pos;

    if l.was_tele {
        l.prev_pos = l.tele_pos;
        l.pos = l.tele_pos;
        l.tele_pos = Vec2::zero();
    }

    // `IntersectLineTeleWeapon(m_Pos, To, &Coltile, &To, &z)`: `To` is the *before-collision*
    // out-parameter (or the original end point when nothing is hit).
    let hit = world.collision.intersect_line_tele_weapon(
        l.pos,
        l.pos + l.dir * l.energy,
        world.config.sv_old_teleport_weapons,
    );
    let res = hit.hit;
    let coltile = hit.collision;
    let to = hit.before_collision;
    let z = hit.tele_nr;

    let from = l.pos;
    if res != 0 {
        if !laser_hit_character(world, &mut l, from, to) {
            // intersected
            l.from = l.pos;
            l.pos = to;

            let temp_pos = l.pos;
            let temp_dir = l.dir * R::from_f64(4.0);

            // `Res == -1` (an `IntersectAir` result) cannot come out of `IntersectLineTeleWeapon`,
            // so the C++ source's temporary `SetCollisionAt(.., TILE_SOLID)`/restore around
            // `MovePoint` (`laser.cpp:142-151`) is dead code here and is not ported (which also
            // keeps `World::collision` immutable and shareable).
            let (new_pos, new_dir, _) = world.collision.move_point(temp_pos, temp_dir, R::ONE);
            l.pos = new_pos;
            l.dir = vmath::normalize(new_dir);

            let distance = vmath::distance(l.from, l.pos);
            // Prevent infinite bounces
            if distance == R::ZERO && l.zero_energy_bounce_in_last_tick {
                l.energy = -R::ONE;
            } else {
                let cost: R = world.tuning.zone(l.tune_zone).laser_bounce_cost();
                l.energy -= distance + cost;
            }
            l.zero_energy_bounce_in_last_tick = distance == R::ZERO;

            if res == i32::from(map::TILE_TELEINWEAPON) && !world.collision.tele_outs((z - 1) as u8).is_empty() {
                let outs_len = world.collision.tele_outs((z - 1) as u8).len();
                let tele_out = world.cores.random_or_0(outs_len as i32);
                l.tele_pos = world.collision.tele_outs((z - 1) as u8)[tele_out as usize];
                l.was_tele = true;
            } else {
                l.bounces += 1;
                l.was_tele = false;
            }

            let bounce_num = world.tuning.zone(l.tune_zone).laser_bounce_num();
            if l.bounces > bounce_num {
                l.energy = -R::ONE;
            }
        }
    } else if !laser_hit_character(world, &mut l, from, to) {
        l.from = l.pos;
        l.pos = to;
        l.energy = -R::ONE;
    }

    // The owner's state can have been changed by the hit above (never its own death: a laser does
    // not damage), so it is re-read here exactly like `laser.cpp:197`'s fresh `GetPlayerChar`.
    let owner_alive = alive_char(world, l.owner).is_some();
    let owner_core = if owner_alive {
        world.cores.get(l.owner as u8).copied()
    } else {
        None
    };
    if l.owner >= 0
        && l.energy <= R::ZERO
        && !l.teleport_cancelled
        && owner_core.is_some_and(|c| c.has_telegun_laser)
        && l.weapon_type == WEAPON_LASER
    {
        let owner_core = owner_core.expect("checked above");
        let dont_hit_self = world.config.sv_old_laser || (l.bounces == 0 && !l.was_tele);
        let not_this = if dont_hit_self { l.owner } else { -1 };
        // `pOwnerChar ? (!LaserHitDisabled && Type == LASER) : SvHit` (`laser.cpp:208`)
        let hit_enabled = !owner_core.laser_hit_disabled;
        let this_only = if hit_enabled { -1 } else { l.owner };
        let hit = intersect_character_ex(world, l.pos, to, R::ZERO, not_this, l.owner, this_only);
        let found = match hit {
            Some((hit_id, _)) => get_nearest_air_pos_player(&world.collision, char_pos(world, hit_id)),
            None => get_nearest_air_pos(&world.collision, l.pos, l.from),
        };
        if let Some(possible_pos) = found {
            let owner = world.characters[l.owner as usize].as_mut().expect("alive owner");
            owner.tele_gun_pos = possible_pos;
            owner.tele_gun_teleport = true;
            owner.is_blue_tele_gun_teleport = l.is_blue_teleport;
        }
    } else if l.owner >= 0 {
        let map_index = world.collision.get_pure_map_index_vec(coltile) as i32;
        let tile_f_index = world.collision.get_front_tile_index(map_index);
        let mut is_switch_tele_gun = world.collision.get_switch_type(map_index) == i32::from(map::TILE_ALLOW_TELE_GUN);
        let mut is_blue_switch_tele_gun =
            world.collision.get_switch_type(map_index) == i32::from(map::TILE_ALLOW_BLUE_TELE_GUN);
        let is_tele_in_weapon = world.collision.is_teleport_weapon(map_index);

        if is_tele_in_weapon == 0 {
            if is_switch_tele_gun || is_blue_switch_tele_gun {
                // Delay specifies which weapon the tile should work for. Delay = 0 means all.
                let delay = world.collision.get_switch_delay(map_index);
                if (delay != 3 && delay != 0) && l.weapon_type == WEAPON_LASER {
                    is_switch_tele_gun = false;
                    is_blue_switch_tele_gun = false;
                }
            }

            l.is_blue_teleport = tile_f_index == i32::from(map::TILE_ALLOW_BLUE_TELE_GUN) || is_blue_switch_tele_gun;

            // Teleport is canceled if the last bounce tile is not a TILE_ALLOW_TELE_GUN.
            // Teleport also works if laser didn't bounce.
            l.teleport_cancelled = l.weapon_type == WEAPON_LASER
                && (tile_f_index != i32::from(map::TILE_ALLOW_TELE_GUN)
                    && tile_f_index != i32::from(map::TILE_ALLOW_BLUE_TELE_GUN)
                    && !is_switch_tele_gun
                    && !is_blue_switch_tele_gun);
        }
    }

    world.lasers[index] = LaserSlot::Laser(l);
}

/// `CLaser::Tick()` (`laser.cpp:262-277`).
pub fn laser_tick<R: Real>(world: &mut World<R>, index: usize) {
    let l = laser_at(world, index);
    // `(g_Config.m_SvDestroyLasersOnDeath || m_BelongsToPracticeTeam) && m_Owner >= 0`; practice
    // mode is unreachable in this world (see the module doc comment of `world`).
    if world.config.sv_destroy_lasers_on_death && l.owner >= 0 && alive_char(world, l.owner).is_none() {
        // `Reset()`
        if let LaserSlot::Laser(l) = &mut world.lasers[index] {
            l.marked_for_destroy = true;
        }
    }

    let delay: R = world.tuning.zone(l.tune_zone).laser_bounce_delay();
    // `(Server()->Tick() - m_EvalTick) > (Server()->TickSpeed() * Delay / 1000.0f)`: the `int` on
    // the left converts to `float`; the right side is `float` throughout.
    if R::from_i32(world.tick - l.eval_tick) > R::from_i32(core::SERVER_TICK_SPEED) * delay / R::from_f64(1000.0) {
        laser_do_bounce(world, index);
    }
}

/// `CGameWorld::RemoveEntitiesFromPlayer` (`gameworld.cpp:160-184`) for lasers: `CLaser::
/// GetOwnerId()` is `m_Owner` (plasma bullets and dragger beams don't override it). The C++ call
/// removes the entities **at once** (`RemoveEntity` + `Destroy`), not at the end of the tick, so a
/// removed laser neither ticks again in the same step nor shows up in a dump taken after the
/// change; it is only ever called from the character pass, `player_tick` or spawn bookkeeping —
/// never from inside the laser pass — so removing in place cannot invalidate an index in use.
pub fn remove_lasers_of_player<R: Real>(world: &mut World<R>, id: i32) {
    if world.lasers.is_empty() {
        return;
    }
    world
        .lasers
        .retain(|slot| !matches!(slot, LaserSlot::Laser(l) if l.owner == id));
}
