// Ported from DDNet 20.1 `src/game/server/entities/{dragger,dragger_beam,gun,plasma,light}.cpp`.
// DDNet's zlib-style license notices for the ported logic:
//
//   /* (c) Shereef Marzouk. See "licence DDRace.txt" and the readme.txt in the root of the     */
//   /* distribution for more information.                                                        */
//   /* copyright (c) 2007 magnus auvinen, see licence.txt for more info */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real`, no `unsafe`, `CEntity`'s
// intrusive linked list replaced by [`World::fixtures`]/[`World::lasers`] (see their doc
// comments), the per-dragger/per-turret `MAX_CLIENTS`-sized arrays moved into side pools so the
// entity itself stays small, cosmetic parts (`Snap`, `CreateSound`) dropped.

//! The map-fixture half of DDNet's `ENTTYPE_LASER` slot: draggers (`CDragger`), turrets (`CGun`)
//! and lights (`CLight`) — static entities, created once by the map scan and ticking in their
//! creation order (newest first) *behind* every dynamic entity — plus the dynamic entities the
//! first two spawn: dragger beams (`CDraggerBeam`) and turret shots (`CPlasma`).
//!
//! `CDoor` is the sixth class sharing the slot; it never ticks and lives in [`World::doors`].

use super::laser::{LaserSlot, add_velocity};
use super::{
    Layer, RANGE_PREFILTER_SLACK, World, character_team, could_be_near_alive_characters_bbox, create_explosion,
    find_characters_in_range_into, freeze_default, intersect_character_ex, unfreeze,
};
use crate::core::{self, MAX_CLIENTS, TEAM_SUPER, WEAPON_GRENADE};
use crate::real::Real;
use crate::vmath::{self, Vec2};

/// `(int)(Server()->TickSpeed() * 0.15f)` (`dragger.cpp:38`, `gun.cpp:35`, `light.cpp:23,88`,
/// `pickup.cpp`): `50 * 0.15f = 7.5000003f`, truncated.
pub const MOVER_PERIOD: i32 = 7;

/// `CDragger` (`dragger.h`), minus the two `MAX_CLIENTS`-sized arrays (see
/// [`DraggerState`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dragger<R: Real> {
    /// `m_Pos`.
    pub pos: Vec2<R>,
    /// `m_Core`: the mover-tile velocity added to `m_Pos` every [`MOVER_PERIOD`] ticks (persists
    /// once set, see `world::Pickup::mcore`).
    pub core: Vec2<R>,
    /// `m_Strength` (`1.0`, `2.0` or `3.0`).
    pub strength: R,
    /// `m_IgnoreWalls`.
    pub ignore_walls: bool,
    /// `m_Layer`.
    pub layer: Layer,
    /// `m_Number`.
    pub number: i32,
    /// Index into [`World::dragger_states`].
    pub state: u16,
}

/// `CDragger::m_aTargetIdInTeam`/`m_apDraggerBeam` (`dragger.h`). A beam pointer is a `bool` per
/// client here (the beam entity itself holds the back-reference, see [`DraggerBeam::dragger`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraggerState {
    /// `m_aTargetIdInTeam`, indexed by DDRace team; `-1` = none (client ids fit an `i8`).
    pub target_in_team: [i8; MAX_CLIENTS],
    /// Bit `c` set <=> `m_apDraggerBeam[c] != nullptr`.
    pub beams: u128,
}

impl Default for DraggerState {
    fn default() -> Self {
        DraggerState {
            target_in_team: [-1; MAX_CLIENTS],
            beams: 0,
        }
    }
}

/// `CDraggerBeam` (`dragger_beam.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraggerBeam<R: Real> {
    /// `m_Pos` (kept in sync with the dragger's by `CDragger::Tick`'s `SetPos`).
    pub pos: Vec2<R>,
    /// `m_Strength`.
    pub strength: R,
    /// `m_IgnoreWalls`.
    pub ignore_walls: bool,
    /// `m_ForClientId`.
    pub for_client: i32,
    /// `m_Active`.
    pub active: bool,
    /// `m_Layer`.
    pub layer: Layer,
    /// `m_Number`.
    pub number: i32,
    /// `m_pDragger`: index into [`World::fixtures`].
    pub dragger: u16,
    /// `m_MarkedForDestroy`.
    pub marked_for_destroy: bool,
}

/// `CGun` (`gun.h`), minus the two `MAX_CLIENTS`-sized arrays (see [`GunState`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gun<R: Real> {
    /// `m_Pos`.
    pub pos: Vec2<R>,
    /// `m_Core`: the mover-tile velocity.
    pub core: Vec2<R>,
    /// `m_Freeze`.
    pub freeze: bool,
    /// `m_Explosive`.
    pub explosive: bool,
    /// `m_Layer`.
    pub layer: Layer,
    /// `m_Number`.
    pub number: i32,
    /// Index into [`World::gun_states`].
    pub state: u16,
}

/// `CGun::m_aLastFireTeam`/`m_aLastFireSolo` (`gun.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GunState {
    /// `m_aLastFireTeam`, indexed by DDRace team.
    pub last_fire_team: [i32; MAX_CLIENTS],
    /// `m_aLastFireSolo`, indexed by client id.
    pub last_fire_solo: [i32; MAX_CLIENTS],
}

impl Default for GunState {
    fn default() -> Self {
        GunState {
            last_fire_team: [0; MAX_CLIENTS],
            last_fire_solo: [0; MAX_CLIENTS],
        }
    }
}

/// `CPlasma` (`plasma.h`): a turret shot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plasma<R: Real> {
    /// `m_Pos`.
    pub pos: Vec2<R>,
    /// `m_Core`: the per-tick displacement (accelerates by `PLASMA_ACCEL`, 1.1 per tick).
    pub core: Vec2<R>,
    /// `m_Freeze` (`true` = freeze on hit, `false` = unfreeze).
    pub freeze: bool,
    /// `m_Explosive`.
    pub explosive: bool,
    /// `m_ForClientId`: the targeted client.
    pub for_client: i32,
    /// `m_EvalTick`.
    pub eval_tick: i32,
    /// `m_LifeTime`.
    pub life_time: i32,
    /// `m_MarkedForDestroy`.
    pub marked_for_destroy: bool,
}

/// `PLASMA_ACCEL` (`plasma.cpp:14`).
const PLASMA_ACCEL: f32 = 1.1;

/// `CLight` (`light.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light<R: Real> {
    /// `m_Pos`.
    pub pos: Vec2<R>,
    /// `m_Core`: the mover-tile velocity.
    pub core: Vec2<R>,
    /// `m_Rotation`.
    pub rotation: R,
    /// `m_To`: the beam's far end, recomputed by `Step()`.
    pub to: Vec2<R>,
    /// `m_Layer`.
    pub layer: Layer,
    /// `m_Number`.
    pub number: i32,
    /// `m_CurveLength`.
    pub curve_length: i32,
    /// `m_LengthL`.
    pub length_l: i32,
    /// `m_AngularSpeed`.
    pub angular_speed: R,
    /// `m_Speed`.
    pub speed: i32,
    /// `m_Length`.
    pub length: i32,
}

/// One static fixture of the `ENTTYPE_LASER` list that ticks (see the module doc comment).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fixture<R: Real> {
    /// `CDragger` (dump kind 3).
    Dragger(Dragger<R>),
    /// `CGun` (dump kind 5).
    Gun(Gun<R>),
    /// `CLight` (dump kind 6).
    Light(Light<R>),
}

impl<R: Real> Fixture<R> {
    /// `m_Pos`.
    pub fn pos(&self) -> Vec2<R> {
        match self {
            Fixture::Dragger(d) => d.pos,
            Fixture::Gun(g) => g.pos,
            Fixture::Light(l) => l.pos,
        }
    }

    /// The trace-b entity-dump kind (`docs/formats.md` §11.2): 3, 5 or 6.
    pub fn dump_kind(&self) -> i32 {
        match self {
            Fixture::Dragger(_) => 3,
            Fixture::Gun(_) => 5,
            Fixture::Light(_) => 6,
        }
    }
}

/// `CLight::CLight(...)` (`light.cpp:15-30`) followed by the `OnEntity` assignments
/// (`gamecontroller.cpp:328-343`) the map scan makes right after construction. `to == pos`: the
/// constructor's own `Step()` runs with `m_CurveLength == 0` (zero-initialized, the controller
/// sets it only afterwards), whose beam is the empty segment `m_Pos..m_Pos`.
#[allow(clippy::too_many_arguments)]
pub fn new_light<R: Real>(
    pos: Vec2<R>,
    rotation: R,
    length: i32,
    layer: Layer,
    number: i32,
    angular_speed: R,
    speed: i32,
    curve_length: i32,
) -> Light<R> {
    Light {
        pos,
        core: Vec2::zero(),
        rotation,
        to: pos,
        layer,
        number,
        curve_length,
        length_l: 0,
        angular_speed,
        speed,
        length,
    }
}

fn switch_status<R: Real>(world: &World<R>, number: i32, team: i32) -> bool {
    world
        .cores
        .switchers
        .get(number as usize)
        .is_some_and(|s| s.status[team as usize])
}

/// `m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[Team]`: the
/// entity is switched off for `team`.
fn switched_off_for<R: Real>(world: &World<R>, layer: Layer, number: i32, team: i32) -> bool {
    layer == Layer::Switch && number > 0 && !switch_status(world, number, team)
}

/// The `Collision()->IntersectNoLaser[NoWalls](m_Pos, target, nullptr, nullptr)` call both
/// `CDragger::LookForPlayersToDrag` and `CDraggerBeam::Tick` make: whether the segment is blocked.
fn dragger_line_blocked<R: Real>(world: &World<R>, ignore_walls: bool, from: Vec2<R>, to: Vec2<R>) -> bool {
    if ignore_walls {
        world.collision.intersect_no_laser_no_walls(from, to).hit != 0
    } else {
        world.collision.intersect_no_laser(from, to).hit != 0
    }
}

/// `GetPlayerChar(id)->m_Pos` (the entity position, see [`super::Character::pos`]) of an alive
/// character.
fn alive_pos<R: Real>(world: &World<R>, id: i32) -> Option<Vec2<R>> {
    if (0..MAX_CLIENTS as i32).contains(&id) {
        world.characters[id as usize].filter(|c| c.alive).map(|c| c.pos)
    } else {
        None
    }
}

// --- CDragger ---------------------------------------------------------------------------------

/// `CDragger::Tick()` (`dragger.cpp:36-55`) for `world.fixtures[index]`. `bbox`: the bounding box of
/// every alive character ([`super::alive_characters_bbox`], computed once per pass).
pub fn dragger_tick<R: Real>(world: &mut World<R>, index: usize, bbox: Option<(Vec2<R>, Vec2<R>)>) {
    if world.tick % MOVER_PERIOD != 0 {
        return;
    }
    let Fixture::Dragger(mut d) = world.fixtures[index] else {
        unreachable!("dragger_tick on a non-dragger fixture")
    };
    if let Some((_, speed)) = world
        .collision
        .mover_speed(d.pos.x.to_i32_trunc(), d.pos.y.to_i32_trunc())
    {
        d.core = speed;
    }
    d.pos += d.core;
    world.fixtures[index] = Fixture::Dragger(d);

    // Adopt the new position for all outgoing laser beams
    let beams = world.dragger_states[d.state as usize].beams;
    if beams != 0 {
        for slot in world.lasers.iter_mut() {
            if let LaserSlot::Beam(b) = slot
                && b.dragger as usize == index
                && beams & (1u128 << b.for_client) != 0
            {
                b.pos = d.pos;
            }
        }
    }

    // `LookForPlayersToDrag` is a provable no-op for a dragger that holds no target and no beam while
    // no alive character can be within `sv_dragger_range` of it: the in-range list is empty, so every
    // team's target stays `-1`, no `aIsTarget` bit is set, no beam is created or removed. (This is the
    // common case on a map with dozens of draggers — each one would otherwise clear and rescan four
    // `MAX_CLIENTS`-sized arrays every 7th tick.)
    let state = &world.dragger_states[d.state as usize];
    if state.beams == 0
        && state.target_in_team.iter().all(|&t| t == -1)
        && !could_be_near_alive_characters_bbox(
            bbox,
            d.pos,
            R::from_i32(world.config.sv_dragger_range) + R::from_f64(RANGE_PREFILTER_SLACK),
        )
    {
        return;
    }
    dragger_look_for_players_to_drag(world, index, d);
}

/// `CDragger::LookForPlayersToDrag()` (`dragger.cpp:57-150`).
fn dragger_look_for_players_to_drag<R: Real>(world: &mut World<R>, index: usize, d: Dragger<R>) {
    // Create a list of players who are in the range of the dragger
    let mut in_range = std::mem::take(&mut world.range_scratch);
    let reach = R::from_i32(world.config.sv_dragger_range) - core::physical_size::<R>();
    find_characters_in_range_into(world, d.pos, reach, &mut in_range);

    // The closest player (within range) in a team is selected as the target
    let mut closest_target_id_in_team = [-1i32; MAX_CLIENTS];
    let mut can_still_be_team_target = [false; MAX_CLIENTS];
    let mut is_target = [false; MAX_CLIENTS];
    let mut min_dist_in_team = [0i32; MAX_CLIENTS];

    for &target_id in &in_range {
        let target_team = character_team(&world.teams_core, target_id);

        // Do not create a dragger beam for super player
        if target_team == TEAM_SUPER {
            continue;
        }
        // If the dragger is disabled for the target's team, no dragger beam will be generated
        if switched_off_for(world, d.layer, d.number, target_team) {
            continue;
        }

        // Dragger beams can be created only for reachable, alive players
        let target_pos = alive_pos(world, target_id).expect("in-range characters are alive");
        let is_reachable = !dragger_line_blocked(world, d.ignore_walls, d.pos, target_pos);
        if is_reachable {
            // Solo players are dragged independently from the rest of the team
            if world.teams_core.get_solo(target_id) {
                is_target[target_id as usize] = true;
            } else {
                let distance = vmath::distance(target_pos, d.pos).to_i32_trunc();
                let t = target_team as usize;
                if min_dist_in_team[t] == 0 || min_dist_in_team[t] > distance {
                    min_dist_in_team[t] = distance;
                    closest_target_id_in_team[t] = target_id;
                }
                can_still_be_team_target[target_id as usize] = true;
            }
        }
    }
    in_range.clear();
    world.range_scratch = in_range;

    // Set the closest player for each team as a target if the team does not have a target player yet
    {
        let targets = &mut world.dragger_states[d.state as usize].target_in_team;
        for i in 0..MAX_CLIENTS {
            if (targets[i] != -1 && !can_still_be_team_target[targets[i] as usize]) || targets[i] == -1 {
                targets[i] = closest_target_id_in_team[i] as i8;
            }
            if targets[i] != -1 {
                is_target[targets[i] as usize] = true;
            }
        }
    }

    for (i, &targeted) in is_target.iter().enumerate() {
        let has_beam = world.dragger_states[d.state as usize].beams & (1u128 << i) != 0;
        if targeted && !has_beam {
            // Create Dragger Beams which have not been created yet
            world.dragger_states[d.state as usize].beams |= 1u128 << i;
            world.lasers.push(LaserSlot::Beam(DraggerBeam {
                pos: d.pos,
                strength: d.strength,
                ignore_walls: d.ignore_walls,
                for_client: i as i32,
                active: true,
                layer: d.layer,
                number: d.number,
                dragger: index as u16,
                marked_for_destroy: false,
            }));
            // The generated dragger beam is placed in the first position in the tick sequence and
            // would therefore no longer be executed automatically in this tick. To execute the
            // dragger beam nevertheless already this tick we call it manually.
            let beam_index = world.lasers.len() - 1;
            beam_tick(world, beam_index);
        } else if !targeted && has_beam {
            // Remove dragger beams that have not yet been deleted
            if let Some(beam_index) = find_beam(world, index, i as i32) {
                beam_reset(world, beam_index);
            }
        }
    }
}

/// The live [`DraggerBeam`] dragger `dragger` holds for `client` (`m_apDraggerBeam[client]`).
fn find_beam<R: Real>(world: &World<R>, dragger: usize, client: i32) -> Option<usize> {
    world.lasers.iter().position(|s| {
        matches!(s, LaserSlot::Beam(b) if b.dragger as usize == dragger && b.for_client == client && !b.marked_for_destroy)
    })
}

// --- CDraggerBeam -----------------------------------------------------------------------------

/// `CDraggerBeam::Reset()` (`dragger_beam.cpp:83-89`).
fn beam_reset<R: Real>(world: &mut World<R>, index: usize) {
    let LaserSlot::Beam(b) = &mut world.lasers[index] else {
        unreachable!("beam_reset on a non-beam slot")
    };
    b.marked_for_destroy = true;
    b.active = false;
    let (dragger, client) = (b.dragger as usize, b.for_client);
    // `m_pDragger->RemoveDraggerBeam(m_ForClientId)`
    if let Fixture::Dragger(d) = world.fixtures[dragger] {
        world.dragger_states[d.state as usize].beams &= !(1u128 << client);
    }
}

/// `CDraggerBeam::Tick()` (`dragger_beam.cpp:33-76`).
pub fn beam_tick<R: Real>(world: &mut World<R>, index: usize) {
    let LaserSlot::Beam(b) = world.lasers[index] else {
        unreachable!("beam_tick on a non-beam slot")
    };
    if !b.active {
        return;
    }

    // Drag only if the player is reachable and alive
    let Some(target_pos) = alive_pos(world, b.for_client) else {
        beam_reset(world, index);
        return;
    };
    let target_team = character_team(&world.teams_core, b.for_client);

    // The following checks are necessary, because the checks in CDragger::LookForPlayersToDrag
    // only take place after CDraggerBeam::Tick and only every 150ms. When the dragger is disabled
    // for the target player's team, the dragger beam dissolves.
    if world.tick % MOVER_PERIOD == 0 && switched_off_for(world, b.layer, b.number, target_team) {
        beam_reset(world, index);
        return;
    }

    // When the dragger can no longer reach the target player, the dragger beam dissolves
    let range = R::from_i32(world.config.sv_dragger_range);
    if vmath::distance(target_pos, b.pos) >= range || dragger_line_blocked(world, b.ignore_walls, b.pos, target_pos) {
        beam_reset(world, index);
    }
    // In the center of the dragger a tee does not experience speed-up
    else if vmath::distance(target_pos, b.pos) > R::from_i32(28) {
        let addition = vmath::normalize(b.pos - target_pos) * b.strength;
        let slot = world.cores.slot_of(b.for_client as u8).expect("alive");
        let move_restrictions = world.characters[b.for_client as usize]
            .expect("alive")
            .move_restrictions;
        add_velocity(world.cores.core_at_mut(slot), addition, move_restrictions);
    }
}

// --- CGun / CPlasma ---------------------------------------------------------------------------

/// `CGun::Tick()` (`gun.cpp:33-45`). `bbox`: see [`dragger_tick`].
pub fn gun_tick<R: Real>(world: &mut World<R>, index: usize, bbox: Option<(Vec2<R>, Vec2<R>)>) {
    if world.tick % MOVER_PERIOD == 0 {
        let Fixture::Gun(g) = &mut world.fixtures[index] else {
            unreachable!("gun_tick on a non-gun fixture")
        };
        if let Some((_, speed)) = world
            .collision
            .mover_speed(g.pos.x.to_i32_trunc(), g.pos.y.to_i32_trunc())
        {
            g.core = speed;
        }
        g.pos += g.core;
    }
    // `Fire()` with nobody within `sv_plasma_range` (+ the target's own radius) has no effect at all
    // (its loops only act on an in-range character), so skip it when no alive character is near.
    // (Read through a reference: on a map with a hundred turrets this runs once per turret per tick.)
    let Fixture::Gun(g) = &world.fixtures[index] else {
        unreachable!("gun_tick on a non-gun fixture")
    };
    if world.config.sv_plasma_per_sec > 0
        && could_be_near_alive_characters_bbox(
            bbox,
            g.pos,
            R::from_i32(world.config.sv_plasma_range) + core::physical_size::<R>() + R::from_f64(RANGE_PREFILTER_SLACK),
        )
    {
        let g = *g;
        gun_fire(world, g);
    }
}

/// `CGun::Fire()` (`gun.cpp:47-132`).
fn gun_fire<R: Real>(world: &mut World<R>, g: Gun<R>) {
    // Create a list of players who are in the range of the turret
    let mut in_range = std::mem::take(&mut world.range_scratch);
    find_characters_in_range_into(world, g.pos, R::from_i32(world.config.sv_plasma_range), &mut in_range);

    // The closest player (within range) in a team is selected as the target
    let mut target_id_in_team = [-1i32; MAX_CLIENTS];
    let mut is_target = [false; MAX_CLIENTS];
    let mut min_dist_in_team = [0i32; MAX_CLIENTS];
    let tick = world.tick;
    let per_sec = world.config.sv_plasma_per_sec;

    for &target_id in &in_range {
        let target_team = character_team(&world.teams_core, target_id);
        // Do not fire at super players
        if target_team == TEAM_SUPER {
            continue;
        }
        // If the turret is disabled for the target's team, the turret will not fire
        if switched_off_for(world, g.layer, g.number, target_team) {
            continue;
        }

        // Turrets can only shoot at a speed of sv_plasma_per_sec
        let target_is_solo = world.teams_core.get_solo(target_id);
        let state = &world.gun_states[g.state as usize];
        if (target_is_solo && state.last_fire_solo[target_id as usize] + core::SERVER_TICK_SPEED / per_sec > tick)
            || (!target_is_solo
                && state.last_fire_team[target_team as usize] + core::SERVER_TICK_SPEED / per_sec > tick)
        {
            continue;
        }

        // Turrets can shoot only at reachable, alive players
        let target_pos = alive_pos(world, target_id).expect("in-range characters are alive");
        let is_reachable = world.collision.intersect_line(g.pos, target_pos).hit == 0;
        if is_reachable {
            // Turrets fire on solo players regardless of the rest of the team
            if target_is_solo {
                is_target[target_id as usize] = true;
                world.gun_states[g.state as usize].last_fire_solo[target_id as usize] = tick;
            } else {
                let distance = vmath::distance(target_pos, g.pos).to_i32_trunc();
                let t = target_team as usize;
                if min_dist_in_team[t] == 0 || min_dist_in_team[t] > distance {
                    min_dist_in_team[t] = distance;
                    target_id_in_team[t] = target_id;
                }
            }
        }
    }
    in_range.clear();
    world.range_scratch = in_range;

    // Set the closest player for each team as a target
    for i in 0..MAX_CLIENTS {
        if target_id_in_team[i] != -1 {
            is_target[target_id_in_team[i] as usize] = true;
            world.gun_states[g.state as usize].last_fire_team[i] = tick;
        }
    }

    for (i, &targeted) in is_target.iter().enumerate() {
        // Fire at each target
        if targeted {
            let target_pos = alive_pos(world, i as i32).expect("targets are alive");
            // `new CPlasma(...)` (`plasma.cpp:16-31`)
            world.lasers.push(LaserSlot::Plasma(Plasma {
                pos: g.pos,
                core: vmath::normalize(target_pos - g.pos),
                freeze: g.freeze,
                explosive: g.explosive,
                for_client: i as i32,
                eval_tick: tick,
                life_time: (R::from_i32(core::SERVER_TICK_SPEED) * R::from_f64(1.5)).to_i32_trunc(),
                marked_for_destroy: false,
            }));
        }
    }
}

/// `CPlasma::Tick()` (`plasma.cpp:33-53`).
pub fn plasma_tick<R: Real>(world: &mut World<R>, index: usize) {
    let LaserSlot::Plasma(p) = world.lasers[index] else {
        unreachable!("plasma_tick on a non-plasma slot")
    };
    // A plasma bullet has only a limited lifetime
    if p.life_time == 0 {
        plasma_reset(world, index);
        return;
    }
    // Without a target, a plasma bullet has no reason to live
    if alive_pos(world, p.for_client).is_none() {
        plasma_reset(world, index);
        return;
    }
    let LaserSlot::Plasma(p) = &mut world.lasers[index] else {
        unreachable!()
    };
    p.life_time -= 1;
    // `Move()`
    p.pos += p.core;
    p.core *= R::from_f64(f64::from(PLASMA_ACCEL));
    let p = *p;
    plasma_hit_character(world, index, p);
    // Plasma bullets may explode twice if they would hit both a player and an obstacle in the next move step
    plasma_hit_obstacle(world, index, p);
}

fn plasma_reset<R: Real>(world: &mut World<R>, index: usize) {
    if let LaserSlot::Plasma(p) = &mut world.lasers[index] {
        p.marked_for_destroy = true;
    }
}

/// `CPlasma::HitCharacter` (`plasma.cpp:61-87`).
fn plasma_hit_character<R: Real>(world: &mut World<R>, index: usize, p: Plasma<R>) -> bool {
    let Some((hit_id, _)) = intersect_character_ex(world, p.pos, p.pos + p.core, R::ZERO, -1, p.for_client, -1) else {
        return false;
    };

    // Super player should not be able to stop the plasma bullets
    if character_team(&world.teams_core, hit_id) == TEAM_SUPER {
        return false;
    }

    {
        let slot = world.cores.slot_of(hit_id as u8).expect("a hit character has a core");
        let mut hit_char = world.characters[hit_id as usize].expect("a hit character exists");
        let mut hit_core = *world.cores.core_at(slot);
        if p.freeze {
            freeze_default(&mut hit_char, &mut hit_core, world.tick, world.config.sv_freeze_delay);
        } else {
            unfreeze(&mut hit_char, &mut hit_core);
        }
        world.characters[hit_id as usize] = Some(hit_char);
        *world.cores.core_at_mut(slot) = hit_core;
    }
    if p.explosive {
        // Plasma Turrets are very precise weapons only one tee gets speed from it,
        // other tees near the explosion remain unaffected
        let target_team = character_team(&world.teams_core, p.for_client);
        create_explosion(world, p.pos, p.for_client, WEAPON_GRENADE, true, target_team);
    }
    plasma_reset(world, index);
    true
}

/// `CPlasma::HitObstacle` (`plasma.cpp:89-105`).
fn plasma_hit_obstacle<R: Real>(world: &mut World<R>, index: usize, p: Plasma<R>) -> bool {
    // Check if the plasma bullet is stopped by a solid block or a laser stopper
    if world.collision.intersect_no_laser(p.pos, p.pos + p.core).hit != 0 {
        if p.explosive {
            // Even in the case of an explosion due to a collision with obstacles, only one player is affected
            let target_team = character_team(&world.teams_core, p.for_client);
            create_explosion(world, p.pos, p.for_client, WEAPON_GRENADE, true, target_team);
        }
        plasma_reset(world, index);
        return true;
    }
    false
}

// --- CLight -----------------------------------------------------------------------------------

/// `CLight::Move()` (`light.cpp:46-71`).
fn light_move<R: Real>(l: &mut Light<R>) {
    if l.speed != 0 {
        if (l.curve_length >= l.length && l.speed > 0) || (l.curve_length <= 0 && l.speed < 0) {
            l.speed = -l.speed;
        }
        l.curve_length += l.speed * MOVER_PERIOD + l.length_l;
        l.length_l = 0;
        if l.curve_length > l.length {
            l.length_l = l.curve_length - l.length;
            l.curve_length = l.length;
        } else if l.curve_length < 0 {
            l.length_l = l.curve_length;
            l.curve_length = 0;
        }
    }

    l.rotation += l.angular_speed * R::from_i32(MOVER_PERIOD);
    let two_pi = R::PI * R::from_i32(2);
    if l.rotation > two_pi {
        l.rotation -= two_pi;
    } else if l.rotation < R::ZERO {
        l.rotation += two_pi;
    }
}

/// `CLight::Step()` (`light.cpp:73-79`).
fn light_step<R: Real>(world: &World<R>, l: &mut Light<R>) {
    light_move(l);
    let direction = Vec2::new(l.rotation.sin(), l.rotation.cos());
    let next_position = l.pos + vmath::normalize(direction) * R::from_i32(l.curve_length);
    l.to = world.collision.intersect_no_laser(l.pos, next_position).collision;
}

/// `CLight::Tick()` (`light.cpp:86-97`).
pub fn light_tick<R: Real>(world: &mut World<R>, index: usize) {
    if world.tick % MOVER_PERIOD == 0 {
        let Fixture::Light(mut l) = world.fixtures[index] else {
            unreachable!("light_tick on a non-light fixture")
        };
        if let Some((_, speed)) = world
            .collision
            .mover_speed(l.pos.x.to_i32_trunc(), l.pos.y.to_i32_trunc())
        {
            l.core = speed;
        }
        l.pos += l.core;
        light_step(world, &mut l);
        world.fixtures[index] = Fixture::Light(l);
    }

    // `HitCharacter()` (`light.cpp:32-44`): `IntersectedCharacters(m_Pos, m_To, 0.0f, nullptr)`
    // (`gameworld.cpp:360-380`) visits the characters in entity-list order; freezing one never
    // changes which others are hit, so the C++ source's collect-then-freeze two-pass shape folds
    // into one.
    let Fixture::Light(l) = &world.fixtures[index] else {
        unreachable!("light_tick on a non-light fixture")
    };
    let (pos, to, layer, number) = (l.pos, l.to, l.layer, l.number);
    let radius = core::physical_size::<R>();
    for i in 0..world.entity_order.len() {
        let id = world.entity_order[i] as i32;
        if !world.characters[id as usize].is_some_and(|c| c.alive) {
            continue;
        }
        let char_pos = world.characters[id as usize].expect("alive").pos;
        let Some(intersect_pos) = vmath::closest_point_on_line(pos, to, char_pos) else {
            continue;
        };
        if vmath::distance(char_pos, intersect_pos) < radius {
            let team = character_team(&world.teams_core, id);
            if switched_off_for(world, layer, number, team) {
                continue;
            }
            let slot = world.cores.slot_of(id as u8).expect("alive");
            let mut c = world.characters[id as usize].expect("alive");
            let mut core = *world.cores.core_at(slot);
            freeze_default(&mut c, &mut core, world.tick, world.config.sv_freeze_delay);
            world.characters[id as usize] = Some(c);
            *world.cores.core_at_mut(slot) = core;
        }
    }
}

impl<R: Real> World<R> {
    /// For a *client-side* reconstruction of the server's world (`ddai-world::LiveWorld`), which has
    /// no way to rebuild the phase of the map's moving fixtures from a snapshot (task 1.6b, review F4):
    ///
    /// - drops every turret and every rotating or opening/closing light (`m_AngularSpeed != 0 ||
    ///   m_Speed != 0`) — the DDNet client does not predict them either, and a wrong-phase light
    ///   mispredicts *more* than no light at all (measured, review 1.6b F4);
    /// - keeps draggers, and static lights, and gives the latter their final beam (`m_To`) at once: the
    ///   server only computes it on the first tick divisible by 7, which on a world that starts mid-game
    ///   would leave the beam empty for up to 6 predicted ticks.
    ///
    /// Must run before anything is stepped (it removes entries from [`World::fixtures`], whose indices
    /// dragger beams refer to). Turret state slots stay allocated, unused.
    pub fn retain_predictable_fixtures(&mut self) {
        self.fixtures.retain(|f| match f {
            Fixture::Dragger(_) => true,
            Fixture::Gun(_) => false,
            Fixture::Light(l) => l.angular_speed == R::ZERO && l.speed == 0,
        });
        for i in 0..self.fixtures.len() {
            if let Fixture::Light(mut l) = self.fixtures[i] {
                light_step(self, &mut l);
                self.fixtures[i] = Fixture::Light(l);
            }
        }
    }
}
