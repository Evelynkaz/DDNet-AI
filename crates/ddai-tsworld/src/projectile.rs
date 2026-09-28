//! Literal port of `src/core/projectile.ts` (TS, `Wranked1/DDNet-AI`, GPL-3.0).
//!
//! **Restructuring vs. TS (see `character_core.rs`'s module doc comment for the same rationale):**
//! TS's `EntityWorld` is an interface `Projectile`/`Laser` hold a reference to; here it is a
//! trait [`EntityWorld`] implemented by [`crate::world::SimWorld`], and `tick`/`doBounce` take
//! `&mut dyn EntityWorld` (in practice always the one `SimWorld`) instead of a stored field.
//! Every number and the order operations happen in is otherwise unchanged from TS.

use crate::collision::Collision;
use crate::tuning::{SERVER_TICK_SPEED, tuning};
use crate::types::{WEAPON_GRENADE, WEAPON_GUN, WEAPON_LASER, WEAPON_SHOTGUN, WorldEvent};
use crate::vmath::{Vec2, clamp, round_to_int, vadd, vdistance, vlength, vmul, vnormalize, vsub};
use ddai_jsmath as js;

/// `EntityWorld` (`projectile.ts:8-21`), implemented by [`crate::world::SimWorld`].
pub trait EntityWorld {
    fn tick(&self) -> i64;
    fn collision(&self) -> &Collision;
    fn sv_hit(&self) -> bool;
    fn is_alive(&self, id: i32) -> bool;
    fn tee_pos(&self, id: i32) -> Option<Vec2>;
    fn intersect_character(
        &self,
        pos0: Vec2,
        pos1: Vec2,
        radius: f64,
        exclude_id: i32,
        only_id: i32,
    ) -> Option<(i32, Vec2)>;
    fn find_characters_in_radius(&self, pos: Vec2, radius: f64) -> Vec<i32>;
    fn apply_force(&mut self, id: i32, force: Vec2);
    fn unfreeze(&mut self, id: i32);
}

/// `calcPos(pos, dir, curvature, speed, time)` (`projectile.ts:23-26`).
fn calc_pos(pos: Vec2, dir: Vec2, curvature: f64, speed: f64, time: f64) -> Vec2 {
    let t = time * speed;
    Vec2 {
        x: pos.x + dir.x * t,
        y: pos.y + dir.y * t + (curvature / 10000.0) * (t * t),
    }
}

/// `weaponCurvatureSpeed(type)` (`projectile.ts:28-37`).
fn weapon_curvature_speed(kind: i32) -> (f64, f64) {
    let tune = tuning();
    if kind == WEAPON_GRENADE {
        (tune.grenade_curvature, tune.grenade_speed)
    } else if kind == WEAPON_GUN {
        (tune.gun_curvature, tune.gun_speed)
    } else {
        (0.0, 0.0)
    }
}

/// `isGameLayerClipped(pos, collision)` (`projectile.ts:39-43`).
pub fn is_game_layer_clipped(pos: Vec2, collision: &Collision) -> bool {
    let tx = js::trunc(round_to_int(pos.x) / 32.0);
    let ty = js::trunc(round_to_int(pos.y) / 32.0);
    tx < -200.0 || tx > (collision.width as f64) + 200.0 || ty < -200.0 || ty > (collision.height as f64) + 200.0
}

/// `createExplosion(world, pos, owner, weapon, noDamage, events)` (`projectile.ts:45-65`).
pub fn create_explosion(
    world: &mut dyn EntityWorld,
    pos: Vec2,
    owner: i32,
    weapon: i32,
    no_damage: bool,
    events: &mut Vec<WorldEvent>,
) {
    let _ = weapon;
    events.push(WorldEvent::Explosion { pos, owner });

    let radius = 135.0;
    let inner_radius = 48.0;
    let tune = tuning();
    for id in world.find_characters_in_radius(pos, radius) {
        let Some(p) = world.tee_pos(id) else { continue };
        let diff = vsub(p, pos);
        let len = vlength(diff);
        let force_dir = if len != 0.0 {
            vnormalize(diff)
        } else {
            Vec2 { x: 0.0, y: 1.0 }
        };
        let falloff = 1.0 - clamp((len - inner_radius) / (radius - inner_radius), 0.0, 1.0);
        let strength = tune.explosion_strength;
        let dmg = strength * falloff;
        if js::trunc(dmg) == 0.0 {
            continue;
        }
        if world.sv_hit() || no_damage || owner == id {
            world.apply_force(id, vmul(force_dir, dmg * 2.0));
        }
    }
}

/// `ProjectileState2` (`projectile.ts:67-70`) — the `saveState`/wire format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectileState2 {
    pub id: i32,
    pub kind: i32,
    pub owner: i32,
    pub pos: Vec2,
    pub dir: Vec2,
    pub start_tick: i64,
    pub life_span: i64,
    pub explosive: bool,
    pub marked_for_destroy: bool,
}

/// `LaserState2` (`projectile.ts:72-75`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaserState2 {
    pub id: i32,
    pub owner: i32,
    pub kind: i32,
    pub pos: Vec2,
    pub dir: Vec2,
    pub energy: f64,
    pub bounces: i32,
    pub eval_tick: i64,
    pub zero_energy_bounce_in_last_tick: bool,
    pub marked_for_destroy: bool,
}

/// `class Projectile` (`projectile.ts:77-157`).
#[derive(Debug, Clone, Copy)]
pub struct Projectile {
    pub id: i32,
    pub kind: i32,
    pub owner: i32,
    pub pos: Vec2,
    pub dir: Vec2,
    pub start_tick: i64,
    pub explosive: bool,
    pub vel: Vec2,
    pub life_span: i64,
    pub marked_for_destroy: bool,
}

impl Projectile {
    /// `constructor(id, type, owner, pos, dir, startTick, lifeSpan, explosive)` (`projectile.ts:89-100`).
    #[allow(clippy::too_many_arguments)] // Literal 1:1 port of the TS constructor's own 8 parameters.
    pub fn new(
        id: i32,
        kind: i32,
        owner: i32,
        pos: Vec2,
        dir: Vec2,
        start_tick: i64,
        life_span: i64,
        explosive: bool,
    ) -> Self {
        let (_curvature, speed) = weapon_curvature_speed(kind);
        Projectile {
            id,
            kind,
            owner,
            pos,
            dir,
            start_tick,
            explosive,
            vel: vmul(dir, speed),
            life_span,
            marked_for_destroy: false,
        }
    }

    /// A placeholder used only by [`crate::world::SimWorld::tick_entities`] to satisfy
    /// `mem::replace` while a real projectile is being ticked in place (never observable —
    /// always immediately overwritten). Not part of TS.
    pub(crate) fn dummy() -> Self {
        Projectile::new(0, 0, -1, Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 0.0 }, 0, 0, false)
    }

    /// `saveState()` (`projectile.ts:102-104`).
    pub fn save_state(&self) -> ProjectileState2 {
        ProjectileState2 {
            id: self.id,
            kind: self.kind,
            owner: self.owner,
            pos: self.pos,
            dir: self.dir,
            start_tick: self.start_tick,
            life_span: self.life_span,
            explosive: self.explosive,
            marked_for_destroy: self.marked_for_destroy,
        }
    }

    /// `Projectile.fromState(world, st)` (`projectile.ts:106-110`).
    pub fn from_state(st: &ProjectileState2) -> Self {
        let mut p = Projectile::new(
            st.id,
            st.kind,
            st.owner,
            st.pos,
            st.dir,
            st.start_tick,
            st.life_span,
            st.explosive,
        );
        p.marked_for_destroy = st.marked_for_destroy;
        p
    }

    /// `posAtTick(tick)` (`projectile.ts:112-116`).
    pub fn pos_at_tick(&self, tick: i64) -> Vec2 {
        let (curvature, speed) = weapon_curvature_speed(self.kind);
        let time = (tick - self.start_tick) as f64 / SERVER_TICK_SPEED;
        calc_pos(self.pos, self.dir, curvature, speed, time)
    }

    /// `tick(world, events)` (`projectile.ts:118-156`).
    pub fn tick(&mut self, world: &mut dyn EntityWorld, events: &mut Vec<WorldEvent>) {
        let prev_pos = self.pos_at_tick(world.tick() - 1);
        let cur_pos = self.pos_at_tick(world.tick());

        let hit = world.collision().intersect_line(prev_pos, cur_pos);
        let collide = hit.collision != 0;

        let mut target_id = -1;
        if world.sv_hit()
            && let Some((id, _pos)) = world.intersect_character(prev_pos, hit.out_pos, 6.0, self.owner, -1)
        {
            target_id = id;
        }

        if self.life_span > -1 {
            self.life_span -= 1;
        }

        if self.owner >= 0 && !world.is_alive(self.owner) {
            self.marked_for_destroy = true;
            return;
        }

        let out_of_bounds = is_game_layer_clipped(cur_pos, world.collision());
        if target_id != -1 || collide || out_of_bounds {
            if self.explosive {
                create_explosion(world, hit.out_pos, self.owner, self.kind, self.owner == -1, events);
                self.marked_for_destroy = true;
                return;
            }
            self.marked_for_destroy = true;
            return;
        }

        if self.life_span == -1 {
            if self.explosive {
                create_explosion(world, hit.out_pos, self.owner, self.kind, self.owner == -1, events);
            }
            self.marked_for_destroy = true;
        }
    }
}

/// `class Laser` (`projectile.ts:159-269`).
#[derive(Debug, Clone, Copy)]
pub struct Laser {
    pub id: i32,
    pub owner: i32,
    pub kind: i32,
    pub pos: Vec2,
    pub dir: Vec2,
    pub energy: f64,
    pub bounces: i32,
    pub eval_tick: i64,
    pub zero_energy_bounce_in_last_tick: bool,
    pub marked_for_destroy: bool,
}

impl Laser {
    /// `constructor(id, owner, type, pos, dir, startEnergy)` (`projectile.ts:187-194`).
    pub fn new(id: i32, owner: i32, kind: i32, pos: Vec2, dir: Vec2, start_energy: f64) -> Self {
        Laser {
            id,
            owner,
            kind,
            pos,
            dir,
            energy: start_energy,
            bounces: 0,
            eval_tick: 0,
            zero_energy_bounce_in_last_tick: false,
            marked_for_destroy: false,
        }
    }

    /// See [`Projectile::dummy`].
    pub(crate) fn dummy() -> Self {
        Laser::new(0, -1, 0, Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 0.0 }, 0.0)
    }

    /// `saveState()` (`projectile.ts:171-173`).
    pub fn save_state(&self) -> LaserState2 {
        LaserState2 {
            id: self.id,
            owner: self.owner,
            kind: self.kind,
            pos: self.pos,
            dir: self.dir,
            energy: self.energy,
            bounces: self.bounces,
            eval_tick: self.eval_tick,
            zero_energy_bounce_in_last_tick: self.zero_energy_bounce_in_last_tick,
            marked_for_destroy: self.marked_for_destroy,
        }
    }

    /// `Laser.fromState(world, st)` (`projectile.ts:175-185`).
    pub fn from_state(st: &LaserState2) -> Self {
        let mut l = Laser::new(st.id, st.owner, st.kind, st.pos, st.dir, st.energy);
        l.bounces = st.bounces;
        l.eval_tick = st.eval_tick;
        l.zero_energy_bounce_in_last_tick = st.zero_energy_bounce_in_last_tick;
        l.marked_for_destroy = st.marked_for_destroy;
        l
    }

    /// `hitCharacter(world, from, to, events)` (`projectile.ts:196-219`).
    fn hit_character(
        &mut self,
        world: &mut dyn EntityWorld,
        from: Vec2,
        to: Vec2,
        events: &mut Vec<WorldEvent>,
    ) -> bool {
        let dont_hit_self = self.bounces == 0;
        let exclude_id = if dont_hit_self { self.owner } else { -1 };
        let only_id = if world.sv_hit() { -1 } else { self.owner };
        let Some((hit_id, hit_pos)) = world.intersect_character(from, to, 0.0, exclude_id, only_id) else {
            return false;
        };

        self.pos = hit_pos;
        self.energy = -1.0;

        if self.kind == WEAPON_SHOTGUN {
            let tune = tuning();
            let strength = tune.shotgun_strength;
            if let Some(hit_tee_pos) = world.tee_pos(hit_id)
                && (hit_tee_pos.x != from.x || hit_tee_pos.y != from.y)
            {
                world.apply_force(hit_id, vmul(vnormalize(vsub(from, hit_tee_pos)), strength));
            }
        } else if self.kind == WEAPON_LASER {
            world.unfreeze(hit_id);
        }

        events.push(WorldEvent::LaserHit {
            from: self.owner,
            to: hit_id,
            weapon: self.kind,
        });
        true
    }

    /// `doBounce(world, events)` (`projectile.ts:221-261`).
    pub fn do_bounce(&mut self, world: &mut dyn EntityWorld, events: &mut Vec<WorldEvent>) {
        self.eval_tick = world.tick();
        if self.energy < 0.0 {
            self.marked_for_destroy = true;
            return;
        }

        let ray_start = self.pos;
        let to = vadd(ray_start, vmul(self.dir, self.energy));
        let hit = world.collision().intersect_line(ray_start, to);

        if hit.collision != 0 {
            let hit_to = hit.out_before_pos;
            if !self.hit_character(world, ray_start, hit_to, events) {
                self.pos = hit_to;

                let mut temp_pos = self.pos;
                let mut temp_vel = vmul(self.dir, 4.0);
                world.collision().move_point(&mut temp_pos, &mut temp_vel, 1.0, None);
                self.pos = temp_pos;
                self.dir = vnormalize(temp_vel);

                let tune = tuning();
                let dist = vdistance(ray_start, self.pos);
                if dist == 0.0 && self.zero_energy_bounce_in_last_tick {
                    self.energy = -1.0;
                } else {
                    self.energy -= dist + tune.laser_bounce_cost;
                }
                self.zero_energy_bounce_in_last_tick = dist == 0.0;

                self.bounces += 1;
                if self.bounces as f64 > tune.laser_bounce_num {
                    self.energy = -1.0;
                }
            }
        } else if !self.hit_character(world, ray_start, to, events) {
            self.pos = to;
            self.energy = -1.0;
        }
    }

    /// `tick(world, events)` (`projectile.ts:263-268`).
    pub fn tick(&mut self, world: &mut dyn EntityWorld, events: &mut Vec<WorldEvent>) {
        let tune = tuning();
        let delay_ticks = SERVER_TICK_SPEED * tune.laser_bounce_delay / 1000.0;
        if (world.tick() - self.eval_tick) as f64 > delay_ticks {
            self.do_bounce(world, events);
        }
    }
}
