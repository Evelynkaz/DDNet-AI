//! Snapshot projectile items -> `ddai_physics::world::Projectile<f32>` (task 2.4b).
//!
//! The DDNet client never spawns map entities for prediction: every projectile in its predicted
//! world (gun/grenade shots and the map's crazy-shotgun cannons alike) is built from a snapshot
//! item (`CGameWorld::NetObjAdd`, `prediction/gameworld.cpp:437-500`), with the item's own
//! `m_StartTick` as the projectile's time origin. This module is that construction, ported:
//!
//! - [`extract`] = `ExtractProjectileInfo` (`client/projectile_data.cpp:15-102`): which of the three
//!   wire shapes carries what (`CNetObj_Projectile`, `CNetObj_DDRaceProjectile`,
//!   `CNetObj_DDNetProjectile`; a legacy `CNetObj_Projectile` whose `m_VelY` has
//!   `LEGACYPROJECTILEFLAG_IS_DDNET` set is really a `CNetObj_DDRaceProjectile` in disguise).
//! - [`projectile_from_view`] = `CProjectile::CProjectile(CGameWorld*, int Id, const
//!   CProjectileData*)` (`prediction/entities/projectile.cpp:171-213`) plus `NetObjAdd`'s
//!   "skip grenades on ball mod" filter (`gameworld.cpp:447-448`).
//!
//! **Approximations** (all shared with the real client, none added by this port):
//! - positions arrive quantised to 0.01 px (`round_to_int(m_Pos * 100)`), map-cannon directions to
//!   1e-6 (`round_to_int(m_Direction * 1e6)`), a player's shot direction as the rounded integer
//!   aim vector (`m_InitDir`, re-normalised here), so a projectile is reproduced to ~0.005 px, not
//!   bit-for-bit — the server's own `float` state is not on the wire;
//! - a legacy snapshot item without extra info (`CNetObj_Projectile`, no `IS_DDNET`) has no
//!   owner/bounce/freeze information at all; the client then guesses `m_Owner = -1` and
//!   `m_Explosive` from the direction length, and so do we (the client's "find the shooter nearby"
//!   heuristic for those is not ported — DDNet 20.x servers send the extended shapes to any client
//!   that announced a DDNet version, which `ddai-client` does).

use ddai_net::generated::enums::{legacyprojectileflagflag as legacy_flag, projectileflagflag as flag};
use ddai_net::generated::objects;
use ddai_net::view::ProjectileView;
use ddai_physics::collision::Collision;
use ddai_physics::core::{MAX_CLIENTS, SERVER_TICK_SPEED, WEAPON_GRENADE, WEAPON_GUN, WEAPON_SHOTGUN};
use ddai_physics::vmath::{Vec2, length, normalize};
use ddai_physics::world::{Layer, Projectile, TUNE_ZONE_COUNT, TuningList};

/// `CProjectileData` (`client/projectile_data.h`): what the client reads out of a projectile item.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectileData {
    pub start_pos: Vec2<f32>,
    pub start_vel: Vec2<f32>,
    pub weapon_type: i32,
    pub start_tick: i32,
    /// `m_ExtraInfo`: the item carried owner/bounce/explosive/freeze (DDRace/DDNet shapes).
    pub extra_info: bool,
    /// `m_Owner`, already sanitised to `-1` or `0..MAX_CLIENTS`.
    pub owner: i32,
    pub bouncing: i32,
    pub explosive: bool,
    pub freeze: bool,
    pub tune_zone: i32,
    pub switch_number: i32,
}

fn clamp_zone(zone: i32) -> i32 {
    if (0..TUNE_ZONE_COUNT as i32).contains(&zone) {
        zone
    } else {
        0
    }
}

fn map_tune_zone(collision: &Collision<f32>, pos: Vec2<f32>) -> i32 {
    collision.is_tune(collision.get_map_index(pos))
}

/// `ExtractProjectileInfoDDRace` (`projectile_data.cpp:61-89`).
fn extract_ddrace(p: &objects::DDRaceProjectile, collision: &Collision<f32>) -> ProjectileData {
    let start_pos = Vec2::new(p.x as f32 / 100.0, p.y as f32 / 100.0);
    let angle = p.angle as f32 / 1_000_000.0;
    let mut owner = p.data & 255;
    if p.data & legacy_flag::NO_OWNER != 0 || owner >= MAX_CLIENTS as i32 {
        owner = -1;
    }
    ProjectileData {
        start_pos,
        // glibc's `sinf`/`cosf` as DDNet's `vec2(sinf(-Angle), cosf(-Angle))` (D-127: `ddai-libm`, the same bits on every platform).
        start_vel: Vec2::new(ddai_libm::sinf(-angle), ddai_libm::cosf(-angle)),
        weapon_type: p.type_,
        start_tick: p.start_tick,
        extra_info: true,
        owner,
        bouncing: (p.data >> 10) & 3,
        explosive: p.data & legacy_flag::EXPLOSIVE != 0,
        freeze: p.data & legacy_flag::FREEZE != 0,
        tune_zone: map_tune_zone(collision, start_pos),
        switch_number: 0,
    }
}

/// `ExtractProjectileInfoDDNet` (`projectile_data.cpp:91-121`).
fn extract_ddnet(p: &objects::DDNetProjectile) -> ProjectileData {
    // Ownerless (map) projectiles carry `m_Direction * 1e6`; owned ones the raw integer aim vector.
    let mut vel = if p.owner < 0 {
        Vec2::new(p.vel_x as f32 / 1e6, p.vel_y as f32 / 1e6)
    } else {
        Vec2::new(p.vel_x as f32, p.vel_y as f32)
    };
    if p.flags & flag::NORMALIZE_VEL != 0 {
        vel = normalize(vel);
    }
    let mut bouncing = 0;
    if p.flags & flag::BOUNCE_HORIZONTAL != 0 {
        bouncing |= 1;
    }
    if p.flags & flag::BOUNCE_VERTICAL != 0 {
        bouncing |= 2;
    }
    ProjectileData {
        start_pos: Vec2::new(p.x as f32 / 100.0, p.y as f32 / 100.0),
        start_vel: vel,
        weapon_type: p.type_,
        start_tick: p.start_tick,
        extra_info: true,
        owner: if (0..MAX_CLIENTS as i32).contains(&p.owner) {
            p.owner
        } else {
            -1
        },
        bouncing,
        explosive: p.flags & flag::EXPLOSIVE != 0,
        freeze: p.flags & flag::FREEZE != 0,
        tune_zone: clamp_zone(p.tune_zone),
        switch_number: p.switch_number,
    }
}

/// `ExtractProjectileInfo` (`projectile_data.cpp:15-59`), one arm per wire shape.
pub fn extract(view: &ProjectileView, collision: &Collision<f32>) -> ProjectileData {
    match view {
        ProjectileView::DDNet(p) => extract_ddnet(p),
        ProjectileView::DDRace(p) => extract_ddrace(p, collision),
        ProjectileView::Legacy(p) => {
            // `UseProjectileExtraInfo`: the server smuggles a `CNetObj_DDRaceProjectile` through
            // the legacy object for clients that know DDNet's antiping but not its own netobjs
            // (`server/entities/projectile.cpp:397-401`) — `m_VelX` is then `m_Angle`, `m_VelY` `m_Data`.
            if p.vel_y >= 0 && p.vel_y & legacy_flag::IS_DDNET != 0 {
                return extract_ddrace(
                    &objects::DDRaceProjectile {
                        x: p.x,
                        y: p.y,
                        angle: p.vel_x,
                        data: p.vel_y,
                        type_: p.type_,
                        start_tick: p.start_tick,
                    },
                    collision,
                );
            }
            let start_pos = Vec2::new(p.x as f32, p.y as f32);
            ProjectileData {
                start_pos,
                start_vel: Vec2::new(p.vel_x as f32 / 100.0, p.vel_y as f32 / 100.0),
                weapon_type: p.type_,
                start_tick: p.start_tick,
                extra_info: false,
                owner: -1,
                bouncing: 0,
                explosive: false,
                freeze: false,
                tune_zone: map_tune_zone(collision, start_pos),
                switch_number: 0,
            }
        }
    }
}

/// Builds the physics-world projectile the client would predict for `view`, as of `world_tick`
/// (the snapshot's tick; `CGameWorld::GameTick()` in the client constructor). `None` where the
/// client skips the item or the world has no model for it: a non-gun/shotgun/grenade type, a
/// non-shotgun projectile whose direction is not a unit vector (`gameworld.cpp:447-448`'s ball-mod
/// workaround), or non-finite numbers (a hostile or corrupt item must not poison the simulation).
///
/// Life span: the client sets `m_LifeSpan = Lifetime - (GameTick - m_StartTick)` (`projectile.cpp:
/// 194-209`; 20 s for a DDRace shotgun, else the weapon's tuned lifetime in the projectile's tune
/// zone) and, for a DDRace shotgun, re-derives it from the fresh `m_StartTick` on every snapshot
/// (`gameworld.cpp:453-454`). The one thing that formula gets wrong for us is a *map* cannon that
/// has not bounced for over 20 s, which the server never expires (`m_LifeSpan == -2`,
/// `gamecontroller.cpp:236,261`) — recognised as an ownerless, bouncing shotgun and given `-2`.
pub fn projectile_from_view(
    view: &ProjectileView,
    world_tick: i32,
    collision: &Collision<f32>,
    tuning: &TuningList,
) -> Option<Projectile<f32>> {
    let d = extract(view, collision);
    if !matches!(d.weapon_type, WEAPON_GUN | WEAPON_SHOTGUN | WEAPON_GRENADE) {
        return None;
    }
    let finite = [d.start_pos.x, d.start_pos.y, d.start_vel.x, d.start_vel.y]
        .iter()
        .all(|v| v.is_finite());
    if !finite {
        return None;
    }
    let dir_len = length(d.start_vel);
    if d.weapon_type != WEAPON_SHOTGUN && (dir_len - 1.0).abs() > 0.02 {
        return None;
    }

    let (owner, bouncing, freeze, explosive) = if d.extra_info {
        (d.owner, d.bouncing, d.freeze, d.explosive)
    } else {
        // `CProjectile::CProjectile`'s no-extra-info branch (`projectile.cpp:183-189`).
        (
            -1,
            0,
            false,
            d.weapon_type == WEAPON_GRENADE && (1.0 - dir_len).abs() < 0.015,
        )
    };

    let tune_zone = clamp_zone(d.tune_zone);
    let zone = tuning.zone(tune_zone);
    let tick_speed = SERVER_TICK_SPEED as f32;
    let lifetime = match d.weapon_type {
        WEAPON_GRENADE => (zone.grenade_lifetime() * tick_speed) as i32,
        WEAPON_GUN => (zone.gun_lifetime() * tick_speed) as i32,
        // DDRace's shotgun projectile lives 20 s regardless of tuning (`!m_IsDDRace` gate).
        _ => 20 * SERVER_TICK_SPEED,
    };
    let life_span = if d.weapon_type == WEAPON_SHOTGUN && owner < 0 && bouncing != 0 {
        -2
    } else {
        lifetime.saturating_sub(world_tick.saturating_sub(d.start_tick))
    };

    Some(Projectile {
        weapon_type: d.weapon_type,
        owner,
        pos: d.start_pos,
        direction: d.start_vel,
        init_dir: d.start_vel,
        life_span,
        start_tick: d.start_tick,
        freeze,
        explosive,
        bouncing,
        tune_zone,
        layer: if d.switch_number > 0 {
            Layer::Switch
        } else {
            Layer::Game
        },
        number: d.switch_number,
        marked_for_destroy: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::map::{MapData, TILE_SOLID, Tile};

    fn open_collision() -> Collision<f32> {
        let (w, h) = (8u32, 8u32);
        let mut game = vec![Tile::default(); (w * h) as usize];
        for x in 0..w {
            game[(7 * w + x) as usize] = Tile {
                index: TILE_SOLID,
                ..Default::default()
            };
        }
        let map = MapData {
            width: w,
            height: h,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        Collision::new(&map)
    }

    fn ddnet(owner: i32, vel: (i32, i32), flags: i32) -> ProjectileView {
        ProjectileView::DDNet(objects::DDNetProjectile {
            x: 12_345,
            y: 6_789,
            vel_x: vel.0,
            vel_y: vel.1,
            type_: WEAPON_SHOTGUN,
            start_tick: 1_000,
            owner,
            switch_number: 0,
            tune_zone: 0,
            flags,
        })
    }

    #[test]
    fn ddnet_map_cannon_reads_position_and_1e6_direction() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        let flags = flag::BOUNCE_VERTICAL | flag::FREEZE | flag::EXPLOSIVE;
        let p = projectile_from_view(&ddnet(-1, (0, -1_000_000), flags), 1_010, &c, &t).unwrap();
        assert_eq!((p.pos.x, p.pos.y), (123.45, 67.89));
        assert_eq!((p.direction.x, p.direction.y), (0.0, -1.0));
        assert_eq!(p.owner, -1);
        assert_eq!(p.bouncing, 2);
        assert!(p.freeze && p.explosive);
        assert_eq!(p.start_tick, 1_000);
        assert_eq!(
            p.life_span, -2,
            "ownerless bouncing shotgun = map cannon: never expires"
        );
        assert_eq!(p.layer, Layer::Game);
    }

    #[test]
    fn ddnet_owned_projectile_normalises_the_integer_aim_vector() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        let mut v = ddnet(3, (300, 400), flag::NORMALIZE_VEL);
        if let ProjectileView::DDNet(p) = &mut v {
            p.type_ = WEAPON_GRENADE;
        }
        let p = projectile_from_view(&v, 1_010, &c, &t).unwrap();
        assert_eq!((p.direction.x, p.direction.y), (0.6, 0.8));
        assert_eq!(p.owner, 3);
        // Default grenade lifetime is 2 s = 100 ticks; 10 already elapsed.
        assert_eq!(p.life_span, 100 - 10);
    }

    #[test]
    fn legacy_item_with_ddnet_flag_is_read_as_a_ddrace_projectile() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        // Owner 5, freeze + explosive + horizontal bounce, angle pi/2 => direction (-1, 0)
        // (`Angle = -atan2(dx, dy)` on the server: dir (-1, 0) => atan2(-1, 0) = -pi/2 => angle = +pi/2).
        let data =
            5 | legacy_flag::IS_DDNET | legacy_flag::FREEZE | legacy_flag::EXPLOSIVE | legacy_flag::BOUNCE_HORIZONTAL;
        let v = ProjectileView::Legacy(objects::Projectile {
            x: 5_000,
            y: 7_000,
            vel_x: (std::f32::consts::FRAC_PI_2 * 1e6) as i32,
            vel_y: data,
            type_: WEAPON_GRENADE,
            start_tick: 500,
        });
        let p = projectile_from_view(&v, 505, &c, &t).unwrap();
        assert_eq!((p.pos.x, p.pos.y), (50.0, 70.0));
        assert!(
            (p.direction.x + 1.0).abs() < 1e-5 && p.direction.y.abs() < 1e-5,
            "{:?}",
            p.direction
        );
        assert_eq!((p.owner, p.bouncing), (5, 1));
        assert!(p.freeze && p.explosive);
    }

    #[test]
    fn ddrace_no_owner_flag_and_out_of_range_owner_mean_ownerless() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        let mk = |data| {
            ProjectileView::DDRace(objects::DDRaceProjectile {
                x: 0,
                y: 0,
                angle: 0,
                data,
                type_: WEAPON_GUN,
                start_tick: 10,
            })
        };
        let p = projectile_from_view(&mk(7 | legacy_flag::NO_OWNER), 10, &c, &t).unwrap();
        assert_eq!(p.owner, -1);
        let p = projectile_from_view(&mk(200), 10, &c, &t).unwrap();
        assert_eq!(p.owner, -1, "owner >= MAX_CLIENTS is not trusted");
        assert_eq!(
            (p.direction.x, p.direction.y),
            (0.0, 1.0),
            "angle 0 points down (sin -0, cos 0)"
        );
    }

    #[test]
    fn plain_legacy_item_has_no_owner_and_guesses_explosive_from_direction_length() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        let mk = |ty, vx, vy| {
            ProjectileView::Legacy(objects::Projectile {
                x: 100,
                y: 100,
                vel_x: vx,
                vel_y: vy,
                type_: ty,
                start_tick: 10,
            })
        };
        let g = projectile_from_view(&mk(WEAPON_GRENADE, 100, 0), 10, &c, &t).unwrap();
        assert!(g.explosive && g.owner == -1 && g.bouncing == 0 && !g.freeze);
        assert_eq!((g.pos.x, g.pos.y), (100.0, 100.0), "legacy positions are whole pixels");
        let gun = projectile_from_view(&mk(WEAPON_GUN, 100, 0), 10, &c, &t).unwrap();
        assert!(!gun.explosive);
    }

    #[test]
    fn items_the_client_skips_or_the_world_cannot_model_are_dropped() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        let mk = |ty, vx, vy| {
            ProjectileView::Legacy(objects::Projectile {
                x: 0,
                y: 0,
                vel_x: vx,
                vel_y: vy,
                type_: ty,
                start_tick: 0,
            })
        };
        // Ball-mod grenade with a non-unit direction (`gameworld.cpp:447-448`).
        assert!(projectile_from_view(&mk(WEAPON_GRENADE, 300, 0), 1, &c, &t).is_none());
        // ... but a shotgun is exempt from that filter.
        assert!(projectile_from_view(&mk(WEAPON_SHOTGUN, 300, 0), 1, &c, &t).is_some());
        // Types with no projectile model (hammer, laser, ninja).
        for ty in [0, 4, 5, 99, -1] {
            assert!(projectile_from_view(&mk(ty, 100, 0), 1, &c, &t).is_none(), "type {ty}");
        }
        // NaN/inf can only come from a zero divisor here; a zero aim vector normalises to zero and is
        // rejected by the unit-length filter for non-shotguns.
        let zero_aim = ddnet(2, (0, 0), flag::NORMALIZE_VEL);
        let mut zero_aim_gun = zero_aim;
        if let ProjectileView::DDNet(p) = &mut zero_aim_gun {
            p.type_ = WEAPON_GUN;
        }
        assert!(projectile_from_view(&zero_aim_gun, 1, &c, &t).is_none());
    }

    #[test]
    fn switch_number_selects_the_switch_layer_and_zone_is_bounds_checked() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        let mut v = ddnet(-1, (0, 1_000_000), flag::BOUNCE_VERTICAL);
        if let ProjectileView::DDNet(p) = &mut v {
            p.switch_number = 4;
            p.tune_zone = 100_000;
        }
        let p = projectile_from_view(&v, 1_001, &c, &t).unwrap();
        assert_eq!((p.layer, p.number), (Layer::Switch, 4));
        assert_eq!(
            p.tune_zone, 0,
            "an out-of-range tune zone must not index past the zone table"
        );
    }

    #[test]
    fn an_old_shotgun_that_is_not_a_map_cannon_expires_like_the_client_says() {
        let c = open_collision();
        let t = TuningList::reset_to_baseline();
        // Owned, non-bouncing shotgun: 20 s minus the elapsed ticks, exactly `projectile.cpp:209`.
        let p = projectile_from_view(&ddnet(1, (0, 1), flag::NORMALIZE_VEL), 1_250, &c, &t).unwrap();
        assert_eq!(p.life_span, 1_000 - 250);
    }
}
