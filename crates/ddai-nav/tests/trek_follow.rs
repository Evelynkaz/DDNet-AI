//! Unit-level behaviour of the trek / seek / follow rules (`bot.ts` `busiestSpot`, `gameSpot`,
//! `startTrek`, `trekGoal`, `steerFollow`), on small synthetic rooms. The TS-side comparison of the
//! follow mode is `arrival_vs_ts.rs`.

use std::collections::HashSet;
use std::sync::Arc;

use ddai_nav::follow::{FOLLOW_MAX_TICKS, Follow, FollowCtx, FollowVerdict, follow_tile};
use ddai_nav::navigator::NavPhase;
use ddai_nav::route::Router;
use ddai_nav::trek::{TREK_STALL_TICKS, Trek, busiest_spot, game_spot};
use ddai_nav::wayblock::wayblocks;
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::{TeeState, blank_tee_state};
use ddai_planner::vmath::Vec2;

fn room(w: u32, h: u32, extra: &[(u32, u32, u8)]) -> MapData {
    let mut game = vec![Tile::default(); (w * h) as usize];
    for x in 0..w {
        game[x as usize].index = TILE_SOLID;
        game[((h - 1) * w + x) as usize].index = TILE_SOLID;
    }
    for y in 0..h {
        game[(y * w) as usize].index = TILE_SOLID;
        game[(y * w + w - 1) as usize].index = TILE_SOLID;
    }
    for &(x, y, index) in extra {
        game[(y * w + x) as usize].index = index;
    }
    MapData {
        width: w,
        height: h,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

fn tee(id: i32, x: f64, y: f64) -> TeeState {
    let mut t = blank_tee_state();
    t.id = id;
    t.alive = true;
    t.pos = Vec2 { x, y };
    t
}

fn world(map: MapData) -> PhysicsWorld {
    PhysicsWorld::new(Arc::new(map), 1)
}

#[test]
fn the_busiest_spot_counts_crowd_and_fights_minus_distance_and_is_none_when_near() {
    let from = Vec2 { x: 100.0, y: 100.0 };
    // A crowd of three far away, one of them mid-swing; a lone tee elsewhere.
    let mut crowd = [tee(1, 3000.0, 100.0), tee(2, 3050.0, 120.0), tee(3, 3100.0, 90.0)];
    crowd[0].attack_tick = 990;
    let lone = tee(4, 100.0, 2500.0);
    let all: Vec<TeeState> = crowd.iter().cloned().chain([lone]).collect();
    let awake = |_: &TeeState| true;
    let spot = busiest_spot(from, &all, 1000, &awake).expect("a spot");
    assert_eq!((spot.tees, spot.busy), (3, 1), "the crowd, one of them busy");
    assert!(spot.x > 2900.0, "{spot:?}");
    // Sleeping tees are not a crowd.
    let napping = |t: &TeeState| t.id == 4;
    let only_lone = busiest_spot(from, &all, 1000, &napping).expect("the lone one");
    assert_eq!(only_lone.tees, 1);
    // Closer than 800 px is "already here": no trip.
    let near = [tee(1, 500.0, 100.0), tee(2, 520.0, 100.0)];
    assert!(busiest_spot(from, &near, 1000, &awake).is_none());
    assert!(busiest_spot(from, &[], 1000, &awake).is_none());
}

#[test]
fn a_game_spot_inside_a_wb_avoid_zone_is_refused() {
    let clb = wayblocks()
        .into_iter()
        .find(|d| d.name == "Copy Love Box")
        .expect("CLB");
    let b = clb.avoid[0];
    let inside = tee(1, f64::from((b.x0 + 2) * 32), f64::from((b.y0 + 2) * 32));
    let from = Vec2 { x: 100.0, y: 100.0 };
    let awake = |_: &TeeState| true;
    assert!(
        game_spot(None, from, std::slice::from_ref(&inside), 1000, &awake).is_some(),
        "without a WB it is fine"
    );
    assert!(
        game_spot(Some(&clb), from, std::slice::from_ref(&inside), 1000, &awake).is_none(),
        "the WB avoids it"
    );
}

#[test]
fn a_trek_refuses_a_goal_it_cannot_reach_and_one_that_stops_short_at_a_freeze() {
    // A wall across the room with a freeze strip in front of it: the best partial route ends two tiles
    // short of the goal beside the freeze, which `startTrek` refuses (the walk would only end at a hazard).
    let mut walled: Vec<(u32, u32, u8)> = (1..11).map(|y| (20, y, TILE_SOLID)).collect();
    walled.extend((1..11).map(|y| (19, y, TILE_FREEZE)));
    let w = world(room(40, 12, &walled));
    let mut router = Router::new(w.collision(), &[]);
    let avoid = HashSet::new();
    let from = Vec2 {
        x: 5.0 * 32.0 + 16.0,
        y: 10.0 * 32.0 + 16.0,
    };
    let err = Trek::start(
        &mut router,
        w.collision(),
        from,
        (30.0 * 32.0 + 16.0, 10.0 * 32.0 + 16.0),
        &avoid,
        0,
    )
    .err();
    assert!(err.is_some_and(|e| e.contains("no route")), "stops short at a freeze");
    // The same wall without the freeze: a partial route to the foot of it is fine.
    let plain: Vec<(u32, u32, u8)> = (1..11).map(|y| (20, y, TILE_SOLID)).collect();
    let p = world(room(40, 12, &plain));
    let mut rp = Router::new(p.collision(), &[]);
    assert!(
        Trek::start(
            &mut rp,
            p.collision(),
            from,
            (30.0 * 32.0 + 16.0, 10.0 * 32.0 + 16.0),
            &avoid,
            0
        )
        .is_ok()
    );
    // An open room: a trek exists.
    let open = world(room(40, 12, &[]));
    let mut r2 = Router::new(open.collision(), &[]);
    let trek = Trek::start(
        &mut r2,
        open.collision(),
        from,
        (30.0 * 32.0 + 16.0, 10.0 * 32.0 + 16.0),
        &avoid,
        0,
    )
    .expect("route");
    assert!(trek.remaining() > 0 && trek.describe().starts_with("walking over"));
}

#[test]
fn a_trek_ends_at_its_last_step_and_a_stalled_step_is_banned() {
    let open = world(room(60, 12, &[]));
    let mut router = Router::new(open.collision(), &[]);
    let mut avoid = HashSet::new();
    let from = Vec2 {
        x: 5.0 * 32.0 + 16.0,
        y: 10.0 * 32.0 + 16.0,
    };
    let to = (50.0 * 32.0 + 16.0, 10.0 * 32.0 + 16.0);
    let mut trek = Trek::start(&mut router, open.collision(), from, to, &avoid, 0).expect("route");
    // Standing still: the first step's goal is returned until the stall window passes.
    let first = trek.goal(from, 1, true, &mut avoid);
    assert!(first.goal.is_some() && !first.ended);
    let stalled = trek.goal(from, TREK_STALL_TICKS + 5, true, &mut avoid);
    assert!(stalled.ended && stalled.goal.is_none(), "the stall ends the trek");
    assert!(stalled.note.as_deref().is_some_and(|n| n.contains("stalled")));
    assert_eq!(avoid.len(), 1, "the step's move is banned for the next search");
    // Walking the whole way (the tee is placed at every step in turn) ends it without a ban.
    let mut avoid2 = HashSet::new();
    let mut trek = Trek::start(&mut router, open.collision(), from, to, &avoid2, 0).expect("route");
    let mut me = from;
    let mut done = false;
    for t in 0..200 {
        let step = trek.goal(me, t, true, &mut avoid2);
        if step.ended {
            done = true;
            break;
        }
        me = step.goal.expect("a goal");
    }
    assert!(done, "reached the end");
    assert!(avoid2.is_empty());
}

fn follow_ctx<'a>(
    tick: i64,
    me: &'a TeeState,
    target: Option<&'a TeeState>,
    phase: NavPhase,
    goal: Option<(i32, i32)>,
) -> FollowCtx<'a> {
    FollowCtx {
        tick,
        me,
        target,
        target_away: false,
        target_on_server: true,
        nav_phase: phase,
        nav_outcome: "walled off",
        goal,
    }
}

#[test]
fn follow_tile_prefers_a_tile_with_a_floor_and_never_a_freeze_or_a_wall() {
    let w = world(room(40, 12, &[(20, 10, TILE_FREEZE)]));
    // The target hangs in the air above the floor row: the goal is a tile standing on it.
    let t = follow_tile(
        w.collision(),
        Vec2 {
            x: 20.0 * 32.0 + 16.0,
            y: 7.0 * 32.0 + 16.0,
        },
    )
    .expect("a tile");
    assert!(t.1 >= 7 && t.1 <= 10, "{t:?}");
    let col = w.collision();
    let (px, py) = (f64::from(t.0 * 32 + 16), f64::from(t.1 * 32 + 16));
    assert!(!ddai_planner::plan_world::PlanCollision::is_solid(col, px, py));
    assert!(!ddai_planner::plan_world::PlanCollision::is_freeze(col, px, py));
    // Deep inside a wall block there is none.
    let walls: Vec<(u32, u32, u8)> = (1..11)
        .flat_map(|y| (10..30).map(move |x| (x, y, TILE_SOLID)))
        .collect();
    let solid = world(room(40, 12, &walls));
    assert!(
        follow_tile(
            solid.collision(),
            Vec2 {
                x: 20.0 * 32.0,
                y: 6.0 * 32.0
            }
        )
        .is_none()
    );
}

#[test]
fn follow_ends_when_it_arrives_loses_them_gives_up_or_counts_deaths() {
    let me = tee(0, 100.0, 100.0);
    let near = tee(1, 130.0, 100.0);
    let far = tee(1, 1000.0, 100.0);
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    assert_eq!(
        f.steer(&follow_ctx(10, &me, Some(&far), NavPhase::Walking, Some((30, 3)))),
        FollowVerdict::Go
    );
    // Within 64 px and not frozen: arrived.
    assert_eq!(
        f.steer(&follow_ctx(20, &me, Some(&near), NavPhase::Walking, Some((4, 3)))),
        FollowVerdict::End("arrived".to_string())
    );
    // Frozen next to them is not arrival.
    let mut frozen = me;
    frozen.frozen = true;
    let mut f = Follow::new(1, (4, 3), me.pos, near.pos, 0);
    assert_eq!(
        f.steer(&follow_ctx(5, &frozen, Some(&near), NavPhase::Walking, Some((4, 3)))),
        FollowVerdict::Go
    );

    // No tee for 5 s ends it; before that it waits.
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    assert_eq!(
        f.steer(&follow_ctx(100, &me, None, NavPhase::Walking, None)),
        FollowVerdict::Wait
    );
    assert!(f.waiting);
    assert!(matches!(
        f.steer(&follow_ctx(300, &me, None, NavPhase::Walking, None)),
        FollowVerdict::End(_)
    ));

    // Left the server.
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    let mut c = follow_ctx(10, &me, Some(&far), NavPhase::Walking, None);
    c.target_on_server = false;
    assert!(matches!(f.steer(&c), FollowVerdict::End(m) if m.contains("left the server")));

    // 120 s of walking is the limit.
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    assert!(matches!(
        f.steer(&follow_ctx(FOLLOW_MAX_TICKS + 1, &me, Some(&far), NavPhase::Walking, None)),
        FollowVerdict::End(m) if m.contains("gave up")
    ));

    // Three deaths of the target, or three of ours that were not our own kill.
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    for _ in 0..3 {
        f.on_kill(1, 0, 100, -1000);
    }
    assert!(
        matches!(f.steer(&follow_ctx(110, &me, Some(&far), NavPhase::Walking, None)), FollowVerdict::End(m) if m.contains("died"))
    );
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    for _ in 0..3 {
        f.on_kill(0, 0, 100, 90); // our own Cl_Kill 10 ticks ago: not counted
    }
    assert_eq!(
        f.steer(&follow_ctx(110, &me, Some(&far), NavPhase::Walking, None)),
        FollowVerdict::Go
    );
    for _ in 0..3 {
        f.on_kill(0, 0, 400, 90);
    }
    assert!(
        matches!(f.steer(&follow_ctx(410, &me, Some(&far), NavPhase::Walking, None)), FollowVerdict::End(m) if m.contains("died"))
    );
}

#[test]
fn follow_retries_a_blocked_walk_as_they_move_and_gives_up_after_three() {
    let me = tee(0, 100.0, 100.0);
    let far = tee(1, 1000.0, 100.0);
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    // Blocked: waits (the goal did not change) for a second, then re-aims at the same tile.
    assert_eq!(
        f.steer(&follow_ctx(10, &me, Some(&far), NavPhase::Blocked, Some((30, 3)))),
        FollowVerdict::Wait
    );
    assert_eq!(
        f.steer(&follow_ctx(40, &me, Some(&far), NavPhase::Blocked, Some((30, 3)))),
        FollowVerdict::Wait
    );
    assert_eq!(
        f.steer(&follow_ctx(70, &me, Some(&far), NavPhase::Blocked, Some((30, 3)))),
        FollowVerdict::Reroute((30, 3))
    );
    // A moved target re-aims at once, and a third distinct "no way" ends it.
    assert_eq!(
        f.steer(&follow_ctx(71, &me, Some(&far), NavPhase::Blocked, Some((32, 3)))),
        FollowVerdict::Reroute((32, 3))
    );
    assert!(matches!(
        f.steer(&follow_ctx(72, &me, Some(&far), NavPhase::Blocked, Some((34, 3)))),
        FollowVerdict::End(m) if m.contains("no way")
    ));
}

#[test]
fn follow_reroutes_after_arriving_short_when_they_moved_or_jumped() {
    let me = tee(0, 100.0, 100.0);
    let far = tee(1, 1000.0, 100.0);
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    // Arrived at the same goal tile and they did not move: "as near as it gets".
    assert!(matches!(
        f.steer(&follow_ctx(10, &me, Some(&far), NavPhase::Arrived, Some((30, 3)))),
        FollowVerdict::End(m) if m.contains("as near as")
    ));
    // They teleported (jumped > 256 px): re-aim even at the same tile.
    let mut f = Follow::new(1, (30, 3), me.pos, far.pos, 0);
    f.steer(&follow_ctx(10, &me, Some(&far), NavPhase::Walking, Some((30, 3))));
    let jumped = tee(1, 2000.0, 100.0);
    assert_eq!(
        f.steer(&follow_ctx(20, &me, Some(&jumped), NavPhase::Walking, Some((62, 3)))),
        FollowVerdict::Reroute((62, 3))
    );
}
