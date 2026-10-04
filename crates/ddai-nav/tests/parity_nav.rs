//! Task 4.2 acceptance criterion 2: parity of the deterministic navigation code with the real TS
//! sources. `#[ignore]`d and gated on `ts-parity` (needs the f64 `ddai-tsworld`) and on dumps made by
//! `tools/ts-trace/gen-nav-dump.mjs`:
//!
//! ```text
//! node tools/ts-trace/gen-nav-dump.mjs --map "<map>" --seed 1 --routes 1500 --out ~/aiddnet/data/traces/nav/clb.jsonl
//! DDAI_NAV_DUMP=~/aiddnet/data/traces/nav/clb.jsonl \
//!   cargo test -p ddai-nav --features ts-parity --release --test parity_nav -- --ignored --nocapture
//! ```
//!
//! Every `route` line is a `findRoute` query (positions, options, the expected steps and cost); the
//! `deadzone`/`spawns` lines are the whole-map answers.

#![cfg(feature = "ts-parity")]

use ddai_nav::crossing::{CrossPhase, SwingCrosser};
use ddai_nav::crossing::{Crossing, TileBox, WallRoute};
use ddai_nav::navigator::{NavCtx, NavOpts, Navigator, tile_goal};
use ddai_nav::route::{MoveKind, RouteOpts, Router, dead_zone, spawn_tiles};
use ddai_nav::wayblock::{
    WbDef, WbSide, WbSideChooser, find_hall_offset, has_wayblock_named, wayblock_for, wayblocks, wb_spot,
};
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::{PlayerInput, TeeState};
use ddai_planner::vmath::Vec2;
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Deserialize)]
struct StepJson {
    x: i32,
    y: i32,
    kind: String,
    ax: i32,
    ay: i32,
    freeze: bool,
    tele: bool,
    #[serde(rename = "move")]
    mv: i32,
    leap: bool,
}

#[derive(Deserialize)]
struct ResultJson {
    cost: i32,
    steps: Vec<StepJson>,
}

#[derive(Deserialize)]
struct OptsJson {
    #[serde(rename = "nearTiles")]
    near_tiles: i32,
    partial: bool,
    #[serde(rename = "allowKill")]
    allow_kill: bool,
    #[serde(rename = "throughFreeze")]
    through_freeze: bool,
    #[serde(rename = "maxNodes")]
    max_nodes: usize,
    avoid: Vec<i32>,
}

/// `(tx, ty, side, then seven flag bytes)` of a `wbzones` row.
type ZoneRow = (i32, i32, String, u8, u8, u8, u8, u8, u8, u8);

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Line {
    Meta {
        #[serde(rename = "mapPath")]
        map_path: String,
        #[serde(rename = "mapSha256")]
        map_sha256: String,
        /// The TS reference the dump was made with (`DDAI_TS_REF`), for the log.
        #[serde(rename = "tsRef", default)]
        ts_ref: String,
        /// Whether that reference has the hall search (af49dfb) or not (c3c619d, the first corpus).
        #[serde(default)]
        hall: bool,
    },
    Hall {
        found: bool,
        dx: i32,
        dy: i32,
        #[serde(rename = "match")]
        matched: String,
    },
    Wbdef {
        name: String,
        def: Option<serde_json::Value>,
    },
    Wbguard {
        def: String,
        side: String,
        geom: serde_json::Value,
    },
    Spawns {
        tiles: Vec<(f64, f64)>,
    },
    Deadzone {
        first: u8,
        runs: Vec<usize>,
        count: usize,
    },
    Route {
        from: (f64, f64),
        to: (f64, f64),
        opts: OptsJson,
        result: Option<ResultJson>,
    },
    Wayblockfor {
        cases: Vec<ForCase>,
    },
    Wbzones {
        def: String,
        rows: Vec<ZoneRow>,
    },
    Wbchooser {
        ops: Vec<ChooserOp>,
    },
    Wbspot {
        side: String,
        here: Option<(i32, i32)>,
        tees: Vec<SpotTee>,
        friends: Vec<i32>,
        spot: (i32, i32),
    },
    Navtrace {
        from: (i32, i32),
        to: (i32, i32),
        #[serde(rename = "throughFreeze")]
        through_freeze: bool,
        #[serde(rename = "withCrossings")]
        with_crossings: bool,
        #[serde(rename = "wallRoute", default)]
        wall_route: bool,
        #[serde(rename = "maxTicks")]
        max_ticks: i64,
        phase: String,
        outcome: String,
        ticks: i64,
        kills: Vec<i64>,
        notes: Vec<String>,
        trace: Vec<(i32, String, String, i32, i32, i32)>,
    },
    Crosstrace {
        crossing: usize,
        lag: i64,
        #[serde(rename = "useWall", default)]
        use_wall: bool,
        start: (f64, f64),
        ended: String,
        reason: String,
        ticks: usize,
        /// `[tick, doing]` at every change of the crosser's description (af49dfb dumps).
        #[serde(default)]
        doings: Vec<(usize, String)>,
        trace: Vec<(i32, String, String, i32, i32)>,
    },
}

#[derive(Deserialize)]
struct ForCase {
    name: String,
    #[serde(rename = "withCol")]
    with_col: Option<String>,
    #[serde(rename = "noCol")]
    no_col: Option<String>,
}

#[derive(Deserialize)]
struct ChooserOp {
    op: String,
    arg: Option<String>,
    side: Option<String>,
    left: Option<i32>,
    right: Option<i32>,
    here: Option<String>,
    tick: Option<i64>,
    nearer: Option<String>,
}

#[derive(Deserialize)]
struct SpotTee {
    id: i32,
    alive: bool,
    frozen: bool,
    x: String,
    y: String,
}

fn boxes_json(v: &[TileBox]) -> serde_json::Value {
    serde_json::json!(v.iter().map(box_json).collect::<Vec<_>>())
}

fn box_json(b: &TileBox) -> serde_json::Value {
    serde_json::json!({"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1})
}

fn tile_json(t: (i32, i32)) -> serde_json::Value {
    serde_json::json!({"tx": t.0, "ty": t.1})
}

fn tiles_json(v: &[(i32, i32)]) -> serde_json::Value {
    serde_json::json!(v.iter().map(|&t| tile_json(t)).collect::<Vec<_>>())
}

fn wall_json(w: &Option<WallRoute>) -> serde_json::Value {
    match w {
        None => serde_json::Value::Null,
        Some(w) => serde_json::json!({
            "anchors": tiles_json(&w.anchors), "shelf": boxes_json(&w.shelf), "room": boxes_json(&w.room),
            "lastRow": w.last_row, "missRow": w.miss_row,
        }),
    }
}

fn crossing_json(c: &Crossing) -> serde_json::Value {
    serde_json::json!({
        "label": c.label, "from": boxes_json(&c.from), "chamber": box_json(&c.chamber), "start": tile_json(c.start),
        "anchors": tiles_json(&c.anchors), "directAnchors": c.direct_anchors, "landing": boxes_json(&c.landing),
        "exit": boxes_json(&c.exit), "exitTile": tile_json(c.exit_tile),
        "hall": c.hall.as_deref().map_or(serde_json::Value::Null, boxes_json),
        "hallTile": c.hall_tile.map_or(serde_json::Value::Null, tile_json),
        "toward": c.toward, "wall": wall_json(&c.wall),
    })
}

/// The TS `WbDef` JSON of `gen-nav-dump.mjs` (`defJson`).
fn def_json(d: &WbDef) -> serde_json::Value {
    let side = |s: &ddai_nav::wayblock::WbSideDef| {
        serde_json::json!({
            "zone": boxes_json(&s.zone), "approach": boxes_json(&s.approach), "leash": boxes_json(&s.leash),
            "spots": tiles_json(&s.spots), "watch": tile_json(s.watch), "crossing": crossing_json(&s.crossing),
        })
    };
    serde_json::json!({
        "name": d.name, "size": {"w": d.size.0, "h": d.size.1}, "avoid": boxes_json(&d.avoid),
        "left": side(&d.left), "right": side(&d.right),
        "crossings": d.crossings.iter().map(crossing_json).collect::<Vec<_>>(),
    })
}

fn side_of(s: &str) -> WbSide {
    if s == "left" { WbSide::Left } else { WbSide::Right }
}

fn bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("f64 bits"))
}

fn kind_of(k: MoveKind) -> &'static str {
    match k {
        MoveKind::Walk => "walk",
        MoveKind::Fall => "fall",
        MoveKind::Jump => "jump",
        MoveKind::Hook => "hook",
        MoveKind::Kill => "kill",
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
#[ignore = "needs the ts-parity feature and a dump from tools/ts-trace/gen-nav-dump.mjs (DDAI_NAV_DUMP)"]
fn routes_dead_zone_and_spawns_match_ts() {
    let path = std::env::var("DDAI_NAV_DUMP").expect("set DDAI_NAV_DUMP");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Line::Meta {
        map_path,
        map_sha256,
        ts_ref,
        hall: dump_has_hall,
    } = serde_json::from_str(lines.next().expect("empty dump")).expect("meta")
    else {
        panic!("first line must be meta")
    };
    // A dump of the first corpus (c3c619d, no hall search) says nothing about what af49dfb changed: the
    // wayblock lookup, the spots (the guard's order), the crossings and the navigator walks. It still
    // proves the routes, the dead zone, the spawns, the side chooser and the zone predicates.
    let legacy = !dump_has_hall;
    println!(
        "dump of {map_path} (TS reference {}, {})",
        if ts_ref.is_empty() { "?" } else { &ts_ref },
        if legacy {
            "legacy: routes/deadzone/spawns/chooser/zones only"
        } else {
            "af49dfb: everything"
        }
    );
    let bytes = std::fs::read(&map_path).unwrap_or_else(|e| panic!("reading map {map_path}: {e}"));
    {
        use sha2::{Digest, Sha256};
        assert_eq!(
            hex(&Sha256::digest(&bytes)),
            map_sha256,
            "the map changed since the dump"
        );
    }
    let ts = ddai_tsworld::load_map_bytes(&bytes).expect("load the map (ddai-tsworld)");
    let map = ddai_map::load_map(&bytes).expect("load the map (ddai-map)").data;
    let spawns = spawn_tiles(&map);
    let mut router = Router::new(&ts.collision, &spawns);

    let (mut routes, mut route_bad, mut found, mut hooks, mut freezes, mut kills, mut teles) = (0, 0, 0, 0, 0, 0, 0);
    let (mut wb_checked, mut chooser_n, mut chooser_bad, mut spot_n, mut spot_bad) = (0, 0, 0, 0, 0);
    let (mut wb_def_bad, mut legacy_skipped) = (0usize, 0usize);
    let (mut cross_n, mut cross_bad, mut cross_arrived) = (0usize, 0usize, 0usize);
    let (mut nav_n, mut nav_bad, mut nav_arrived) = (0usize, 0usize, 0usize);
    let mut dead_checked = false;
    let mut spawns_checked = false;
    for line in lines {
        match serde_json::from_str::<Line>(line).unwrap_or_else(|e| panic!("bad line: {e}")) {
            Line::Meta { .. } => panic!("second meta"),
            Line::Spawns { tiles } => {
                assert_eq!(tiles, spawns, "spawnTiles");
                spawns_checked = true;
            }
            Line::Deadzone { first, runs, count } => {
                let got = dead_zone(&router.grid, &spawns);
                let mut want = Vec::with_capacity(got.len());
                let mut cur = first;
                for r in runs {
                    want.extend(std::iter::repeat_n(cur, r));
                    cur ^= 1;
                }
                assert_eq!(want.len(), got.len(), "dead zone size");
                assert_eq!(want.iter().map(|&v| usize::from(v)).sum::<usize>(), count);
                let bad = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                println!("deadZone: {} dead tiles, {bad} mismatches", count);
                assert_eq!(bad, 0, "deadZone differs from TS in {bad} tiles");
                dead_checked = true;
            }
            Line::Hall { found, dx, dy, matched } => {
                let got = find_hall_offset(&ts.collision);
                assert_eq!(got.is_some(), found, "findHallOffset found / not found");
                if let Some(g) = got {
                    assert_eq!((g.dx, g.dy), (dx, dy), "findHallOffset offset");
                    assert_eq!(
                        g.matched.to_bits(),
                        u64::from_str_radix(&matched, 16).expect("match bits"),
                        "findHallOffset match"
                    );
                }
                println!("findHallOffset: {:?}", got.map(|g| (g.dx, g.dy, g.matched)));
                wb_checked += 1;
            }
            Line::Wbdef { .. } if legacy => {
                legacy_skipped += 1;
            }
            Line::Wbdef { name, def } => {
                let got = wayblock_for(&name, Some(&ts.collision)).as_ref().map(def_json);
                if got != def {
                    wb_def_bad += 1;
                    eprintln!("wayblockFor({name:?}) differs:\n  rust {got:?}\n  ts   {def:?}");
                }
                wb_checked += 1;
            }
            Line::Wbguard { def, side, geom } => {
                let d = wayblock_for("Copy Love Box", Some(&ts.collision)).expect("def");
                assert_eq!(d.name, def, "the definition the guard geometry was dumped for");
                let g = d.guard_geom(side_of(&side));
                let boxj = |b: &TileBox| serde_json::json!({"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1});
                let got = serde_json::json!({
                    "shelf": boxj(&g.shelf), "column": boxj(&g.column), "landing": boxj(&g.landing),
                    "foot": boxj(&g.foot), "passage": boxj(&g.passage), "corridor": boxj(&g.corridor),
                    "job": {"tx": g.job.0, "ty": g.job.1}, "stepOff": {"tx": g.step_off.0, "ty": g.step_off.1},
                });
                assert_eq!(got, geom, "wbGuardGeom({def}, {side})");
                wb_checked += 1;
            }
            Line::Wayblockfor { .. } if legacy => {}
            Line::Wayblockfor { cases } => {
                for c in cases {
                    let with = wayblock_for(&c.name, Some(&ts.collision)).map(|d| d.name.to_string());
                    let no = has_wayblock_named(&c.name).then(|| {
                        wayblock_for::<ddai_tsworld::Collision>(&c.name, None)
                            .map(|d| d.name.to_string())
                            .expect("named")
                    });
                    assert_eq!(with, c.with_col, "wayblockFor({:?}, col)", c.name);
                    assert_eq!(no, c.no_col, "wayblockFor({:?})", c.name);
                }
                wb_checked += 1;
            }
            Line::Wbzones { def, rows } => {
                let d = wayblocks()
                    .into_iter()
                    .find(|d| d.name == def)
                    .or_else(|| wayblock_for("Copy Love Box", Some(&ts.collision)).filter(|d| d.name == def))
                    .expect("def");
                let mut bad = 0;
                for (tx, ty, side_at, zl, zr, hl, hr, ll, lr, walk) in &rows {
                    let got_side = d.side_at(*tx, *ty).map_or("-", WbSide::name);
                    let ok = got_side == side_at
                        && u8::from(d.in_zone(WbSide::Left, *tx, *ty)) == *zl
                        && u8::from(d.in_zone(WbSide::Right, *tx, *ty)) == *zr
                        && u8::from(d.in_hall(WbSide::Left, *tx, *ty)) == *hl
                        && u8::from(d.in_hall(WbSide::Right, *tx, *ty)) == *hr
                        && u8::from(d.in_leash(WbSide::Left, *tx, *ty)) == *ll
                        && u8::from(d.in_leash(WbSide::Right, *tx, *ty)) == *lr
                        && u8::from(d.walk_allowed(*tx, *ty)) == *walk;
                    bad += usize::from(!ok);
                }
                println!("wbzones {def}: {} tiles, {bad} mismatches", rows.len());
                assert_eq!(bad, 0, "WB zone predicates differ from TS ({def})");
                wb_checked += 1;
            }
            Line::Wbchooser { ops } => {
                chooser_n += 1;
                let mut ch = WbSideChooser::default();
                let mut bad = false;
                for o in &ops {
                    let got = match o.op.as_str() {
                        "reset" => {
                            ch.reset();
                            ch.side
                        }
                        "adopt" => {
                            ch.adopt(side_of(o.arg.as_deref().expect("arg")));
                            ch.side
                        }
                        _ => {
                            let here = o.here.as_deref().map(side_of);
                            Some(ch.update(
                                (o.left.expect("left"), o.right.expect("right")),
                                here,
                                o.tick.expect("tick"),
                                side_of(o.nearer.as_deref().expect("nearer")),
                            ))
                        }
                    };
                    let want = o.side.as_deref().map(side_of);
                    bad |= got != want;
                }
                if bad {
                    chooser_bad += 1;
                }
            }
            Line::Wbspot { .. } | Line::Navtrace { .. } | Line::Crosstrace { .. } if legacy => {
                legacy_skipped += 1;
            }
            Line::Wbspot {
                side,
                here,
                tees,
                friends,
                spot,
            } => {
                spot_n += 1;
                let def = wayblocks().remove(0);
                let ts_tees: Vec<TeeState> = tees
                    .iter()
                    .map(|t| {
                        let mut st = ddai_planner::types::blank_tee_state();
                        st.id = t.id;
                        st.pos = Vec2 {
                            x: bits(&t.x),
                            y: bits(&t.y),
                        };
                        st.alive = t.alive;
                        st.frozen = t.frozen;
                        st
                    })
                    .collect();
                let got = wb_spot(0, &ts_tees, &|id| friends.contains(&id), &def, side_of(&side), here);
                if got != spot {
                    spot_bad += 1;
                    if spot_bad <= 3 {
                        eprintln!("wbSpot differs: got {got:?}, TS {spot:?} (side {side}, here {here:?})");
                    }
                }
            }
            Line::Navtrace {
                from,
                to,
                through_freeze,
                with_crossings,
                wall_route,
                max_ticks,
                phase,
                outcome,
                ticks,
                kills,
                notes,
                trace,
            } => {
                nav_n += 1;
                let def = wayblock_for("Copy Love Box", Some(&ts.collision));
                let crossings = if with_crossings {
                    def.map(|d| d.crossings).unwrap_or_default()
                } else {
                    Vec::new()
                };
                let mut world = ddai_tsworld::SimWorld::new(
                    ts.collision.clone(),
                    ddai_tsworld::world::SimWorldOptions {
                        respawn_delay_ticks: Some(0),
                        infinite_ammo: Some(true),
                        sv_hit: Some(true),
                        all_weapons: None,
                        no_weak_hook: None,
                    },
                );
                let start = Vec2 {
                    x: f64::from(from.0 * 32 + 16),
                    y: f64::from(from.1 * 32 + 16),
                };
                <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut world, 0, start);
                let goal = tile_goal(&ts.collision, to.0, to.1);
                let mut nav: Navigator<ddai_tsworld::SimWorld> = Navigator::new(
                    vec![goal],
                    NavOpts {
                        through_freeze,
                        crossings,
                        ..NavOpts::default()
                    },
                );
                nav.wall_route = wall_route;
                let mut make = {
                    let template = <ddai_tsworld::SimWorld as PlanWorld>::new_scratch(&world);
                    move || <ddai_tsworld::SimWorld as PlanWorld>::new_scratch(&template)
                };
                let mut tick = 1000i64;
                let mut got_kills: Vec<i64> = Vec::new();
                let mut got_notes: Vec<String> = Vec::new();
                let mut n_spawn = 0usize;
                let mut diverged = false;
                let mut t = 0i64;
                while t < max_ticks {
                    let me = <ddai_tsworld::SimWorld as PlanWorld>::get_tee(&world, 0).expect("tee");
                    let inp = {
                        let mut ctx = NavCtx {
                            col: world.collision(),
                            router: &mut router,
                            make_sim: &mut make,
                        };
                        nav.step(&mut ctx, &me, tick, &[], 0)
                    };
                    got_notes.extend(nav.take_notes());
                    if let Some(want) = trace.get(t as usize) {
                        let got = (
                            inp.direction,
                            inp.target_x.to_bits(),
                            inp.target_y.to_bits(),
                            inp.jump,
                            inp.hook,
                            inp.fire,
                        );
                        let w = (
                            want.0,
                            u64::from_str_radix(&want.1, 16).unwrap(),
                            u64::from_str_radix(&want.2, 16).unwrap(),
                            want.3,
                            want.4,
                            want.5,
                        );
                        if got != w && !diverged {
                            diverged = true;
                            eprintln!(
                                "navtrace {nav_n} {from:?}->{to:?} (freeze {through_freeze}) differs at tick {t}: got {got:?} want {w:?}"
                            );
                        }
                    }
                    if nav.take_kill() {
                        got_kills.push(t);
                        let sp = if spawns.is_empty() {
                            start
                        } else {
                            let s = spawns[n_spawn % spawns.len()];
                            n_spawn += 1;
                            Vec2 { x: s.0, y: s.1 }
                        };
                        let mut st = ddai_planner::types::blank_tee_state();
                        st.id = 0;
                        st.alive = true;
                        st.pos = sp;
                        st.hook_pos = sp;
                        st.jumps_left = 2;
                        st.active_weapon = 1;
                        st.deep_frozen = Some(false);
                        <ddai_tsworld::SimWorld as PlanWorld>::apply_tee_state(&mut world, 0, &st);
                        nav.respawned();
                        t += 1;
                        tick += 1;
                        continue;
                    }
                    if nav.done() {
                        break;
                    }
                    <ddai_tsworld::SimWorld as PlanWorld>::set_input(&mut world, 0, inp);
                    <ddai_tsworld::SimWorld as PlanWorld>::step(&mut world);
                    t += 1;
                    tick += 1;
                }
                let ok = !diverged
                    && nav.phase().name() == phase
                    && nav.outcome() == outcome
                    && t == ticks
                    && got_kills == kills
                    && got_notes == notes;
                if !ok {
                    nav_bad += 1;
                    if !diverged {
                        eprintln!(
                            "navtrace {nav_n} {from:?}->{to:?}: phase {} / {phase}, ticks {t} / {ticks}, kills {} / {}, notes {} / {}, outcome {:?} / {outcome:?}",
                            nav.phase().name(),
                            got_kills.len(),
                            kills.len(),
                            got_notes.len(),
                            notes.len(),
                            nav.outcome()
                        );
                        for (a, b) in got_notes.iter().zip(&notes) {
                            if a != b {
                                eprintln!("  note differs: {a:?} / {b:?}");
                                break;
                            }
                        }
                    }
                }
                nav_arrived += usize::from(phase == "arrived");
            }
            Line::Crosstrace {
                crossing,
                lag,
                use_wall,
                start,
                ended,
                reason,
                ticks,
                doings,
                trace,
            } => {
                cross_n += 1;
                let def = wayblock_for("Copy Love Box", Some(&ts.collision)).expect("the map has a WB");
                let mut world = ddai_tsworld::SimWorld::new(
                    ts.collision.clone(),
                    ddai_tsworld::world::SimWorldOptions {
                        respawn_delay_ticks: Some(0),
                        infinite_ammo: Some(true),
                        sv_hit: Some(true),
                        all_weapons: None,
                        no_weak_hook: None,
                    },
                );
                <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut world, 0, Vec2 { x: start.0, y: start.1 });
                let sim = <ddai_tsworld::SimWorld as PlanWorld>::new_scratch(&world);
                let mut crosser = SwingCrosser::new(sim, def.crossings[crossing].clone());
                crosser.use_wall = use_wall;
                let mut ok = true;
                let mut got_ticks = 0usize;
                let mut got_doings: Vec<(usize, String)> = Vec::new();
                for (t, want) in trace.iter().enumerate() {
                    let tick = 1000i64 + t as i64;
                    let me = <ddai_tsworld::SimWorld as PlanWorld>::get_tee(&world, 0).expect("tee");
                    let inp: PlayerInput = crosser.step(world.collision(), &me, tick, lag);
                    got_ticks = t + 1;
                    let doing = crosser.doing();
                    if got_doings.last().is_none_or(|(_, d)| *d != doing) {
                        got_doings.push((t, doing));
                    }
                    let got = (
                        inp.direction,
                        inp.target_x.to_bits(),
                        inp.target_y.to_bits(),
                        inp.jump,
                        inp.hook,
                    );
                    let wantt = (
                        want.0,
                        u64::from_str_radix(&want.1, 16).unwrap(),
                        u64::from_str_radix(&want.2, 16).unwrap(),
                        want.3,
                        want.4,
                    );
                    if got != wantt {
                        ok = false;
                        if cross_bad < 3 {
                            eprintln!(
                                "crosstrace {cross_n} (crossing {crossing}, lag {lag}) differs at tick {t}: got {got:?} want {wantt:?}"
                            );
                        }
                        break;
                    }
                    if crosser.done() {
                        break;
                    }
                    <ddai_tsworld::SimWorld as PlanWorld>::set_input(&mut world, 0, inp);
                    <ddai_tsworld::SimWorld as PlanWorld>::step(&mut world);
                }
                if ok && !doings.is_empty() && got_doings != doings {
                    ok = false;
                    eprintln!(
                        "crosstrace {cross_n}: the descriptions differ:\n  rust {got_doings:?}\n  ts   {doings:?}"
                    );
                }
                let phase = match crosser.phase() {
                    CrossPhase::Arrived => "arrived",
                    CrossPhase::Failed => "failed",
                    _ => "",
                };
                if ok && (phase != ended || got_ticks != ticks || (phase == "failed" && crosser.reason() != reason)) {
                    ok = false;
                    eprintln!(
                        "crosstrace {cross_n}: ended {phase:?} after {got_ticks}, TS {ended:?} after {ticks} ({reason:?} vs {:?})",
                        crosser.reason()
                    );
                }
                if !ok {
                    cross_bad += 1;
                }
                cross_arrived += usize::from(ended == "arrived");
            }
            Line::Route { from, to, opts, result } => {
                routes += 1;
                let avoid: HashSet<i32> = opts.avoid.iter().copied().collect();
                let ropts = RouteOpts {
                    near_tiles: opts.near_tiles,
                    partial: opts.partial,
                    allow_kill: opts.allow_kill,
                    through_freeze: opts.through_freeze,
                    max_nodes: opts.max_nodes,
                    avoid: Some(&avoid),
                    ..RouteOpts::default()
                };
                let got = router.find_route(from, to, &ropts);
                let same = match (&got, &result) {
                    (None, None) => true,
                    (Some(g), Some(w)) => {
                        g.cost == w.cost
                            && g.steps.len() == w.steps.len()
                            && g.steps.iter().zip(&w.steps).all(|(a, b)| {
                                a.x == b.x
                                    && a.y == b.y
                                    && kind_of(a.kind) == b.kind
                                    && a.anchor.unwrap_or((-1, -1)) == (b.ax, b.ay)
                                    && a.freeze == b.freeze
                                    && a.tele == b.tele
                                    && a.move_key == b.mv
                                    && a.leap == b.leap
                            })
                    }
                    _ => false,
                };
                if !same {
                    route_bad += 1;
                    if route_bad <= 5 {
                        eprintln!(
                            "route {routes} differs: from {from:?} to {to:?} opts near={} partial={} kill={} freeze={}",
                            opts.near_tiles, opts.partial, opts.allow_kill, opts.through_freeze
                        );
                    }
                }
                if let Some(w) = &result {
                    found += 1;
                    for s in &w.steps {
                        hooks += i32::from(s.kind == "hook");
                        freezes += i32::from(s.freeze);
                        kills += i32::from(s.kind == "kill");
                        teles += i32::from(s.tele);
                    }
                }
            }
        }
    }
    println!(
        "{path}: {routes} routes ({found} found; {hooks} hook, {freezes} freeze, {kills} kill, {teles} tele steps), {route_bad} mismatches; dead zone checked: {dead_checked}, spawns checked: {spawns_checked}; heap overflow drops: {}",
        router.heap_dropped()
    );
    if nav_n > 0 {
        println!("navigator: {nav_n} runs ({nav_arrived} arrived in TS), {nav_bad} mismatches");
    }
    if chooser_n + spot_n + wb_checked + cross_n > 0 {
        println!(
            "wayblock: {wb_checked} table checks, {chooser_n} chooser sequences ({chooser_bad} mismatches), {spot_n} wbSpot cases ({spot_bad} mismatches); crossings: {cross_n} traces ({cross_arrived} arrived in TS), {cross_bad} mismatches"
        );
    }
    if legacy_skipped > 0 {
        println!("legacy dump: {legacy_skipped} wbSpot/navigator/crossing lines not replayed (changed by af49dfb)");
    }
    assert_eq!(wb_def_bad, 0, "wayblockFor differs from TS");
    assert_eq!(route_bad, 0, "findRoute differs from TS");
    assert_eq!(chooser_bad, 0, "WbSideChooser differs from TS");
    assert_eq!(spot_bad, 0, "wbSpot differs from TS");
    assert_eq!(cross_bad, 0, "SwingCrosser differs from TS");
    assert_eq!(nav_bad, 0, "Navigator differs from TS");
}
