//! Task 3.8: parity of the pieces upstream af49dfb added to the planner, one by one, against the dumps of
//! `tools/ts-trace/gen-v2-component-dump.mjs` (made with `DDAI_TS_REF` = the af49dfb sources): `ceilingField` (the whole grid),
//! `ropeIntercept`, `wallSwingLines`/`airChainLines`, `sealedIn(..., passive)`. The full-planner dumps (`parity_planner.rs`)
//! exercise them through decisions; these say which piece is wrong when one is.
//!
//! `#[ignore]`d (needs the `ts-parity` feature and a locally generated dump):
//! ```text
//! DDAI_TS_REF=~/aiddnet/data/scratch/ts-af49dfb node tools/ts-trace/gen-v2-component-dump.mjs --map "<map>" --seed 1 --cases 1500 \
//!   --out ~/aiddnet/data/traces/planner-af49dfb/components/clb.jsonl
//! DDAI_V2_COMPONENT_DUMP=~/aiddnet/data/traces/planner-af49dfb/components/clb.jsonl \
//!   cargo test -p ddai-planner --features ts-parity --release --test parity_components_v2 -- --ignored --nocapture
//! ```

#![cfg(feature = "ts-parity")]

use ddai_planner::fields::{ceiling_field, rope_intercept};
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::planner::PlanStep;
use ddai_planner::seal::{sealed_in, sealed_in_passive};
use ddai_planner::throw_lines::{air_chain_lines, wall_swing_lines};
use ddai_planner::types::{PlayerInput, TeeState};
use ddai_planner::vmath::{Vec2, vec2};
use serde::Deserialize;

fn bits_to_f64(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex f64 bits"))
}

#[derive(Deserialize)]
struct StepJson {
    dir: i32,
    jump: i32,
    hook: i32,
    fire: i32,
    aim: String,
}

#[derive(Deserialize)]
struct InputJson {
    direction: i32,
    #[serde(rename = "targetX")]
    target_x: String,
    #[serde(rename = "targetY")]
    target_y: String,
    jump: i32,
    fire: i32,
    hook: i32,
    #[serde(rename = "playerFlags")]
    player_flags: i32,
    #[serde(rename = "wantedWeapon")]
    wanted_weapon: i32,
    #[serde(rename = "nextWeapon")]
    next_weapon: i32,
    #[serde(rename = "prevWeapon")]
    prev_weapon: i32,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
#[allow(clippy::large_enum_variant)]
enum Line {
    Meta {
        #[serde(rename = "mapPath")]
        map_path: String,
        #[serde(rename = "mapSha256")]
        map_sha256: String,
    },
    Ceiling {
        width: i32,
        height: i32,
        dist: Vec<u8>,
    },
    Intercept {
        #[serde(rename = "fromX")]
        from_x: String,
        #[serde(rename = "fromY")]
        from_y: String,
        #[serde(rename = "posX")]
        pos_x: String,
        #[serde(rename = "posY")]
        pos_y: String,
        #[serde(rename = "velX")]
        vel_x: String,
        #[serde(rename = "velY")]
        vel_y: String,
        x: String,
        y: String,
    },
    Wallswing {
        steps: i32,
        #[serde(rename = "stepTicks")]
        step_ticks: Vec<i32>,
        at: String,
        #[serde(rename = "wallDir")]
        wall_dir: i32,
        lines: Vec<Vec<StepJson>>,
    },
    Airchain {
        steps: i32,
        #[serde(rename = "stepTicks")]
        step_ticks: Vec<i32>,
        at: String,
        #[serde(rename = "wallDir")]
        wall_dir: i32,
        #[serde(rename = "airJump")]
        air_jump: bool,
        lines: Vec<Vec<StepJson>>,
    },
    Sealpassive {
        #[serde(rename = "posX")]
        pos_x: String,
        #[serde(rename = "posY")]
        pos_y: String,
        #[serde(rename = "velX")]
        vel_x: String,
        #[serde(rename = "velY")]
        vel_y: String,
        frozen: bool,
        #[serde(rename = "freezeTicksLeft")]
        freeze_ticks_left: i64,
        input: InputJson,
        plain: bool,
        passive: bool,
    },
}

fn lines_match(got: &[Vec<PlanStep>], want: &[Vec<StepJson>]) -> bool {
    got.len() == want.len()
        && got.iter().zip(want).all(|(g, w)| {
            g.len() == w.len()
                && g.iter().zip(w).all(|(gs, ws)| {
                    gs.dir == ws.dir
                        && gs.jump == ws.jump
                        && gs.hook == ws.hook
                        && gs.fire == ws.fire
                        && gs.aim.to_bits() == bits_to_f64(&ws.aim).to_bits()
                })
        })
}

#[test]
#[ignore = "needs the ts-parity feature and a locally-generated dump, see this file's doc comment"]
fn v2_components_match_ts() {
    let path = std::env::var("DDAI_V2_COMPONENT_DUMP")
        .expect("set DDAI_V2_COMPONENT_DUMP to a gen-v2-component-dump.mjs output");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Line::Meta { map_path, map_sha256 } =
        serde_json::from_str(lines.next().expect("empty dump")).expect("bad meta line")
    else {
        panic!("first line must be meta");
    };
    let bytes = std::fs::read(&map_path).unwrap_or_else(|e| panic!("reading map {map_path}: {e}"));
    {
        use sha2::{Digest, Sha256};
        let sha: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(sha, map_sha256, "map changed since the dump was generated");
    }
    let collision = ddai_tsworld::load_map_bytes(&bytes)
        .unwrap_or_else(|e| panic!("loading map: {e:?}"))
        .collision;
    let new_world = || {
        ddai_tsworld::SimWorld::new(
            collision.clone(),
            ddai_tsworld::world::SimWorldOptions {
                respawn_delay_ticks: Some(0),
                infinite_ammo: Some(true),
                sv_hit: Some(true),
                all_weapons: None,
                no_weak_hook: None,
            },
        )
    };

    let mut counts = [0usize; 5];
    let mut mismatches = [0usize; 5];
    let mut passive_true = 0usize;
    let mut passive_differs = 0usize;
    for line in lines {
        match serde_json::from_str::<Line>(line).unwrap_or_else(|e| panic!("bad line: {e}\n{line}")) {
            Line::Meta { .. } => panic!("unexpected second meta line"),
            Line::Ceiling { width, height, dist } => {
                counts[0] += 1;
                let got = ceiling_field(&collision);
                if got.width != width || got.height != height || got.dist != dist {
                    mismatches[0] += 1;
                    eprintln!("ceilingField mismatch");
                }
            }
            Line::Intercept {
                from_x,
                from_y,
                pos_x,
                pos_y,
                vel_x,
                vel_y,
                x,
                y,
            } => {
                counts[1] += 1;
                let got = rope_intercept(
                    &collision,
                    vec2(bits_to_f64(&from_x), bits_to_f64(&from_y)),
                    vec2(bits_to_f64(&pos_x), bits_to_f64(&pos_y)),
                    vec2(bits_to_f64(&vel_x), bits_to_f64(&vel_y)),
                );
                if got.x.to_bits() != bits_to_f64(&x).to_bits() || got.y.to_bits() != bits_to_f64(&y).to_bits() {
                    mismatches[1] += 1;
                    eprintln!(
                        "ropeIntercept mismatch: got {got:?} want ({}, {})",
                        bits_to_f64(&x),
                        bits_to_f64(&y)
                    );
                }
            }
            Line::Wallswing {
                steps,
                step_ticks,
                at,
                wall_dir,
                lines,
            } => {
                counts[2] += 1;
                if !lines_match(
                    &wall_swing_lines(steps, &step_ticks, bits_to_f64(&at), wall_dir),
                    &lines,
                ) {
                    mismatches[2] += 1;
                    eprintln!("wallSwingLines mismatch: steps={steps} ticks={step_ticks:?} wall_dir={wall_dir}");
                }
            }
            Line::Airchain {
                steps,
                step_ticks,
                at,
                wall_dir,
                air_jump,
                lines,
            } => {
                counts[3] += 1;
                if !lines_match(
                    &air_chain_lines(steps, &step_ticks, bits_to_f64(&at), wall_dir, air_jump),
                    &lines,
                ) {
                    mismatches[3] += 1;
                    eprintln!(
                        "airChainLines mismatch: steps={steps} ticks={step_ticks:?} wall_dir={wall_dir} air_jump={air_jump}"
                    );
                }
            }
            Line::Sealpassive {
                pos_x,
                pos_y,
                vel_x,
                vel_y,
                frozen,
                freeze_ticks_left,
                input,
                plain,
                passive,
            } => {
                counts[4] += 1;
                let pos = Vec2 {
                    x: bits_to_f64(&pos_x),
                    y: bits_to_f64(&pos_y),
                };
                let vel = Vec2 {
                    x: bits_to_f64(&vel_x),
                    y: bits_to_f64(&vel_y),
                };
                let held = PlayerInput {
                    direction: input.direction,
                    target_x: bits_to_f64(&input.target_x),
                    target_y: bits_to_f64(&input.target_y),
                    jump: input.jump,
                    fire: input.fire,
                    hook: input.hook,
                    player_flags: input.player_flags,
                    wanted_weapon: input.wanted_weapon,
                    next_weapon: input.next_weapon,
                    prev_weapon: input.prev_weapon,
                };
                let run = |passive_mode: bool| {
                    let mut w = new_world();
                    <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut w, 0, pos);
                    <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut w, 1, collision_anywhere());
                    let base = <ddai_tsworld::SimWorld as PlanWorld>::get_tee(&w, 0).unwrap();
                    let state = TeeState {
                        pos,
                        vel,
                        frozen,
                        freeze_ticks_left,
                        ..base
                    };
                    if passive_mode {
                        sealed_in_passive(&mut w, 0, &state, held)
                    } else {
                        sealed_in(&mut w, 0, &state, held)
                    }
                };
                let (got_plain, got_passive) = (run(false), run(true));
                passive_true += usize::from(passive);
                passive_differs += usize::from(passive != plain);
                if got_plain != plain || got_passive != passive {
                    mismatches[4] += 1;
                    eprintln!(
                        "sealedIn mismatch at {pos:?} frozen={frozen}/{freeze_ticks_left}: plain got={got_plain} want={plain}, passive got={got_passive} want={passive}"
                    );
                }
            }
        }
    }
    let names = [
        "ceilingField",
        "ropeIntercept",
        "wallSwingLines",
        "airChainLines",
        "sealedIn(passive)",
    ];
    for i in 0..5 {
        eprintln!("{}: {} cases, {} mismatches", names[i], counts[i], mismatches[i]);
    }
    eprintln!("sealedIn passive: {passive_true} sealed, {passive_differs} differ from the plain answer");
    assert!(counts.iter().all(|&n| n > 0), "the dump lacks a component: {counts:?}");
    assert!(mismatches.iter().all(|&n| n == 0), "mismatches: {mismatches:?}");
    assert!(
        passive_true > 0 && passive_differs > 0,
        "the passive cases are not mixed"
    );
}

/// The second tee of a `sealed_in` case is stripped by `sealed_in` itself; any position will do.
fn collision_anywhere() -> Vec2 {
    vec2(48.0, 48.0)
}
