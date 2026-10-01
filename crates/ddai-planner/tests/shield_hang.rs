//! Task 3.5b, review F1: the shield's "hanging from our own tile hook is settled" rule (`hang_ok`) must not call a
//! tee that is still swinging into freeze safe. Grid of 360 start states on four maps (a hookable wall, a freeze
//! patch on the swing path, a freeze floor; the tee jumpless and moving away fast, the input under test keeps
//! drifting right): whenever `escape_exists_ext` with the anchor extras says "an escape exists", at least one
//! of those escapes must really survive (36 ticks of the escape, then coasting on the hook for 300 ticks).
//! The first version of the rule (a grabbed tile hook alone counts) failed 9 of the states.

use std::collections::HashMap;
use std::sync::Arc;

use ddai_physics::map::{MapData, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID, Tile};
use ddai_planner::hybrid::anchors::{AnchorCache, hook_escapes};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::shield::{Bounded, ShieldBuffers, ShieldOpts, escape_exists_bounded, escape_exists_ext};
use ddai_planner::types::{HOOK_GRABBED, PlayerInput, empty_input};
use ddai_planner::vmath::Vec2;

fn mk(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Arc<MapData> {
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            game[y * w + x] = Tile {
                index: f(x, y),
                ..Tile::default()
            };
        }
    }
    Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

/// Left wall hookable, a freeze patch on the swing path (`x3..=px1`, `py0..=py1`), a freeze floor far below;
/// ceiling and right wall unhookable.
fn swing_patch(py0: usize, py1: usize, px1: usize) -> Arc<MapData> {
    mk(60, 40, move |x, y| {
        if x <= 2 && (1..=38).contains(&y) {
            TILE_SOLID
        } else if y == 0 || x == 59 || y == 39 || x == 0 {
            TILE_NOHOOK
        } else if ((3..=px1).contains(&x) && (py0..=py1).contains(&y)) || (36..=38).contains(&y) {
            TILE_FREEZE
        } else {
            0
        }
    })
}

fn big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn a_hang_that_still_swings_into_freeze_is_not_called_settled() {
    big_stack(|| {
        let mut states = 0;
        let mut false_safe = Vec::new();
        let mut hang_accepts = 0; // verdicts the rule turned from "no escape" into a surviving "escape"
        for (py0, py1, px1) in [(16usize, 18usize, 6usize), (18, 20, 8), (20, 22, 10), (15, 17, 4)] {
            let map = swing_patch(py0, py1, px1);
            for xt in [8.0, 9.0, 10.0, 11.0, 12.0] {
                for yt in [8.0, 10.0, 12.0] {
                    for vx in [10.0, 15.0, 20.0, 25.0, 30.0, 35.0] {
                        let mut pw = PhysicsWorld::new(map.clone(), 1);
                        pw.add_tee(
                            0,
                            Vec2 {
                                x: xt * 32.0 + 16.0,
                                y: yt * 32.0 + 16.0,
                            },
                        );
                        let mut st = pw.get_tee(0).unwrap();
                        st.vel = Vec2 { x: vx, y: 0.0 };
                        st.jumped = 3;
                        st.jumped_total = Some(2);
                        st.jumps_left = 0;
                        pw.apply_tee_state(0, &st);
                        let me = pw.get_tee(0).unwrap();
                        let mut input = empty_input();
                        input.direction = 1;
                        input.target_x = 300.0;
                        input.target_y = 0.0;
                        let others: HashMap<i32, PlayerInput> = HashMap::new();
                        let mut cache = AnchorCache::new();
                        let anchors = cache.select(pw.collision(), me.pos, 8);
                        let extras = hook_escapes(&anchors, me.pos, 3);
                        let old = escape_exists_bounded(&mut pw, 0, &input, 2, &others, None);
                        let mut bufs = ShieldBuffers::default();
                        let new = escape_exists_ext(
                            &mut pw,
                            0,
                            &input,
                            2,
                            &others,
                            None,
                            &mut bufs,
                            &mut ShieldOpts {
                                first: None,
                                extras: &extras,
                            },
                        );
                        // Ground truth: does any extra escape survive 2 + 36 + 300 ticks, holding the hook?
                        let mut any_survives = false;
                        for esc in &extras {
                            let saved = pw.save_state();
                            for _ in 0..2 {
                                pw.set_input(0, input);
                                pw.step();
                            }
                            let mut froze = false;
                            for t in 0..336 {
                                let mut i = *esc;
                                if t > 0 {
                                    i.jump = 0;
                                }
                                if t >= 36 {
                                    i.direction = 0;
                                }
                                pw.set_input(0, i);
                                pw.step();
                                let m = pw.get_tee(0).unwrap();
                                froze |= m.frozen || !m.alive;
                            }
                            pw.restore_state(&saved);
                            any_survives |= !froze;
                        }
                        states += 1;
                        if new == Bounded::Done(true) && !any_survives {
                            false_safe.push((py0, xt, yt, vx));
                        }
                        if old == Bounded::Done(false) && new == Bounded::Done(true) && any_survives {
                            hang_accepts += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(states, 360);
        assert!(
            false_safe.is_empty(),
            "the shield called {} doomed hang(s) safe: {false_safe:?}",
            false_safe.len()
        );
        // Not vacuous: settled hangs are still accepted (the reason for the rule: T14-like states).
        assert!(hang_accepts > 50, "only {hang_accepts} settled hangs were accepted");
    });
}

#[test]
fn a_tee_hanging_at_rest_on_a_tile_hook_is_still_called_settled() {
    big_stack(|| {
        // No patch at all: the tee drifts away from the wall, hooks it and hangs; nothing to swing into.
        let map = swing_patch(30, 30, 3);
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        pw.add_tee(
            0,
            Vec2 {
                x: 9.0 * 32.0 + 16.0,
                y: 8.0 * 32.0 + 16.0,
            },
        );
        let mut st = pw.get_tee(0).unwrap();
        st.vel = Vec2 { x: 10.0, y: 0.0 };
        st.jumped = 3;
        st.jumped_total = Some(2);
        st.jumps_left = 0;
        pw.apply_tee_state(0, &st);
        let me = pw.get_tee(0).unwrap();
        let mut input = empty_input();
        input.direction = 1;
        input.target_x = 300.0;
        let mut cache = AnchorCache::new();
        let anchors = cache.select(pw.collision(), me.pos, 8);
        let extras = hook_escapes(&anchors, me.pos, 3);
        let mut bufs = ShieldBuffers::default();
        let v = escape_exists_ext(
            &mut pw,
            0,
            &input,
            2,
            &HashMap::new(),
            None,
            &mut bufs,
            &mut ShieldOpts {
                first: None,
                extras: &extras,
            },
        );
        assert_eq!(v, Bounded::Done(true));
        let _ = HOOK_GRABBED;
    });
}

// ---- review round 2, F1: free-fall apexes ---------------------------------------------------------------------
// 300 random maps (hookable walls, ceilings and platforms, freeze right under them), 2,979 start states with
// a jumpless tee: within ~42 px of the anchor the hook does not pull, so a tee rising through that zone falls
// freely and is "calm" (|v| <= 1.5) for up to 7 ticks at the top of every arc. With 6 calm ticks required, 2 of
// these states were declared safe (accepted at tick 43 / 45, frozen at 54 / 61). The run must be longer than the
// apex: `ceil(2 * 1.5 / gravity) + 2` = 8.

struct Rnd(u64);
impl Rnd {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn i(&mut self, a: i64, b: i64) -> i64 {
        a + (self.f() * (b - a + 1) as f64) as i64
    }
}

fn esc(dir: i32, jump: i32, ax: f64, ay: f64, hook: i32) -> PlayerInput {
    let mut e = empty_input();
    e.direction = dir;
    e.jump = jump;
    e.target_x = ax;
    e.target_y = ay;
    e.hook = hook;
    e
}

/// The shield's own walk/jump/hook escapes for a tee moving at `vx` (the TS `escapes()` list).
fn std_escapes(vx: f64, ax: f64, ay: f64) -> Vec<PlayerInput> {
    let b = if vx.abs() < 0.5 {
        0
    } else if vx > 0.0 {
        -1
    } else {
        1
    };
    let mut out = if b == 0 {
        vec![esc(0, 0, ax, ay, 0), esc(0, 1, ax, ay, 0)]
    } else {
        vec![
            esc(0, 0, ax, ay, 0),
            esc(0, 1, ax, ay, 0),
            esc(b, 0, ax, ay, 0),
            esc(b, 1, ax, ay, 0),
        ]
    };
    for a in [0, b] {
        out.push(esc(b, 1, f64::from(a) * 150.0, -300.0, 1));
        if b == 0 {
            break;
        }
    }
    out
}

#[test]
fn a_free_fall_apex_near_the_anchor_is_not_a_settled_hang() {
    big_stack(|| {
        let (w, h) = (40usize, 30usize);
        let mut rng = Rnd(0x9e3779b97f4a7c15);
        let (mut trials, mut accepted, mut hang_accepted, mut false_safe) = (0, 0, 0, Vec::new());
        for m in 0..300 {
            let mut cells = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let c = &mut cells[y * w + x];
                    if x <= 1 || y == 0 {
                        *c = TILE_SOLID;
                    }
                    if x == w - 1 || y == h - 1 {
                        *c = TILE_NOHOOK;
                    }
                    if y >= h - 3 && y < h - 1 && x > 1 && x < w - 1 {
                        *c = TILE_FREEZE;
                    }
                }
            }
            for _ in 0..rng.i(1, 3) {
                let (px, py, pw) = (rng.i(3, 30) as usize, rng.i(3, 20) as usize, rng.i(1, 6) as usize);
                for x in px..(px + pw).min(w - 2) {
                    cells[py * w + x] = TILE_SOLID;
                }
                let fy = py + 1 + rng.i(0, 4) as usize;
                if fy < h - 1 {
                    for x in px.saturating_sub(1)..(px + pw + 1).min(w - 2) {
                        if cells[fy * w + x] == 0 {
                            cells[fy * w + x] = TILE_FREEZE;
                        }
                    }
                }
            }
            let (bx, by) = (rng.i(3, 34) as usize, rng.i(2, 24) as usize);
            for y in by..(by + rng.i(1, 3) as usize).min(h - 1) {
                for x in bx..(bx + rng.i(1, 4) as usize).min(w - 1) {
                    if cells[y * w + x] == 0 {
                        cells[y * w + x] = TILE_FREEZE;
                    }
                }
            }
            let map = Arc::new(MapData {
                width: w as u32,
                height: h as u32,
                game: cells
                    .iter()
                    .map(|&c| Tile {
                        index: c,
                        ..Tile::default()
                    })
                    .collect(),
                front: None,
                tele: None,
                speedup: None,
                switch: None,
                tune: None,
                settings: Vec::new(),
            });
            for _ in 0..12 {
                let (tx, ty) = (rng.i(2, 37) as usize, rng.i(1, 26) as usize);
                if cells[ty * w + tx] != 0 {
                    continue;
                }
                let mut pw = PhysicsWorld::new(map.clone(), 1);
                pw.add_tee(
                    0,
                    Vec2 {
                        x: tx as f64 * 32.0 + 16.0,
                        y: ty as f64 * 32.0 + 16.0,
                    },
                );
                let mut st = pw.get_tee(0).unwrap();
                st.vel = Vec2 {
                    x: rng.f() * 24.0 - 12.0,
                    y: rng.f() * 20.0 - 10.0,
                };
                if rng.f() < 0.7 {
                    st.jumped = 3;
                    st.jumped_total = Some(2);
                    st.jumps_left = 0;
                }
                pw.apply_tee_state(0, &st);
                let me = pw.get_tee(0).unwrap();
                if me.frozen {
                    continue;
                }
                let mut input = empty_input();
                input.direction = rng.i(-1, 1) as i32;
                #[allow(clippy::approx_constant)] // the reviewer's exact random stream
                let a = rng.f() * 6.283;
                input.target_x = (a.cos() * 300.0).round();
                input.target_y = (a.sin() * 300.0).round();
                let others: HashMap<i32, PlayerInput> = HashMap::new();
                let mut cache = AnchorCache::new();
                let anchors = cache.select(pw.collision(), me.pos, 8);
                let extras = hook_escapes(&anchors, me.pos, 3);
                if extras.is_empty() {
                    continue;
                }
                let old = escape_exists_bounded(&mut pw, 0, &input, 2, &others, None);
                let mut bufs = ShieldBuffers::default();
                let new = escape_exists_ext(
                    &mut pw,
                    0,
                    &input,
                    2,
                    &others,
                    None,
                    &mut bufs,
                    &mut ShieldOpts {
                        first: None,
                        extras: &extras,
                    },
                );
                trials += 1;
                if new != Bounded::Done(true) {
                    continue;
                }
                accepted += 1;
                if old == Bounded::Done(true) {
                    continue;
                }
                hang_accepted += 1;
                // Ground truth over the standard escapes and the extras, over the shield's own horizon:
                // hold 2 ticks, escape 36, coast the settle ticks.
                let saved = pw.save_state();
                for _ in 0..2 {
                    pw.set_input(0, input);
                    pw.step();
                }
                let after = pw.get_tee(0).unwrap();
                let mut all = std_escapes(after.vel.x, input.target_x, input.target_y);
                all.extend(extras.iter().copied());
                let held = pw.save_state();
                let mut survives = false;
                if after.alive && !after.frozen {
                    for e in &all {
                        pw.restore_state(&held);
                        let mut ok = true;
                        for t in 0..126 {
                            let i = if t < 36 {
                                if t == 0 || e.jump == 0 {
                                    *e
                                } else {
                                    PlayerInput { jump: 0, ..*e }
                                }
                            } else {
                                PlayerInput {
                                    hook: e.hook,
                                    target_x: e.target_x,
                                    target_y: e.target_y,
                                    ..empty_input()
                                }
                            };
                            pw.set_input(0, i);
                            pw.step();
                            let m = pw.get_tee(0).unwrap();
                            if m.frozen || !m.alive {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            survives = true;
                            break;
                        }
                    }
                }
                pw.restore_state(&saved);
                if !survives {
                    false_safe.push((m, tx, ty));
                }
            }
        }
        assert_eq!(trials, 2979);
        assert!(
            false_safe.is_empty(),
            "hang rule declared doomed states safe (map, tx, ty): {false_safe:?}"
        );
        // Not vacuous: the rule still accepts settled hangs (2,413 states say "an escape exists"; 3.5 alone says so in 1,723).
        assert!(
            accepted > 2300 && hang_accepted > 600,
            "accepted {accepted}, of them by the hang rule {hang_accepted}"
        );
    });
}
