//! Task 3.17 (D-111): the learned window model inside the whole bot pipeline, offline.
//!
//! Synthetic snapshots through [`ddai_bot::Bot`] with a probing brain (which records the exact world it is handed). The model is a constant
//! network (a bundle written to a temp file: "walks right" or "walks left" at every tick) so the expected effect is known: the target's
//! predicted state in the brain's world must move with the model's inputs, and only then.
//!
//! * **default identity**: no model, a model whose kill marker exists, and a model that was never asked for decide the same snapshot
//!   sequence the same way (outputs, the worlds the brain sees);
//! * **the model's effect**: with the model on the target is predicted walking where the model says;
//! * **the guard**: a model that loses to hold is benched (STATUS says so), hold's world comes back, the model keeps being scored;
//! * **the marker** switches it off and on at run time, outside the decision path;
//! * **the log** is written, parses, and has no name in it;
//! * **allocation**: steady-state snapshots with the model on allocate nothing in the bot's own code.

mod support;

use std::path::Path;
use std::time::{Duration, Instant};

use ddai_bot::oppnet::{WINDOW_MODEL_OFF_MARKER, WindowModelConfig, WindowModelRt};
use ddai_bot::{Bot, BrainKind, Relations};
use ddai_brain::Action;
use ddai_oppnet::bundle::OppBundle;
use ddai_oppnet::feature::{INPUT_DIM, OUT_DIM};
use ddai_oppnet::live::analyze::Report;
use ddai_oppnet::live::guard::GuardConfig;
use ddai_oppnet::net::Mlp;
use support::*;

/// Writes a constant model to `dir`: tick-wise direction class `dir_class` (0 left, 1 none, 2 right).
fn write_model(dir: &Path, name: &str, dir_class: usize) -> std::path::PathBuf {
    let mut net = Mlp::new(INPUT_DIM, 8, 8, OUT_DIM, 1);
    net.params.iter_mut().for_each(|p| *p = 0.0);
    let n = net.params.len();
    for k in 0..8 {
        net.params[n - OUT_DIM + k * 7 + dir_class] = 5.0;
    }
    let path = dir.join(name);
    OppBundle::new(net, 1, 1, 0.0, "constant".into()).save(&path).unwrap();
    path
}

/// The scenarios move the opponent far from us, past the regime gate's reach; the gate has its own test below.
fn ungated(mut c: WindowModelConfig) -> WindowModelConfig {
    c.gate = ddai_oppnet::live::RegimeGate::off();
    c
}

fn quick_guard() -> GuardConfig {
    GuardConfig {
        windows: 24,
        min_windows: 12,
        margin: 0.05,
        retry_after: 30,
        retry_margin: 0.0,
    }
}

struct Rig {
    bot: Bot,
    sc: Scenario,
    log: std::rc::Rc<std::cell::RefCell<Vec<Seen>>>,
}

/// A bot on the plain room against one opponent; `model` loads a window model (the bot's brain kind is the hybrid's, the brain itself the probe).
fn rig(model: Option<&WindowModelConfig>) -> Rig {
    let (probe, log, _resets, _action) = Probe::new(Action::neutral());
    let mut bot = bot_with(Box::new(probe), cfg(BrainKind::Hybrid), Relations::new());
    let map = room(&[]);
    bot.on_map_loaded(std::sync::Arc::clone(&map));
    if let Some(m) = model {
        bot.set_window_model(Some(WindowModelRt::load(m, Instant::now()).expect("the model loads")));
    }
    let mut a = tee(0, 1000);
    let mut b = tee(1, 1200);
    a.angle = 100;
    b.angle = 200;
    Rig {
        bot,
        sc: Scenario::new(map, vec![a, b]),
        log,
    }
}

/// One snapshot: the opponent's aim changes (so it is not AFK) and it walks `dir` (1 right, -1 left, 0 stands).
fn step(r: &mut Rig, dir: i32) {
    {
        let t = r.sc.tee_mut(1);
        t.angle = (t.angle + 37) % 1000;
        t.direction = dir;
        t.x += dir * 18;
    }
    run(&mut r.bot, &mut r.sc, 1);
}

fn target_motion(r: &Rig) -> Vec<(f32, f32)> {
    r.log.borrow().iter().filter_map(|s| s.target_vel).collect()
}

#[test]
fn off_is_the_default_and_a_killed_model_decides_exactly_like_no_model() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let model = write_model(dir.path(), "m.oppnet", 2);
        // The marker exists from the start: the model is loaded but never called.
        let mut killed_cfg = ungated(WindowModelConfig::in_data_dir(model.clone(), dir.path()));
        std::fs::create_dir_all(dir.path().join("bot")).unwrap();
        std::fs::write(dir.path().join("bot").join(WINDOW_MODEL_OFF_MARKER), "").unwrap();
        killed_cfg.log = None;
        let mut plain = rig(None);
        let mut killed = rig(Some(&killed_cfg));
        for i in 0..50 {
            let d = if i % 20 < 10 { 1 } else { -1 };
            step(&mut plain, d);
            step(&mut killed, d);
        }
        assert_eq!(plain.bot.window_model_status().0, "off");
        assert_eq!(killed.bot.window_model_status().0, "killed");
        let (a, b) = (plain.log.borrow(), killed.log.borrow());
        assert!(a.len() > 30, "the brain decided: {}", a.len());
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                (x.world_tick, x.self_x, x.target_pos, x.target_vel),
                (y.world_tick, y.self_x, y.target_pos, y.target_vel),
                "the brain's world at tick {}",
                x.obs_tick
            );
        }
        // Nothing was predicted, nothing logged.
        let g = killed.bot.window_model_status().1.expect("a model is loaded");
        assert_eq!((g.predicted, g.used, g.resolved), (0, 0, 0));
    });
}

#[test]
fn the_model_puts_its_inputs_into_the_targets_predicted_motion() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let mut off = rig(None);
        // "Right": the target stands in every snapshot, but the brain's world has it walking right through the lag window.
        let right = {
            let mut c = ungated(WindowModelConfig::new(write_model(dir.path(), "right.oppnet", 2)));
            // A standing opponent is a case where "walks right" loses to hold, so the guard would bench it; this test is about the effect of the
            // inputs, so the guard is set to never judge.
            c.guard = GuardConfig {
                margin: 1e6,
                ..quick_guard()
            };
            c
        };
        let mut on = rig(Some(&right));
        for _ in 0..24 {
            step(&mut off, 0);
            step(&mut on, 0);
        }
        let (v_off, v_on) = (target_motion(&off), target_motion(&on));
        assert!(
            v_off.iter().all(|v| v.0 == 0.0),
            "hold: a standing target stays standing"
        );
        let walking = v_on.iter().filter(|v| v.0 > 1.0).count();
        assert!(
            walking >= v_on.len() - 3,
            "{walking} of {} decisions saw it walk right",
            v_on.len()
        );
        // Everything else the brain sees is the same: our own tee, the ticks.
        let (a, b) = (off.log.borrow(), on.log.borrow());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!((x.world_tick, x.self_x), (y.world_tick, y.self_x));
        }
        let (word, g) = on.bot.window_model_status();
        assert_eq!(word, "on");
        let g = g.unwrap();
        assert!(g.predicted >= 20 && g.used == g.predicted, "{g:?}");
        assert_eq!(g.sha256.len(), 64);
    });
}

#[test]
fn a_model_that_loses_to_hold_is_benched_and_hold_comes_back() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        // The opponent walks right all the time; the model says "left".
        let mut c = ungated(WindowModelConfig::new(write_model(dir.path(), "left.oppnet", 0)));
        c.guard = quick_guard();
        let mut r = rig(Some(&c));
        let mut states = Vec::new();
        for _ in 0..60 {
            step(&mut r, 1);
            states.push(r.bot.window_model_status().0);
        }
        assert_eq!(states[3], "on", "it drives until the guard has seen enough");
        assert_eq!(*states.last().unwrap(), "hold", "{states:?}");
        let g = r.bot.window_model_status().1.unwrap();
        assert_eq!(g.driver, "hold");
        assert_eq!(g.fallbacks, 1);
        assert!(g.model_cost > g.hold_cost, "{g:?}");
        assert!(g.used < g.predicted, "the shadow keeps predicting: {g:?}");
        // While the model drove, the target was predicted walking left; after the verdict it walks as its snapshot shows (right).
        let v = target_motion(&r);
        assert!(v[3].0 < 0.0, "driven by the model: {:?}", v[3]);
        assert!(v.last().unwrap().0 > 0.0, "benched: hold again {:?}", v.last());
    });
}

#[test]
fn the_marker_switches_the_model_off_and_on_once_a_second() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let cfg = ungated(WindowModelConfig::in_data_dir(
            write_model(dir.path(), "m.oppnet", 2),
            dir.path(),
        ));
        let marker = dir.path().join("bot").join(WINDOW_MODEL_OFF_MARKER);
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        let mut r = rig(Some(&cfg));
        let t0 = Instant::now();
        for _ in 0..6 {
            step(&mut r, 0);
        }
        assert_eq!(r.bot.window_model_status().0, "on");
        let before = r.bot.window_model_status().1.unwrap().predicted;
        assert!(before > 0);
        // The marker appears; the bot notices it at the next look (a second after the last), not before.
        std::fs::write(&marker, "").unwrap();
        r.bot.window_model_poll(t0 + Duration::from_millis(100));
        assert_eq!(r.bot.window_model_status().0, "on", "not looked at yet");
        r.bot.window_model_poll(t0 + Duration::from_millis(1500));
        assert_eq!(r.bot.window_model_status().0, "killed");
        for _ in 0..6 {
            step(&mut r, 0);
        }
        assert_eq!(
            r.bot.window_model_status().1.unwrap().predicted,
            before,
            "a killed model is not called"
        );
        let v = target_motion(&r);
        assert!(v.last().unwrap().0 == 0.0, "hold again: {:?}", v.last());
        // Gone again: back on.
        std::fs::remove_file(&marker).unwrap();
        r.bot.window_model_poll(t0 + Duration::from_millis(3000));
        assert_eq!(r.bot.window_model_status().0, "on");
        for _ in 0..6 {
            step(&mut r, 0);
        }
        assert!(r.bot.window_model_status().1.unwrap().predicted > before);
        assert!(target_motion(&r).last().unwrap().0 > 1.0, "the model drives again");
    });
}

#[test]
fn the_log_is_written_bounded_and_without_names() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = ungated(WindowModelConfig::in_data_dir(
            write_model(dir.path(), "m.oppnet", 2),
            dir.path(),
        ));
        cfg.guard = quick_guard();
        let log_path = cfg.log.clone().unwrap();
        let mut r = rig(Some(&cfg));
        // A player whose name must never reach the log.
        r.sc.player_mut(1).name = "SecretNickname".to_string();
        let t0 = Instant::now();
        for i in 0..40 {
            step(&mut r, if i < 20 { 1 } else { 0 });
        }
        r.bot.window_model_poll(t0 + Duration::from_secs(2));
        drop(r); // joins the writer thread
        let text = std::fs::read_to_string(&log_path).expect("the log exists");
        assert!(
            !text.contains("SecretNickname") && !text.to_lowercase().contains("secretnick"),
            "{text}"
        );
        let mut lines = text.lines();
        let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(header["ev"], "open");
        assert_eq!(
            header["model"].as_str().unwrap().len(),
            64,
            "the sha256 of the model file"
        );
        let mut report = Report::new();
        for l in text.lines() {
            report.add_line(l);
        }
        assert_eq!(report.bad_lines, 0);
        assert!(report.samples() > 20, "{}", report.samples());
        // The scenario's snapshots are two ticks apart and the driver's lead is 3: windows of 4 ticks (to_tick = pred_tick), odd steps seen.
        let lags: Vec<u8> = (0..9).filter(|&w| report.pooled(Some(w), None).n > 0).collect();
        assert!(
            !lags.is_empty() && lags.iter().all(|&w| (1..=8).contains(&w)),
            "{lags:?}"
        );
        let tag = text.lines().nth(1).unwrap();
        assert!(
            tag.contains(r#""o":"c1-"#),
            "the opponent is a tag, c<id>-<hash>: {tag}"
        );
        // A file that is not writable is a log that stays empty, not a crash: covered by the writer's own tests.
    });
}

#[test]
fn a_steady_state_snapshot_with_the_model_on_allocates_nothing_in_the_bots_own_code() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = ungated(WindowModelConfig::new(write_model(dir.path(), "m.oppnet", 2)));
        cfg.guard = quick_guard();
        // The idle brain keeps the planner's allocating helpers out of the picture (see the allocation test of `scenarios.rs`); the kind is
        // the hybrid's so that the window model acts.
        let mut bot = bot_with(Box::new(ddai_brain::IdleBrain), cfg_kind_hybrid(), Relations::new());
        let map = room(&[]);
        bot.on_map_loaded(map.clone());
        bot.set_window_model(Some(WindowModelRt::load(&cfg, Instant::now()).unwrap()));
        let mut tees = vec![tee(0, 1000)];
        for i in 1..=6 {
            tees.push(tee(i, 1000 + 120 * i));
        }
        let mut sc = Scenario::new(map, tees);
        for _ in 0..80 {
            for i in 1..=6 {
                let t = sc.tee_mut(i);
                t.angle = (t.angle + 37) % 1000;
            }
            run(&mut bot, &mut sc, 1);
        }
        assert_eq!(bot.target_id(), 1);
        let mut snaps = Vec::new();
        for _ in 0..200 {
            for i in 1..=6 {
                let t = sc.tee_mut(i);
                t.angle = (t.angle + 37) % 1000;
            }
            snaps.push(sc.snapshot());
            sc.tick += 2;
        }
        let info = allocation_counter::measure(|| {
            for snap in &snaps {
                let out = bot.on_snapshot(snap);
                std::hint::black_box(out);
                bot.on_input_sent(snap.pred_tick + 1, &out.input.expect("decided"));
            }
        });
        assert_eq!(
            info.count_total, 0,
            "allocations per snapshot in steady state: {info:?}"
        );
        let g = bot.window_model_status().1.unwrap();
        assert!(
            g.predicted >= 200 && g.resolved > 100,
            "the model really ran inside the measured loop: {g:?}"
        );
    });
}

fn cfg_kind_hybrid() -> ddai_bot::BotConfig {
    cfg(BrainKind::Hybrid)
}

#[test]
fn outside_the_regime_the_model_is_not_called_and_hold_decides() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        // The default gate: a duel within 480 px and nobody else near. A third tee 300 px away is a crowd.
        let mut cfg = WindowModelConfig::new(write_model(dir.path(), "right.oppnet", 2));
        cfg.guard = GuardConfig {
            margin: 1e6,
            ..quick_guard()
        };
        let mut duel = rig(Some(&cfg));
        let mut crowd = rig(Some(&cfg));
        crowd.sc.tees.push(tee(2, 1100));
        crowd.sc.players.push(player(2, "p2"));
        for _ in 0..20 {
            step(&mut duel, 0);
            step(&mut crowd, 0);
        }
        let g = duel.bot.window_model_status().1.unwrap();
        assert!(g.predicted >= 15 && g.skipped_regime == 0, "{g:?}");
        let g = crowd.bot.window_model_status().1.unwrap();
        assert_eq!(g.predicted, 0, "{g:?}");
        assert!(g.skipped_regime >= 15, "{g:?}");
        assert!(
            target_motion(&crowd).iter().all(|v| v.0 == 0.0),
            "hold decides in a crowd"
        );
    });
}
