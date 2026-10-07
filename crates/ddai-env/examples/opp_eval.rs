//! Task 3.15 (E-028): the physics side of the offline evaluation of an opponent-input model.
//!
//! Plays the games of an arena config (our hybrid **without** the model against the competitor's planner, as in `opp_record`) and, at every decision tick
//! `T`, rolls the exact physics through the next ticks with our own actual inputs and four assumptions about the opponent's:
//!
//! * `snap`: what the hybrid assumes today, "it keeps what the snapshot shows" (`enemy_input_from_tee`);
//! * `true`: it repeats its true previous input (jump and hook levels included);
//! * `model`: the network's prediction (`--model`), turned into inputs the way the brain does (`input_from_prediction`);
//! * `oracle`: its true inputs (the error must be exactly 0: this checks the harness).
//!
//! and compares the opponent's position after `k` ticks with where it really was. Reported per condition: the share of windows with an error below 1 px
//! and the median, p90 and mean error (px), for `k = 1 .. 8`, with the window end (`k` = our lag) marked.
//!
//! Also, over the window of our lag: **event agreement** with the oracle (a hammer hit of the opponent on us inside the window; the opponent's hook holding us at
//! its end), as a confusion table per method; and with `--decisions` **decision agreement**: three shadow hybrids (the window rolled by hold, by the model, by the
//! oracle: a `WindowModel` that returns the opponent's true inputs) are shown the same views, in order, and what they decide is compared (they do not play: the game
//! is the recorded one).
//!
//! ```text
//! cargo run --release -p ddai-env --example opp_eval -- --config configs/arena/e028-data-test.toml --model ~/aiddnet/data/runs/E-028/m1.oppnet --threads 3
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::{Brain, ResetContext, WorldView};
use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain, hybrid_config};
use ddai_env::game::play_game_watched;
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::{PlayerSetup, Sim};
use ddai_oppnet::OppPredictor;
use ddai_physics::core::PlayerInput as Wire;
use ddai_physics::world::World;
use ddai_planner::brains::ClockKind;
use ddai_planner::brains::enemy_input_from_tee;
use ddai_planner::hybrid::window::{PredictedInput, WindowCtx, WindowModel, input_from_prediction};
use ddai_planner::hybrid::{HybridBrain, NoProposer};
use ddai_planner::physics_adapter::{PhysicsWorld, from_ddnet_input};
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::{PlayerInput, WorldEvent};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

const K: usize = 8;
const METHODS: [&str; 4] = ["snap", "true", "model", "oracle"];

struct Args {
    config: PathBuf,
    model: Option<PathBuf>,
    gate: f32,
    threads: usize,
    games: Option<u32>,
    only: Option<String>,
    decisions: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        model: None,
        gate: 0.0,
        threads: 3,
        games: None,
        only: None,
        decisions: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--config" => a.config = PathBuf::from(v()?),
            "--model" => a.model = Some(PathBuf::from(v()?)),
            "--decisions" => a.decisions = true,
            "--gate" => a.gate = v()?.parse().map_err(|e| format!("--gate: {e}"))?,
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--games" => a.games = Some(v()?.parse().map_err(|e| format!("--games: {e}"))?),
            "--only" => a.only = Some(v()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.config.as_os_str().is_empty() {
        return Err(
            "usage: opp_eval --config <toml> [--model <bundle>] [--gate <logit margin>] [--decisions] [--threads N] [--games N] [--only <substring>]".into(),
        );
    }
    Ok(a)
}

/// One tick of the game as the observer saw it.
struct Snap {
    /// The world at this tick, kept only for decision ticks until they are processed.
    world: Option<World<f32>>,
    /// The inputs applied in the step that led to this tick, by slot.
    applied: [Wire; 2],
    pos1: [f32; 2],
}

/// Position errors per method and tick of the window, over all windows.
#[derive(Default, Clone)]
struct Errors {
    /// `errs[method][k - 1]`.
    errs: Vec<Vec<Vec<f32>>>,
    /// Per method, against the oracle, over the window: `[tp, fp, fn, tn]` of "a hammer hit on us" and of "its hook holds us at the end".
    hammer: [[u64; 4]; 4],
    hook: [[u64; 4]; 4],
    dec: DecAgree,
}

/// Agreement of the shadow hybrids' decisions with the oracle's (direction, jump, hook, the three together).
#[derive(Default, Clone)]
struct DecAgree {
    n: u64,
    /// Decisions where the oracle shadow decided other than the hold shadow (the windows where the prediction could matter).
    differs: u64,
    /// Equal to the oracle's: `[dir, jump, hook, all three]`, hold and model, over all decisions and over the `differs` ones.
    hold: [u64; 4],
    model: [u64; 4],
    hold_d: [u64; 4],
    model_d: [u64; 4],
    /// Work (ms of the work clock, D-042) of every decision of the hold and the model shadow: same views, so the two lists are paired.
    work_hold: Vec<f32>,
    work_model: Vec<f32>,
}

impl Errors {
    fn new() -> Errors {
        Errors {
            errs: vec![vec![Vec::new(); K]; METHODS.len()],
            hammer: [[0; 4]; 4],
            hook: [[0; 4]; 4],
            dec: DecAgree::default(),
        }
    }

    fn merge(&mut self, o: &Errors) {
        for m in 0..METHODS.len() {
            for c in 0..4 {
                self.hammer[m][c] += o.hammer[m][c];
                self.hook[m][c] += o.hook[m][c];
            }
        }
        let (a, b) = (&mut self.dec, &o.dec);
        a.n += b.n;
        a.differs += b.differs;
        a.work_hold.extend_from_slice(&b.work_hold);
        a.work_model.extend_from_slice(&b.work_model);
        for c in 0..4 {
            a.hold[c] += b.hold[c];
            a.model[c] += b.model[c];
            a.hold_d[c] += b.hold_d[c];
            a.model_d[c] += b.model_d[c];
        }
        for m in 0..METHODS.len() {
            for k in 0..K {
                self.errs[m][k].extend_from_slice(&o.errs[m][k]);
            }
        }
    }
}

/// The opponent's true inputs of the window, handed to the oracle shadow before each of its decisions.
struct OracleModel(Rc<RefCell<Vec<PredictedInput>>>);

impl WindowModel for OracleModel {
    fn name(&self) -> &str {
        "oracle"
    }
    fn predict(&mut self, _ctx: &WindowCtx<'_>, out: &mut [Option<PredictedInput>]) {
        for (o, p) in out.iter_mut().zip(self.0.borrow().iter()) {
            *o = Some(*p);
        }
    }
}

/// Three hybrids that see the recorded game's views and decide, with the window rolled by hold, by the model and by the oracle.
struct Shadows {
    hold: HybridBrain,
    model: HybridBrain,
    oracle: HybridBrain,
    truth: Rc<RefCell<Vec<PredictedInput>>>,
    map: Arc<ddai_physics::map::MapData>,
}

struct GameEval {
    shadows: Option<Shadows>,
    pw: PhysicsWorld,
    model: Option<OppPredictor>,
    lag: usize,
    decide_every: i32,
    tick0: i32,
    snaps: Vec<Snap>,
    next_decision: i32,
    errors: Errors,
    out: [Option<PredictedInput>; K],
}

impl GameEval {
    fn snap_at(&self, tick: i32) -> Option<&Snap> {
        usize::try_from(tick - self.tick0).ok().and_then(|i| self.snaps.get(i))
    }

    fn on_tick(&mut self, sim: &Sim, tick: i32) {
        let w = sim.pw.inner();
        if self.snaps.is_empty() {
            self.tick0 = tick;
            self.next_decision = tick + (self.decide_every - tick.rem_euclid(self.decide_every)) % self.decide_every;
        }
        let input = |id: u8| w.cores.get(id).map_or_else(Wire::default, |c| c.input);
        let pos1 = w.cores.get(1).map_or([0.0, 0.0], |c| [c.pos.x, c.pos.y]);
        let keep = tick % self.decide_every == 0;
        self.snaps.push(Snap {
            world: keep.then(|| w.clone()),
            applied: [input(0), input(1)],
            pos1,
        });
        // A decision tick is processed once the window after it is complete.
        while self.next_decision + K as i32 <= tick {
            let t = self.next_decision;
            self.process(t);
            self.next_decision += self.decide_every;
        }
    }

    fn finish(&mut self, last: i32) {
        while self.next_decision < last {
            let t = self.next_decision;
            self.process(t);
            self.next_decision += self.decide_every;
        }
    }

    /// Rolls the exact physics from decision tick `t` and records the opponent's position errors.
    fn process(&mut self, t: i32) {
        let last = self.tick0 + self.snaps.len() as i32 - 1;
        let steps = ((last - t).max(0) as usize).min(K);
        let lag = self.lag;
        let Some(world) = self.snap_at(t).and_then(|s| s.world.as_ref()).cloned() else {
            return;
        };
        if steps < lag || steps == 0 {
            // Not even the in-flight window is known (the game ended inside it).
            if let Some(s) = self.snaps.get_mut((t - self.tick0) as usize) {
                s.world = None;
            }
            return;
        }
        // Inputs applied at steps t .. t + steps are those stored with ticks t + 1 .. t + steps.
        let ours: Vec<Wire> = (0..steps)
            .map(|j| self.snap_at(t + 1 + j as i32).expect("tick").applied[0])
            .collect();
        let theirs: Vec<Wire> = (0..steps)
            .map(|j| self.snap_at(t + 1 + j as i32).expect("tick").applied[1])
            .collect();
        let truth: Vec<[f32; 2]> = (0..steps)
            .map(|j| self.snap_at(t + 1 + j as i32).expect("tick").pos1)
            .collect();
        let prev_true = world.cores.get(1).map_or_else(Wire::default, |c| c.input);

        // The model is asked once per decision, in order, with our in-flight inputs.
        let mut predicted: [Option<PredictedInput>; K] = [None; K];
        if let Some(m) = self.model.as_mut() {
            self.out = [None; K];
            m.predict(
                &WindowCtx {
                    world: &world,
                    self_id: 0,
                    victim_id: 1,
                    in_flight: &ours[..lag],
                },
                &mut self.out,
            );
            predicted = self.out;
        }

        let mut hammer_flag = [false; 4];
        let mut hook_flag = [false; 4];
        let pw = &mut self.pw;
        for (mi, name) in METHODS.iter().enumerate() {
            if *name == "model" && self.model.is_none() {
                continue;
            }
            pw.sync_from(&world);
            let opp_tee = pw.get_tee(1);
            let Some(opp_tee) = opp_tee else { continue };
            let snap_input = enemy_input_from_tee(&opp_tee);
            let mut fire = prev_true.fire;
            for j in 0..steps {
                pw.set_input(0, from_ddnet_input(&ours[j]));
                let x: PlayerInput = match *name {
                    "snap" => snap_input,
                    "true" => from_ddnet_input(&prev_true),
                    "model" => match predicted[j] {
                        Some(p) => input_from_prediction(&p, &snap_input, &mut fire),
                        None => snap_input,
                    },
                    _ => from_ddnet_input(&theirs[j]),
                };
                pw.set_input(1, x);
                let events = pw.step();
                if j < lag {
                    hammer_flag[mi] |= events
                        .iter()
                        .any(|e| matches!(e, WorldEvent::HammerHit { from: 1, to: 0 }));
                }
                let Some(tee) = pw.get_tee(1) else { break };
                if j + 1 == lag {
                    hook_flag[mi] = tee.hooked_player == 0;
                }
                let e = ((tee.pos.x as f32 - truth[j][0]).powi(2) + (tee.pos.y as f32 - truth[j][1]).powi(2)).sqrt();
                self.errors.errs[mi][j].push(e);
            }
        }
        let o = METHODS.len() - 1;
        for mi in 0..METHODS.len() {
            for (flags, table) in [
                (&hammer_flag, &mut self.errors.hammer),
                (&hook_flag, &mut self.errors.hook),
            ] {
                let c = match (flags[mi], flags[o]) {
                    (true, true) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                };
                table[mi][c] += 1;
            }
        }
        if let Some(sh) = self.shadows.as_mut() {
            // The oracle's window: the opponent's true inputs, `press` where its fire counter moved on to an odd value.
            let mut prev = prev_true;
            let truth: Vec<PredictedInput> = theirs[..lag]
                .iter()
                .map(|cur| {
                    let press = cur.fire != prev.fire && cur.fire & 1 != 0;
                    prev = *cur;
                    PredictedInput {
                        direction: cur.direction,
                        jump: cur.jump != 0,
                        hook: cur.hook != 0,
                        press,
                        aim: ddai_jsmath::atan2(f64::from(cur.target_y), f64::from(cur.target_x)),
                    }
                })
                .collect();
            *sh.truth.borrow_mut() = truth;
            let ids = [0, 1];
            if let Some(obs) = ddai_env::observe::observation(&world, &sh.map, 0, &ids, Some(1)) {
                let view = WorldView {
                    world: &world,
                    self_id: 0,
                    lag_ticks: lag as u32,
                    in_flight: &ours[..lag],
                };
                let (h, m, or) = (
                    sh.hold.decide_in(&obs, Some(&view)),
                    sh.model.decide_in(&obs, Some(&view)),
                    sh.oracle.decide_in(&obs, Some(&view)),
                );
                let eq = |x: &ddai_brain::Action| {
                    let (d, j, k) = (x.direction == or.direction, x.jump == or.jump, x.hook == or.hook);
                    [d, j, k, d && j && k]
                };
                let (eh, em) = (eq(&h), eq(&m));
                let d = &mut self.errors.dec;
                d.work_hold.extend(shadow_work_ms(&sh.hold));
                d.work_model.extend(shadow_work_ms(&sh.model));
                d.n += 1;
                let differs = !eh[3];
                d.differs += u64::from(differs);
                for c in 0..4 {
                    d.hold[c] += u64::from(eh[c]);
                    d.model[c] += u64::from(em[c]);
                    if differs {
                        d.hold_d[c] += u64::from(eh[c]);
                        d.model_d[c] += u64::from(em[c]);
                    }
                }
            }
        }
        if let Some(s) = self.snaps.get_mut((t - self.tick0) as usize) {
            s.world = None;
        }
    }
}

/// Work of the decision a brain just made, in ms of the work clock: `(ticks x sim_tees + units + proposal_units) x 1.25 us`, from its telemetry.
fn shadow_work_ms(b: &HybridBrain) -> Option<f32> {
    let v: serde_json::Value = serde_json::from_str(&b.telemetry()?).ok()?;
    let last = v.get("last").filter(|l| !l.is_null())?;
    let w = last.get("work")?;
    let n = |k: &str| w.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let tees = last
        .get("sim_tees")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1)
        .max(1);
    Some(
        ((n("ticks") * tees + n("units") + n("proposal_units")) as f64 * ddai_planner::hybrid::WORK_US_PER_TEE_TICK
            / 1000.0) as f32,
    )
}

fn run_game(
    cfg: &RunConfig,
    arenas: &BTreeMap<String, Arena>,
    cond_i: usize,
    g: u32,
    model: Option<&PathBuf>,
    gate: f32,
    decisions: bool,
) -> Result<Errors, String> {
    let cond = &cfg.condition[cond_i];
    let arena = &arenas[&cond.arena];
    let rules = cfg.rules_for(cond);
    let slots: Vec<PlayerSpec> = cond.slots();
    if slots.len() != 2 {
        return Err("two players needed".into());
    }
    let players: Vec<PlayerSetup> = slots
        .iter()
        .map(|s| {
            let b = builtin_brain(s).map_err(|e| e.to_string())?;
            let label = s.label.clone().unwrap_or_else(|| b.name().to_string());
            Ok(PlayerSetup {
                brain: b,
                lag: s.lag,
                label,
            })
        })
        .collect::<Result<_, String>>()?;
    let seed = cfg.base_seed.wrapping_add(u64::from(g));
    let template = arena.new_world();
    let shadows = if decisions {
        let model_path = model.ok_or("--decisions needs --model")?;
        let truth = Rc::new(RefCell::new(Vec::new()));
        let build = |window: bool| -> Result<HybridBrain, String> {
            let (mut c, clock): (_, ClockKind) = hybrid_config(&slots[0]).map_err(|e| e.to_string())?;
            c.window_model = window;
            let mut b = HybridBrain::new(c, clock, Box::new(NoProposer))?;
            b.reset(&ResetContext {
                map: Arc::clone(&arena.map),
                self_id: 0,
                seed,
            });
            Ok(b)
        };
        let hold = build(false)?;
        let mut m = build(true)?;
        m.set_window_model(Box::new(OppPredictor::load(model_path)?.with_gate(gate)));
        let mut o = build(true)?;
        o.set_window_model(Box::new(OracleModel(Rc::clone(&truth))));
        Some(Shadows {
            hold,
            model: m,
            oracle: o,
            truth,
            map: Arc::clone(&arena.map),
        })
    } else {
        None
    };
    let mut ev = GameEval {
        shadows,
        pw: PhysicsWorld::from_world(template, Arc::clone(&arena.map)),
        model: model
            .map(|p| OppPredictor::load(p).map(|m| m.with_gate(gate)))
            .transpose()?,
        lag: slots[0].lag as usize,
        decide_every: rules.decide_every,
        tick0: 0,
        snaps: Vec::new(),
        next_decision: 0,
        errors: Errors::new(),
        out: [None; K],
    };
    let mut last = 0;
    play_game_watched(arena, &rules, seed, layout_of(arena, g), players, &mut |sim, tick| {
        ev.on_tick(sim, tick);
        last = tick;
        let w = sim.pw.inner();
        // Stop at the first freeze, as the recorder does.
        !(ddai_env::observe::is_out(w, 0) || ddai_env::observe::is_out(w, 1))
    })
    .map_err(|e| e.to_string())?;
    ev.finish(last);
    Ok(ev.errors)
}

fn pctl(v: &mut [f32], p: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

fn main() -> Result<(), String> {
    let a = parse_args()?;
    let text = std::fs::read_to_string(&a.config).map_err(|e| format!("{}: {e}", a.config.display()))?;
    let cfg = RunConfig::parse(&text).map_err(|e| e.to_string())?;
    let arenas_dir = cfg
        .arenas_dir
        .as_deref()
        .map_or_else(|| PathBuf::from("configs/arenas"), PathBuf::from);
    let map_dir = cfg.map_dir.as_deref().map_or_else(
        || PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps"),
        PathBuf::from,
    );
    let arenas = load_arenas(&cfg, &arenas_dir, &map_dir).map_err(|e| e.to_string())?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let mut pooled = Errors::new();
    for (ci, cond) in cfg.condition.iter().enumerate() {
        if a.only.as_deref().is_some_and(|s| !cond.name.contains(s)) {
            continue;
        }
        let n = a.games.unwrap_or_else(|| cfg.games_for(cond));
        let lag = cond.slots()[0].lag as usize;
        let res: Vec<Result<Errors, String>> = pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|g| run_game(&cfg, &arenas, ci, g, a.model.as_ref(), a.gate, a.decisions))
                .collect()
        });
        let mut e = Errors::new();
        for r in res {
            e.merge(&r?);
        }
        pooled.merge(&e);
        println!("\n### {} ({n} games, {} windows)\n", cond.name, e.errs[0][0].len());
        print_table(&mut e, lag);
        print_events(&e);
    }
    println!("\n### all conditions pooled ({} windows)\n", pooled.errs[0][0].len());
    print_table(&mut pooled, 0);
    print_events(&pooled);
    Ok(())
}

fn print_table(e: &mut Errors, lag: usize) {
    println!(
        "| k | method | n | exact (<1 px) % | median px | p90 px | mean px |\n|---:|---|---:|---:|---:|---:|---:|"
    );
    for k in 1..=K {
        for (mi, name) in METHODS.iter().enumerate() {
            let v = &mut e.errs[mi][k - 1];
            if v.is_empty() {
                continue;
            }
            let exact = 100.0 * v.iter().filter(|&&x| x < 1.0).count() as f64 / v.len() as f64;
            let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
            let mark = if k == lag { " (window end)" } else { "" };
            println!(
                "| {k}{mark} | {name} | {} | {exact:.1} | {:.2} | {:.2} | {mean:.2} |",
                v.len(),
                pctl(v, 0.5),
                pctl(v, 0.9)
            );
        }
    }
}

fn print_events(e: &Errors) {
    let pc = |a: u64, n: u64| 100.0 * a as f64 / n.max(1) as f64;
    for (what, t) in [
        ("a hammer hit of the opponent on us inside the window", &e.hammer),
        ("the opponent's hook holds us at the end of the window", &e.hook),
    ] {
        println!(
            "\nEvent agreement with the oracle, {what} (windows with it in the oracle: {}):\n",
            t[3][0] + t[3][2]
        );
        println!(
            "| method | tp | fp | fn | tn | accuracy % | precision % | recall % |\n|---|---:|---:|---:|---:|---:|---:|---:|"
        );
        for (mi, name) in METHODS.iter().enumerate().take(3) {
            let [tp, fp, fnn, tn] = t[mi];
            println!(
                "| {name} | {tp} | {fp} | {fnn} | {tn} | {:.1} | {:.1} | {:.1} |",
                pc(tp + tn, tp + fp + fnn + tn),
                pc(tp, tp + fp),
                pc(tp, tp + fnn)
            );
        }
    }
    let d = &e.dec;
    if d.n > 0 {
        let (mut wh, mut wm) = (d.work_hold.clone(), d.work_model.clone());
        println!(
            "\nWork per decision on identical views (work clock, ms) p50 / p90 / p99 / max: hold {:.2} / {:.2} / {:.2} / {:.2}; model {:.2} / {:.2} / {:.2} / {:.2} ({} decisions)",
            pctl(&mut wh, 0.5),
            pctl(&mut wh, 0.9),
            pctl(&mut wh, 0.99),
            pctl(&mut wh, 1.0),
            pctl(&mut wm, 0.5),
            pctl(&mut wm, 0.9),
            pctl(&mut wm, 0.99),
            pctl(&mut wm, 1.0),
            wh.len()
        );
        println!(
            "\nDecision agreement with the oracle shadow ({} decisions, {} where the oracle decided other than hold: {:.1}%):\n",
            d.n,
            d.differs,
            pc(d.differs, d.n)
        );
        println!(
            "| shadow | direction % | jump % | hook % | all three % | all three % on the decisions where oracle != hold |\n|---|---:|---:|---:|---:|---:|"
        );
        for (name, all, dd) in [("hold", &d.hold, &d.hold_d), ("model", &d.model, &d.model_d)] {
            println!(
                "| {name} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
                pc(all[0], d.n),
                pc(all[1], d.n),
                pc(all[2], d.n),
                pc(all[3], d.n),
                pc(dd[3], d.differs)
            );
        }
    }
}
