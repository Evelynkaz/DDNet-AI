//! Task 3.21 (E-036): the opponent-input predictors on **live clips** at the live window (2 ticks), through the live path (`LiveWorld`, the roll with
//! the target playing the model's inputs). Read-only on the clips; sessions are never mixed (`--sessions`).
//!
//! For every duel frame `T` (regime gate, nobody frozen, the next frame 2 ticks later) the snapshot at `T + 2` shows the opponent's direction, aim and hook state
//! of window tick `k = 1` and, through its `attack_tick`, in which of the ticks `k = 0, 1` it swung its weapon. The world rolled to `T + 2` with the target playing
//! hold / the model's inputs is compared with the snapshot at `T + 2` (position of the opponent and of ours).
//!
//! ```text
//! cargo run --release -p ddai-env --example live_eval -- --clips DIR... --model name=path[,name=path] [--lag 2] [--sessions 0,1] [--thr 0]
//! ```

#[path = "../live_data/convert.rs"]
mod convert;

use std::path::PathBuf;

use ddai_brain::Observation;
use ddai_oppnet::AnyPredictor;
use ddai_oppnet::clipdata::{ClipGame, labels_at};
use ddai_oppnet::feature::wrap_angle;
use ddai_oppnet::frame::TeeFrame;
use ddai_oppnet::live::{RegimeGate, victim_input};
use ddai_physics::core::{MAX_CLIENTS, PlayerInput as Wire};
use ddai_planner::hybrid::window::{PredictedInput, WindowCtx, WindowModel};
use ddai_world::player_input_from_net;

const H: usize = 4;

/// What the evaluation needs of a model (any version).
trait Model {
    fn reset(&mut self);
    fn call(&mut self, ctx: &WindowCtx<'_>, out: &mut [Option<PredictedInput>]);
    /// The press logit of window tick `k` of the last call (`None`: the model gives none).
    fn press_logit(&self, k: usize) -> Option<f32>;
    /// The logit above which the model's own decoding predicts a swing.
    fn press_thr(&self) -> f32;
}

struct Any(AnyPredictor);

impl Model for Any {
    fn reset(&mut self) {
        self.0.reset();
    }
    fn call(&mut self, ctx: &WindowCtx<'_>, out: &mut [Option<PredictedInput>]) {
        self.0.predict(ctx, out);
    }
    fn press_logit(&self, k: usize) -> Option<f32> {
        let (l, head) = match &self.0 {
            AnyPredictor::V1(p) => (p.logits(), ddai_oppnet::feature::HEAD_DIM),
            AnyPredictor::V2(p) => (p.logits(), ddai_oppnet::v2::feature::HEAD_DIM),
        };
        l.get(k * head + 5).copied()
    }
    fn press_thr(&self) -> f32 {
        match &self.0 {
            AnyPredictor::V1(_) => 0.0,
            AnyPredictor::V2(p) => p.decode().press,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Roll {
    me: [f32; 4],
    opp: [f32; 4],
}

#[derive(Clone, Copy, Default)]
struct ModelPred {
    out: [Option<PredictedInput>; H],
    press_logit: [Option<f32>; H],
    roll: Roll,
}

#[derive(Clone)]
struct Pred {
    hold_roll: Roll,
    models: Vec<ModelPred>,
}

fn roll_of(w: &ddai_physics::world::World<f32>, own: i32, opp: i32) -> Roll {
    let g = |id: i32| {
        w.cores
            .get(id as u8)
            .map_or([f32::NAN; 4], |c| [c.pos.x, c.pos.y, c.vel.x, c.vel.y])
    };
    Roll {
        me: g(own),
        opp: g(opp),
    }
}

#[derive(Default, Clone)]
struct Agg {
    n: u64,
    dir_n: u64,
    dir_hold: u64,
    dir_model: u64,
    hook_n: u64,
    hook_hold: u64,
    hook_model: u64,
    aim_n: u64,
    aim_hold: f64,
    aim_model: f64,
    /// Press events (ticks k = 0, 1 pooled), predicted positives, true positives; window-level (any press in the window).
    ev: u64,
    pos: u64,
    tp: u64,
    win_n: u64,
    win_ev: u64,
    win_pos: u64,
    win_tp: u64,
    /// Press logit samples `(logit, label)`.
    scores: Vec<(f32, bool)>,
    err_opp: [Vec<f32>; 2],
    err_me: [Vec<f32>; 2],
}

fn dist(a: [f32; 4], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn pct(a: u64, n: u64) -> f64 {
    100.0 * a as f64 / n.max(1) as f64
}

fn q(v: &mut [f32], p: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

fn auc(s: &[(f32, bool)]) -> f64 {
    let mut v: Vec<(f32, bool)> = s.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut rank_sum, mut npos) = (0.0f64, 0u64);
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for x in &v[i..=j] {
            if x.1 {
                rank_sum += r;
                npos += 1;
            }
        }
        i = j + 1;
    }
    let nneg = v.len() as u64 - npos;
    if npos == 0 || nneg == 0 {
        return f64::NAN;
    }
    (rank_sum - npos as f64 * (npos as f64 + 1.0) / 2.0) / (npos as f64 * nneg as f64)
}

fn logloss(s: &[(f32, bool)]) -> (f64, f64) {
    let n = s.len().max(1) as f64;
    let base = s.iter().filter(|x| x.1).count() as f64 / n;
    let ll_const = -(base * base.max(1e-9).ln() + (1.0 - base) * (1.0 - base).max(1e-9).ln());
    let ll: f64 = s
        .iter()
        .map(|&(z, y)| {
            let z = f64::from(z);
            z.max(0.0) + (-z.abs()).exp().ln_1p() - if y { z } else { 0.0 }
        })
        .sum::<f64>()
        / n;
    (ll, ll_const)
}

fn main() -> Result<(), String> {
    let mut clips = Vec::new();
    let mut models: Vec<(String, PathBuf)> = Vec::new();
    let mut lag = 2usize;
    let mut sessions: Vec<u8> = vec![0, 1];
    let mut thr: Option<f32> = None;
    let mut decode = String::new();
    let mut maps = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache");
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--clips" => clips.push(PathBuf::from(v()?)),
            "--model" => {
                for m in v()?.split(',') {
                    let (n, p) = m.split_once('=').ok_or("--model name=path[,name=path]")?;
                    models.push((n.to_string(), PathBuf::from(p)));
                }
            }
            "--lag" => lag = v()?.parse().map_err(|e| format!("--lag: {e}"))?,
            "--sessions" => {
                sessions = v()?
                    .split(',')
                    .map(|s| s.parse().map_err(|e| format!("--sessions: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--thr" => thr = Some(v()?.parse().map_err(|e| format!("--thr: {e}"))?),
            "--decode" => decode = v()?,
            "--maps" => maps = PathBuf::from(v()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if clips.is_empty() || lag == 0 || lag % 2 != 0 || lag > 2 * (H / 2) {
        return Err(
            "usage: live_eval --clips DIR... --model name=path[,..] [--lag 2|4] [--sessions 0,1] [--thr 0]".into(),
        );
    }
    let gate = RegimeGate::default();
    let keep = [true; MAX_CLIENTS];
    // Per session: per model: the aggregate. Index 0 of the models is hold.
    let names: Vec<String> = std::iter::once("hold".to_string())
        .chain(models.iter().map(|m| m.0.clone()))
        .collect();
    let mut aggs: std::collections::BTreeMap<u8, Vec<Agg>> = std::collections::BTreeMap::new();
    for (path, clip, session) in convert::collect_clips(&clips)? {
        if !sessions.contains(&session) {
            continue;
        }
        let map = convert::load_map(&maps, &clip)?;
        let mut ms: Vec<Box<dyn Model>> = models
            .iter()
            .map(|(_, p)| {
                let m = match AnyPredictor::load(p)? {
                    AnyPredictor::V2(m) => {
                        let mut d = *m.decode();
                        ddai_oppnet::v2::predictor::apply_decode_overrides(&mut d, &decode)?;
                        AnyPredictor::V2(m.with_decode(d))
                    }
                    other => other,
                };
                Ok::<_, String>(Box::new(Any(m)) as Box<dyn Model>)
            })
            .collect::<Result<_, _>>()?;
        let mut obs: Option<Observation> = None;
        let mut last_tick = i32::MIN;
        let mut own_buf: Vec<Wire> = Vec::new();
        let mut victim_buf: Vec<Vec<Wire>> = vec![Vec::new(); ms.len()];
        let src = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let thrs: Vec<f32> = ms.iter().map(|m| thr.unwrap_or_else(|| m.press_thr())).collect();
        let (games, preds) = convert::convert_with(&clip, map, &gate, &src, session, &mut |c| {
            let w = c.lw.base_world();
            let (Some(me), Some(op)) = (
                TeeFrame::from_world(w, c.own, c.opp),
                TeeFrame::from_world(w, c.opp, c.own),
            ) else {
                return None;
            };
            // A new run (a gap or another target): the models start afresh.
            if w.tick != last_tick + 2 {
                for m in &mut ms {
                    m.reset();
                }
            }
            last_tick = w.tick;
            for m in &mut ms {
                m.call(
                    &WindowCtx {
                        world: w,
                        self_id: c.own,
                        victim_id: c.opp,
                        in_flight: &[],
                    },
                    &mut [],
                );
            }
            if !c.duel || me.freeze_left > 0 || op.freeze_left > 0 {
                return None;
            }
            let t = w.tick;
            // Our inputs for the steps T .. T + lag, tagged T + 1 ..= T + lag, from the frames after this one.
            let sent_at = |tag: i32| {
                c.clip.frames[c.idx + 1..(c.idx + 4).min(c.clip.frames.len())]
                    .iter()
                    .flat_map(|f| f.sent.iter())
                    .find(|s| s.tick == tag)
                    .map(|s| player_input_from_net(s.input.to_net()))
            };
            let inflight: Option<Vec<(i32, Wire)>> =
                (1..=lag as i32).map(|k| sent_at(t + k).map(|i| (t + k, i))).collect();
            let inflight = inflight?;
            let to_tick = t + lag as i32;
            let obs = obs.get_or_insert_with(|| c.lw.build_observation(c.lw.base_world(), None));
            c.lw.own_inputs_over(to_tick, &inflight, &mut own_buf);
            let hold = c.lw.held_input_of(c.opp).unwrap_or_default();
            let hold_world =
                c.lw.predict_local_observation_with(to_tick, &inflight, &keep, Some(c.opp), obs, None);
            let hold_roll = roll_of(hold_world, c.own, c.opp);
            let mut mp = Vec::with_capacity(ms.len());
            for (mi, m) in ms.iter_mut().enumerate() {
                let mut out = [None; H];
                m.call(
                    &WindowCtx {
                        world: c.lw.base_world(),
                        self_id: c.own,
                        victim_id: c.opp,
                        in_flight: &own_buf,
                    },
                    &mut out[..lag.min(H)],
                );
                let mut pl = [None; H];
                for (k, p) in pl.iter_mut().enumerate() {
                    *p = m.press_logit(k);
                }
                let mut fire = hold.fire;
                victim_buf[mi].clear();
                for slot in out.iter().take(lag) {
                    victim_buf[mi].push(match slot {
                        Some(p) => victim_input(p, &hold, &mut fire),
                        None => hold,
                    });
                }
                let rolled = c.lw.predict_local_observation_with(
                    to_tick,
                    &inflight,
                    &keep,
                    Some(c.opp),
                    obs,
                    Some((c.opp, &victim_buf[mi])),
                );
                mp.push(ModelPred {
                    out,
                    press_logit: pl,
                    roll: roll_of(rolled, c.own, c.opp),
                });
            }
            Some(Pred { hold_roll, models: mp })
        });
        let entry = aggs.entry(session).or_insert_with(|| vec![Agg::default(); names.len()]);
        for (g, ps) in games.iter().zip(&preds) {
            accumulate(g, ps, lag, &thrs, entry);
        }
    }
    for (session, v) in &mut aggs {
        println!("\n## session {session}, window {lag}\n");
        report(&names, v);
    }
    // All the chosen sessions pooled.
    let mut pooled = vec![Agg::default(); names.len()];
    for v in aggs.values() {
        for (p, a) in pooled.iter_mut().zip(v) {
            merge(p, a);
        }
    }
    println!("\n## sessions {sessions:?} pooled, window {lag}\n");
    report(&names, &mut pooled);
    Ok(())
}

fn merge(p: &mut Agg, a: &Agg) {
    p.n += a.n;
    p.dir_n += a.dir_n;
    p.dir_hold += a.dir_hold;
    p.dir_model += a.dir_model;
    p.hook_n += a.hook_n;
    p.hook_hold += a.hook_hold;
    p.hook_model += a.hook_model;
    p.aim_n += a.aim_n;
    p.aim_hold += a.aim_hold;
    p.aim_model += a.aim_model;
    p.ev += a.ev;
    p.pos += a.pos;
    p.tp += a.tp;
    p.win_n += a.win_n;
    p.win_ev += a.win_ev;
    p.win_pos += a.win_pos;
    p.win_tp += a.win_tp;
    p.scores.extend_from_slice(&a.scores);
    for i in 0..2 {
        p.err_opp[i].extend_from_slice(&a.err_opp[i]);
        p.err_me[i].extend_from_slice(&a.err_me[i]);
    }
}

fn accumulate(g: &ClipGame, ps: &[Option<Pred>], lag: usize, thrs: &[f32], aggs: &mut [Agg]) {
    for (i, p) in ps.iter().enumerate() {
        let Some(p) = p else { continue };
        let fut = lag / 2;
        if !g.consecutive(i, 0, fut) {
            continue;
        }
        let l = labels_at(g, i, H);
        let t0 = &g.ticks[i];
        let truth = &g.ticks[i + fut];
        if !truth.frames[1].alive || truth.frames[1].freeze_left > 0 || !truth.frames[0].alive {
            continue;
        }
        let tp_opp = truth.frames[1].pos;
        let tp_me = truth.frames[0].pos;
        for (mi, a) in aggs.iter_mut().enumerate() {
            // Index 0 is hold, the models follow.
            let (roll, mp) = if mi == 0 {
                (p.hold_roll, None)
            } else {
                (p.models[mi - 1].roll, Some(&p.models[mi - 1]))
            };
            a.n += 1;
            a.err_opp[0].push(dist(roll.opp, tp_opp));
            a.err_me[0].push(dist(roll.me, tp_me));
            // The tick-1 reading (the snapshot at T + 2 shows the step T + 1).
            if l.v_obs & 0b10 != 0 {
                let hold_dir = t0.frames[1].direction;
                let hold_hook = t0.frames[1].hook_state > 0;
                let a_dir = l.dir[1];
                let a_hook = l.hook & 0b10 != 0;
                let a_aim = f64::from(t0.frames[1].angle) + f64::from(l.aim_delta[1]);
                a.dir_n += 1;
                a.hook_n += 1;
                a.aim_n += 1;
                match mp.and_then(|m| m.out[1]) {
                    None if mp.is_some() || mi == 0 => {
                        a.dir_model += u64::from(hold_dir == a_dir);
                        a.hook_model += u64::from(hold_hook == a_hook);
                        a.aim_model += wrap_angle(f64::from(t0.frames[1].angle) - a_aim).abs();
                    }
                    Some(o) => {
                        a.dir_model += u64::from(o.direction == i32::from(a_dir));
                        a.hook_model += u64::from(o.hook == a_hook);
                        a.aim_model += wrap_angle(o.aim - a_aim).abs();
                    }
                    None => {}
                }
                a.dir_hold += u64::from(hold_dir == a_dir);
                a.hook_hold += u64::from(hold_hook == a_hook);
                a.aim_hold += wrap_angle(f64::from(t0.frames[1].angle) - a_aim).abs();
            }
            // Swings in k = 0, 1.
            let mut any_ev = false;
            let mut any_pred = false;
            let mut valid = true;
            for k in 0..2usize {
                if l.v_press >> k & 1 == 0 {
                    valid = false;
                    continue;
                }
                let ev = l.press >> k & 1 != 0;
                let pred = mp.is_some_and(|m| m.press_logit[k].is_some_and(|z| z > thrs[mi - 1]));
                a.ev += u64::from(ev);
                a.pos += u64::from(pred);
                a.tp += u64::from(ev && pred);
                any_ev |= ev;
                any_pred |= pred;
                if let Some(z) = mp.and_then(|m| m.press_logit[k]) {
                    a.scores.push((z, ev));
                }
            }
            if valid {
                a.win_n += 1;
                a.win_ev += u64::from(any_ev);
                a.win_pos += u64::from(any_pred);
                a.win_tp += u64::from(any_ev && any_pred);
            }
        }
    }
}

fn report(names: &[String], v: &mut [Agg]) {
    println!(
        "| arm | samples | dir k=1 % | hook k=1 % | aim k=1 err rad | opp <1px % / mean / p90 px | me <1px % / mean / p90 px |\n|---|---:|---|---|---|---|---|"
    );
    for (name, a) in names.iter().zip(v.iter_mut()) {
        let model = name != "hold";
        let (d, h, am) = if model {
            (a.dir_model, a.hook_model, a.aim_model)
        } else {
            (a.dir_hold, a.hook_hold, a.aim_hold)
        };
        let st = |e: &mut Vec<f32>| {
            let n = e.len().max(1) as f64;
            let ex = 100.0 * e.iter().filter(|&&x| x < 1.0).count() as f64 / n;
            let mean = e.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
            format!("{ex:.1} / {mean:.2} / {:.2}", q(e, 0.9))
        };
        let (mut eo, mut em) = (a.err_opp[0].clone(), a.err_me[0].clone());
        println!(
            "| {name} | {} | {:.1} ({}) | {:.1} | {:.4} | {} | {} |",
            a.n,
            pct(d, a.dir_n),
            a.dir_n,
            pct(h, a.hook_n),
            am / a.aim_n.max(1) as f64,
            st(&mut eo),
            st(&mut em)
        );
    }
    println!(
        "\nFire (weapon use) in the window ticks k = 0, 1 -- tick level, then window level (any swing in the window); hold never swings.\n"
    );
    println!(
        "| arm | swings | predicted | true pos | precision % | recall % | window: events / predicted / hit | AUC | logloss (const) |\n|---|---:|---:|---:|---:|---:|---|---:|---|"
    );
    for (name, a) in names.iter().zip(v.iter()) {
        let (ll, llc) = logloss(&a.scores);
        println!(
            "| {name} | {} | {} | {} | {:.1} | {:.1} | {} / {} / {} | {:.3} | {:.4} ({:.4}) |",
            a.ev,
            a.pos,
            a.tp,
            pct(a.tp, a.pos),
            pct(a.tp, a.ev),
            a.win_ev,
            a.win_pos,
            a.win_tp,
            auc(&a.scores),
            ll,
            llc
        );
    }
}
