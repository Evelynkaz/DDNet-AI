//! Information probes for the hook decision (task 8.6): is the information the planner's press / release depends on in the fly's INPUT,
//! or is the connectome network / its training the bottleneck?
//!
//! The same teacher labels as the BC (post-freeze rounds, the first `first` decisions after the freeze, `teacher-val` = seeds divisible by
//! `val_mod`), the same split by the latch (the own previous played hook key: *released* = the decision to press, *held* = to keep or
//! release), and three small models (one hidden layer, per latch, trained on the non-validation episodes with the BC's step weights):
//!
//! * **(a)** the fly's own encoder input vector (what the connectome receives; the masked view the hook head reads) of the last `frames`
//!   decisions stacked;
//! * **(b)** privileged exact-state features (the PPO critic's input: both tees, relative position, tile windows, BFS distances to freeze
//!   and exit, the clock) of the current decision plus the kinematics of the previous `frames - 1`;
//! * **(c)** (a) plus a few derived physics features: the distance and time to a freeze or kill tile along the current pull of the own
//!   hook and along the victim's way to the own tee, the pull direction against the nearest freeze, the victim's depth in the freeze.
//!
//! and next to them the AUROC of the real flies' hook probability on the same decisions. Every number is the AUROC on the validation
//! episodes (never trained on).

use std::collections::BTreeMap;
use std::path::Path;

use ddai_brain::{HOOK_IDLE, Observation};
use ddai_fly::bc::mask_own_hook;
use ddai_fly::brain::FlyBrainConfig;
use ddai_fly::bundle::FlyBrainTemplate;
use ddai_fly::flat_adam::{FlatAdamConfig, FlatAdamState};
use ddai_fly::rng::SplitMix64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::experiment::{Env, expand_home};
use crate::learner::{FlyLearner, FlyTrainConfig, Learner};
use crate::metrics::auroc;
use crate::ppo::critic::{EpisodeClock, INPUT_DIM, MapFields, critic_features};
use crate::runner::ExperimentConfig;
use crate::seq::{Window, observation_of, targets_of};
use crate::store::TeacherStore;
use crate::teacher_data::episode_to_seq;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeSpec {
    pub rounds: Vec<u32>,
    pub first: usize,
    pub frames: usize,
    pub hidden: usize,
    pub epochs: usize,
    pub seeds: usize,
    /// At most this many training rows per latch state (the rest is dropped at random).
    pub max_train_rows: usize,
    /// Only the probes on the flies' own network state (skip (a)-(c), already measured).
    pub state_only: bool,
    /// Skip the probes on the membrane state (the heaviest).
    pub no_membrane: bool,
}

impl Default for ProbeSpec {
    fn default() -> Self {
        ProbeSpec {
            rounds: vec![7, 8, 100],
            first: 16,
            frames: 4,
            hidden: 32,
            epochs: 40,
            seeds: 2,
            max_train_rows: 14_000,
            state_only: false,
            no_membrane: false,
        }
    }
}

/// One decision of the probe's data.
struct Row {
    /// The game (bank start) the decision belongs to: the cluster of the bootstrap.
    cluster: u64,
    val: bool,
    latch: bool,
    label: bool,
    weight: f32,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    /// The hook probability of each reference fly (validation rows only).
    fly: Vec<f32>,
    /// Per state fly: the calibrated DN z-scores of the hook view (the readout's input) of the last `frames` decisions stacked.
    dn: Vec<Vec<f32>>,
    /// Per state fly: the mean z of each hook type group of this decision, what the decoder's hook head reads.
    grp: Vec<Vec<f32>>,
    /// Per state fly: the full membrane state `V` after this decision (kept for validation rows and every third training row only).
    hid: Vec<Vec<f32>>,
}

/// The derived physics features of (c): a handful of numbers, all in `[-2, 2]`.
pub fn derived_features(obs: &Observation, f: &MapFields) -> Vec<f32> {
    let me = &obs.self_state;
    let mut out = Vec::with_capacity(12);
    // 1. The own hook's pull: from the own tee towards the hook (flying or attached).
    let hooking = me.hook_state != HOOK_IDLE && me.hook_state != ddai_brain::HOOK_RETRACTED;
    let (dx, dy) = (me.hook_pos.x - me.pos.x, me.hook_pos.y - me.pos.y);
    let len = (dx * dx + dy * dy).sqrt();
    if hooking && len > 1.0 {
        let (ux, uy) = (dx / len, dy / len);
        let (d_freeze, d_solid) = f.ray_distances(me.pos.x, me.pos.y, ux, uy, 600.0);
        let speed_along = me.vel.x * ux + me.vel.y * uy;
        out.extend([
            1.0,
            (d_freeze / 400.0).min(1.5),
            (d_solid / 400.0).min(1.5),
            (speed_along / 16.0).clamp(-2.0, 2.0),
            // Ticks to the freeze at the current speed along the pull (capped), in units of 50 ticks.
            if speed_along > 0.5 {
                (d_freeze / speed_along / 50.0).min(2.0)
            } else {
                2.0
            },
        ]);
    } else {
        out.extend([0.0, 1.5, 1.5, 0.0, 2.0]);
    }
    // 2. The victim: its way to the own tee (a hooked victim is pulled along it), its depth in the freeze and the distance to the nearest
    //    freeze tile from it along that way, and the pull direction against the direction to the nearest freeze tile.
    match obs.others.first() {
        Some(o) => {
            let (vx, vy) = (me.pos.x - o.pos.x, me.pos.y - o.pos.y);
            let vlen = (vx * vx + vy * vy).sqrt().max(1.0);
            let (ux, uy) = (vx / vlen, vy / vlen);
            let (d_freeze, _) = f.ray_distances(o.pos.x, o.pos.y, ux, uy, 600.0);
            out.push((d_freeze / 400.0).min(1.5));
            out.push(f32::from(f.exit_distance(o.pos.x, o.pos.y)) / 8.0);
            out.push(f32::from(f.freeze_distance(o.pos.x, o.pos.y)) / 8.0);
            out.push((vlen / 400.0).min(2.0));
            let pulled = f32::from(u8::from(o.hooked_player >= 0 || me.hooked_player >= 0));
            out.push(pulled);
            // Direction to the nearest freeze tile from the victim (8 compass rays), its cosine with the way to the own tee.
            let mut best = (f32::MAX, 0.0f32, 0.0f32);
            for k in 0..8 {
                let a = std::f32::consts::TAU * k as f32 / 8.0;
                let (cx, cy) = (a.cos(), a.sin());
                let (d, _) = f.ray_distances(o.pos.x, o.pos.y, cx, cy, 400.0);
                if d < best.0 {
                    best = (d, cx, cy);
                }
            }
            out.push(if best.0 < f32::MAX {
                best.1 * ux + best.2 * uy
            } else {
                0.0
            });
            out.push((best.0.min(400.0)) / 400.0);
        }
        None => out.extend([0.0; 8]),
    }
    out
}

/// Reads the post-freeze rows of the BC's teacher data.
#[allow(clippy::too_many_arguments)]
fn collect_rows(
    cfg: &ExperimentConfig,
    env: &Env,
    spec: &ProbeSpec,
    encoder: &FlyLearner,
    flies: &[FlyLearner],
    states: &[FlyBrainTemplate],
    log: &mut dyn FnMut(&str),
) -> Result<Vec<Row>, String> {
    let holdout = env.holdout_names();
    let fields: BTreeMap<String, MapFields> = env
        .arenas
        .iter()
        .map(|(n, a)| (n.clone(), MapFields::new(&a.map)))
        .collect();
    let mut rows = Vec::new();
    for dir in &cfg.teacher_base {
        let store = TeacherStore::open(&expand_home(dir)).map_err(|e| e.to_string())?;
        for ci in 0..store.manifest.chunks.len() {
            let round = store.manifest.chunks[ci].round;
            if !spec.rounds.contains(&round) {
                continue;
            }
            let chunk = store.read_chunk(ci).map_err(|e| e.to_string())?;
            for ep in &chunk.episodes {
                let arena = &store.manifest.arenas[ep.arena as usize].name;
                let (Some(map), Some(field)) = (env.maps.get(arena), fields.get(arena)) else {
                    continue;
                };
                if holdout.contains(arena) {
                    continue;
                }
                let val = cfg.teacher_data.val_mod > 0 && ep.seed.is_multiple_of(cfg.teacher_data.val_mod);
                let seq = episode_to_seq(ep, map, round, &cfg.teacher_data, arena);
                let obs: Vec<Observation> = seq.steps.iter().map(|s| observation_of(s, map, false)).collect();
                let masked: Vec<Observation> = obs.iter().map(mask_own_hook).collect();
                let clock = EpisodeClock {
                    freeze_tick: Some(ep.end_tick),
                    window: 250,
                    max_ticks: 3000,
                };
                let first_after = ep
                    .steps
                    .iter()
                    .position(|s| s.tick >= ep.end_tick)
                    .unwrap_or(ep.steps.len());
                // The reference flies' hook probability over the whole episode (a recurrent run from the episode's first decision).
                let fly_p: Vec<Vec<f32>> = if val {
                    let w = Window {
                        observations: obs.clone(),
                        targets: seq.steps.iter().map(|s| targets_of(s, 1.0, false)).collect(),
                        start: 0,
                        mirrored: false,
                    };
                    flies
                        .iter()
                        .map(|f| f.window_logits_played(&w).iter().map(|l| l.hook_prob()).collect())
                        .collect()
                } else {
                    Vec::new()
                };
                // The state flies: a recurrent run over the whole episode on the masked view (what the hook head reads), keeping the DN z-scores
                // and the membrane state after every decision.
                let runs: Vec<StateRun> = states
                    .iter()
                    .map(|t| {
                        let mut brain = t.instantiate(FlyBrainConfig::default());
                        ddai_brain::Brain::reset(
                            &mut brain,
                            &ddai_brain::ResetContext {
                                map: map.map.clone(),
                                self_id: 0,
                                seed: 1,
                            },
                        );
                        let (mut zs, mut vs) = (Vec::with_capacity(obs.len()), Vec::with_capacity(obs.len()));
                        for o in &masked {
                            let _ = brain.forward_logits(o);
                            zs.push(brain.last_dn_z().to_vec());
                            vs.push(brain.state_v().to_vec());
                        }
                        (zs, vs)
                    })
                    .collect();
                let mut inputs: Vec<Option<Vec<f32>>> = vec![None; obs.len()];
                let input_of = |i: usize, inputs: &mut Vec<Option<Vec<f32>>>| -> Vec<f32> {
                    if inputs[i].is_none() {
                        let mut v = Vec::new();
                        encoder.encoder_input(&masked[i], &mut v);
                        inputs[i] = Some(v);
                    }
                    inputs[i].clone().expect("just set")
                };
                for i in first_after..(first_after + spec.first).min(obs.len()) {
                    let st = &seq.steps[i];
                    if st.weight <= 0.0 {
                        continue;
                    }
                    let frame = |back: usize| i.saturating_sub(back);
                    let mut a = Vec::new();
                    for back in 0..spec.frames {
                        a.extend(input_of(frame(back), &mut inputs));
                    }
                    let mut b = critic_features(&obs[i], field, &clock);
                    debug_assert_eq!(b.len(), INPUT_DIM);
                    for back in 1..spec.frames {
                        let prev = critic_features(&obs[frame(back)], field, &clock);
                        b.extend_from_slice(&prev[..33]);
                    }
                    let mut c = a.clone();
                    c.extend(derived_features(&obs[i], field));
                    let keep_hid = val || rows.len() % 3 == 0;
                    let dn: Vec<Vec<f32>> = runs
                        .iter()
                        .map(|(zs, _)| {
                            (0..spec.frames)
                                .flat_map(|back| zs[frame(back)].iter().copied())
                                .collect()
                        })
                        .collect();
                    let hid: Vec<Vec<f32>> = if keep_hid {
                        runs.iter().map(|(_, vs)| vs[i].clone()).collect()
                    } else {
                        Vec::new()
                    };
                    let grp: Vec<Vec<f32>> = states
                        .iter()
                        .zip(&runs)
                        .map(|(t, (zs, _))| t.decoder().hook_group_means(&zs[i]))
                        .collect();
                    rows.push(Row {
                        cluster: ep.seed,
                        val,
                        latch: st.latch,
                        label: st.label.hook,
                        weight: st.weight,
                        a,
                        b,
                        c,
                        fly: fly_p.iter().map(|p| p[i]).collect(),
                        dn,
                        grp,
                        hid,
                    });
                }
            }
        }
        log(&format!("{dir}: {} rows so far", rows.len()));
    }
    Ok(rows)
}

fn summarise(fits: &[Fit]) -> (Vec<f64>, [f64; 2]) {
    let n = fits.len() as f64;
    (
        fits.iter().map(|f| f.auc).collect(),
        [
            fits.iter().map(|f| f.lo).sum::<f64>() / n,
            fits.iter().map(|f| f.hi).sum::<f64>() / n,
        ],
    )
}

/// One state fly's run over an episode: the DN z-scores and the membrane state after every decision.
type StateRun = (Vec<Vec<f32>>, Vec<Vec<f32>>);

// --- a small MLP ---------------------------------------------------------------------------------------

struct Mlp {
    d: usize,
    h: usize,
    p: Vec<f32>,
    /// No ReLU: a linear model (of rank `h`).
    linear: bool,
}

impl Mlp {
    fn new(d: usize, h: usize, seed: u64, linear: bool) -> Mlp {
        let mut rng = SplitMix64::new(seed);
        let mut p = vec![0.0f32; h * d + h + h + 1];
        let s1 = (2.0 / d as f32).sqrt();
        for w in &mut p[..h * d] {
            *w = (rng.next_f32_unit() * 2.0 - 1.0) * s1 * 1.7;
        }
        let s2 = (1.0 / h as f32).sqrt();
        for w in &mut p[h * d + h..h * d + 2 * h] {
            *w = (rng.next_f32_unit() * 2.0 - 1.0) * s2 * 1.7;
        }
        Mlp { d, h, p, linear }
    }

    fn forward(&self, x: &[f32], act: &mut [f32]) -> f32 {
        let (d, h) = (self.d, self.h);
        let (w1, rest) = self.p.split_at(h * d);
        let (b1, rest) = rest.split_at(h);
        let (w2, b2) = rest.split_at(h);
        let mut z = b2[0];
        for j in 0..h {
            let row = &w1[j * d..(j + 1) * d];
            let pre = b1[j] + row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>();
            let a = if self.linear { pre } else { pre.max(0.0) };
            act[j] = a;
            z += w2[j] * a;
        }
        z
    }

    /// Adds `weight * d loss / d params` of one example whose logit gradient is `dz` into `g`.
    fn backward(&self, x: &[f32], act: &[f32], dz: f32, g: &mut [f32]) {
        let (d, h) = (self.d, self.h);
        let (w2_off, b2_off) = (h * d + h, h * d + 2 * h);
        g[b2_off] += dz;
        for j in 0..h {
            g[w2_off + j] += dz * act[j];
            if self.linear || act[j] > 0.0 {
                let dh = dz * self.p[w2_off + j];
                g[h * d + j] += dh;
                let row = &mut g[j * d..(j + 1) * d];
                for (gw, v) in row.iter_mut().zip(x) {
                    *gw += dh * v;
                }
            }
        }
    }
}

fn weighted_bce(z: f32, y: bool, w: f32, class_w: [f32; 2]) -> (f32, f32) {
    let p = 1.0 / (1.0 + (-z).exp());
    let yf = f32::from(u8::from(y));
    let cw = class_w[usize::from(y)];
    let loss = -cw * w * (yf * p.max(1e-9).ln() + (1.0 - yf) * (1.0 - p).max(1e-9).ln());
    (loss, cw * w * (p - yf))
}

/// Per-feature mean and standard deviation of `x`.
fn standardiser(x: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let d = x[0].len();
    let n = x.len() as f64;
    let mut mean = vec![0.0f64; d];
    for r in x {
        for (m, v) in mean.iter_mut().zip(r.iter()) {
            *m += f64::from(*v);
        }
    }
    mean.iter_mut().for_each(|m| *m /= n);
    let mut var = vec![0.0f64; d];
    for r in x {
        for ((s, v), m) in var.iter_mut().zip(r.iter()).zip(&mean) {
            *s += (f64::from(*v) - m).powi(2);
        }
    }
    (
        mean.iter().map(|&m| m as f32).collect(),
        var.iter().map(|&v| ((v / n).sqrt() as f32).max(1e-3)).collect(),
    )
}

/// An AUROC with the 95% interval of a bootstrap over **games** (resampling the validation starts, not the decisions: the decisions of a
/// start are one cluster).
#[derive(Debug, Clone, Copy)]
struct Fit {
    auc: f64,
    lo: f64,
    hi: f64,
}

/// [`Fit`] of `scores` (one per row of `test`) with a cluster bootstrap of 300 resamples.
fn fit_of(scores: &[(f32, bool)], test: &[&Row]) -> Fit {
    let auc = auroc(scores);
    let mut clusters: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, r) in test.iter().enumerate() {
        clusters.entry(r.cluster).or_default().push(i);
    }
    let groups: Vec<&Vec<usize>> = clusters.values().collect();
    let mut rng = SplitMix64::new(0xB007);
    let mut aucs: Vec<f64> = (0..300)
        .map(|_| {
            let mut s = Vec::with_capacity(scores.len());
            for _ in 0..groups.len() {
                let g = groups[(rng.next_u64() % groups.len() as u64) as usize];
                s.extend(g.iter().map(|&i| scores[i]));
            }
            auroc(&s)
        })
        .collect();
    aucs.sort_by(f64::total_cmp);
    Fit {
        auc,
        lo: aucs[7],
        hi: aucs[292],
    }
}

/// [`fit_and_score_with`] for a feature vector stored in the row.
fn fit_and_score(
    feats: &dyn Fn(&Row) -> &Vec<f32>,
    train: &[&Row],
    stop: &[&Row],
    test: &[&Row],
    spec: &ProbeSpec,
    seed: u64,
    pool: &rayon::ThreadPool,
) -> Fit {
    fit_and_score_with(&|r: &Row| feats(r).clone(), false, train, stop, test, spec, seed, pool)
}

/// Trains the MLP on `train` (weighted), stops on the loss of `stop`, returns the AUROC on `test`.
#[allow(clippy::too_many_arguments)]
fn fit_and_score_with(
    feats: &dyn Fn(&Row) -> Vec<f32>,
    linear: bool,
    train: &[&Row],
    stop: &[&Row],
    test: &[&Row],
    spec: &ProbeSpec,
    seed: u64,
    pool: &rayon::ThreadPool,
) -> Fit {
    let raw: Vec<Vec<f32>> = train.iter().map(|r| feats(r)).collect();
    let (mean, std) = standardiser(&raw);
    let norm = |v: Vec<f32>| -> Vec<f32> {
        v.iter()
            .zip(mean.iter().zip(&std))
            .map(|(v, (m, s))| (v - m) / s)
            .collect()
    };
    let xs: Vec<Vec<f32>> = raw.into_iter().map(norm).collect();
    let xstop: Vec<Vec<f32>> = stop.iter().map(|r| norm(feats(r))).collect();
    let xtest: Vec<Vec<f32>> = test.iter().map(|r| norm(feats(r))).collect();
    let d = xs[0].len();
    let mut net = Mlp::new(d, if linear { 1 } else { spec.hidden }, seed, linear);
    let mut adam = FlatAdamState::new(net.p.len());
    let cfg = FlatAdamConfig {
        lr: 2e-3,
        ..FlatAdamConfig::default()
    };
    // Positive-class weight: the rarer class gets the larger one (the BC's per-hazard weights, in spirit).
    let pos_rate = train.iter().map(|r| f32::from(u8::from(r.label))).sum::<f32>() / train.len() as f32;
    let ratio = |rare: f32| ((1.0 - rare) / rare.max(0.05)).clamp(1.0, 5.0);
    let class_w = if pos_rate < 0.5 {
        [1.0, ratio(pos_rate)]
    } else {
        [ratio(1.0 - pos_rate), 1.0]
    };
    let mut order: Vec<usize> = (0..xs.len()).collect();
    let mut rng = SplitMix64::new(seed ^ 0x9E37);
    let loss_on = |net: &Mlp, x: &[Vec<f32>], rows: &[&Row]| -> f64 {
        let mut act = vec![0.0f32; net.h];
        x.iter()
            .zip(rows)
            .map(|(v, r)| f64::from(weighted_bce(net.forward(v, &mut act), r.label, r.weight, class_w).0))
            .sum::<f64>()
            / x.len().max(1) as f64
    };
    let (mut best, mut best_p, mut bad) = (f64::MAX, net.p.clone(), 0);
    for _epoch in 0..spec.epochs {
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        for batch in order.chunks(128) {
            let parts: Vec<Vec<f32>> = pool.install(|| {
                batch
                    .par_chunks(43)
                    .map(|chunk| {
                        let mut g = vec![0.0f32; net.p.len()];
                        let mut act = vec![0.0f32; net.h];
                        for &i in chunk {
                            let z = net.forward(&xs[i], &mut act);
                            let (_, dz) = weighted_bce(z, train[i].label, train[i].weight, class_w);
                            net.backward(&xs[i], &act, dz, &mut g);
                        }
                        g
                    })
                    .collect()
            });
            let mut g = vec![0.0f32; net.p.len()];
            for part in &parts {
                for (a, b) in g.iter_mut().zip(part) {
                    *a += b / batch.len() as f32;
                }
            }
            adam.step(&mut net.p, &g, &cfg);
        }
        let l = loss_on(&net, &xstop, stop);
        if l < best - 1e-4 {
            (best, best_p, bad) = (l, net.p.clone(), 0);
        } else {
            bad += 1;
            if bad >= 5 {
                break;
            }
        }
    }
    net.p = best_p;
    let mut act = vec![0.0f32; net.h];
    let scores: Vec<(f32, bool)> = xtest
        .iter()
        .zip(test)
        .map(|(x, r)| (net.forward(x, &mut act), r.label))
        .collect();
    fit_of(&scores, test)
}

/// The result: AUROC per latch state and probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub latch: bool,
    pub n_train: usize,
    pub n_val: usize,
    pub label_rate_train: f64,
    pub label_rate_val: f64,
    /// `(name, mean AUROC over seeds, per-seed values, mean 95% game-bootstrap interval)` of the probes.
    pub probes: Vec<(String, f64, Vec<f64>, [f64; 2])>,
    /// `(bundle, AUROC, 95% game-bootstrap interval)` of the real flies' hook probability on the validation rows.
    pub flies: Vec<(String, f64, [f64; 2])>,
}

pub fn run_probe(
    cfg: &ExperimentConfig,
    env: &Env,
    spec: &ProbeSpec,
    fly_bundles: &[(String, std::path::PathBuf)],
    state_bundles: &[(String, std::path::PathBuf)],
    threads: usize,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<ProbeResult>, String> {
    let load = |p: &Path| -> Result<FlyLearner, String> {
        let b = ddai_fly::bundle::load_bundle(p).map_err(|e| e.to_string())?;
        FlyLearner::from_bundle(b, &expand_home(&cfg.flyg), FlyTrainConfig::default())
    };
    let first = fly_bundles.first().ok_or("no bundle")?;
    let encoder = load(&first.1)?;
    let flies: Vec<FlyLearner> = fly_bundles.iter().map(|(_, p)| load(p)).collect::<Result<_, _>>()?;
    let mut flies = flies;
    for f in &mut flies {
        f.set_hook_view(ddai_fly::bc::HookView::MaskedForHookHead);
    }
    let states: Vec<FlyBrainTemplate> = state_bundles
        .iter()
        .map(|(_, p)| FlyBrainTemplate::load(p, Some(&expand_home(&cfg.flyg))).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let rows = collect_rows(cfg, env, spec, &encoder, &flies, &states, log)?;
    if let Some(r) = rows.iter().find(|r| !r.hid.is_empty()) {
        log(&format!(
            "dimensions: encoder input {} ({} frames), privileged {}, DN z {} ({} frames), membrane state {}",
            r.a.len() / spec.frames,
            spec.frames,
            r.b.len(),
            r.dn.first().map_or(0, Vec::len) / spec.frames,
            spec.frames,
            r.hid.first().map_or(0, Vec::len)
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for latch in [false, true] {
        let mine: Vec<&Row> = rows.iter().filter(|r| r.latch == latch).collect();
        let val: Vec<&Row> = mine.iter().copied().filter(|r| r.val).collect();
        let mut train: Vec<&Row> = mine.iter().copied().filter(|r| !r.val).collect();
        let mut rng = SplitMix64::new(5);
        while train.len() > spec.max_train_rows {
            let i = (rng.next_u64() % train.len() as u64) as usize;
            train.swap_remove(i);
        }
        // 10% of the training rows stop the training (taken in order: the rows of an episode are contiguous, so this is nearly by episode).
        let cut = train.len() * 9 / 10;
        let (fit, stop) = train.split_at(cut);
        let rate = |r: &[&Row]| r.iter().filter(|x| x.label).count() as f64 / r.len().max(1) as f64;
        log(&format!(
            "latch {latch}: {} train ({} stop) / {} validation rows; label rate {:.3} / {:.3}",
            fit.len(),
            stop.len(),
            val.len(),
            rate(fit),
            rate(&val)
        ));
        let mut probes = Vec::new();
        // The probes on the fly's own network state: its DN z-scores (what the decoder reads) and its full membrane state, per state fly.
        for (k, (name, _)) in state_bundles.iter().enumerate() {
            // The decoder's own view: the 11 hook-group means of the current frame, fitted by a linear model (the 12-parameter hook head,
            // re-fitted optimally on these decisions) and by the MLP; and a linear model and an MLP on all DN z-scores of the current frame (task 8.7:
            // the second is the wide readout `mlp-dn` in a probe's form).
            for (kind, which, linear) in [
                ("hook-group means, linear (the hook head's own features)", 0u8, true),
                ("hook-group means, MLP", 0, false),
                ("all DN z-scores, current frame, linear", 1, true),
                ("all DN z-scores, current frame, MLP", 1, false),
            ] {
                let sel = move |r: &Row| -> Vec<f32> {
                    if which == 0 {
                        r.grp[k].clone()
                    } else {
                        r.dn[k][..r.dn[k].len() / spec.frames].to_vec()
                    }
                };
                let fits: Vec<Fit> = (0..spec.seeds)
                    .map(|s| fit_and_score_with(&sel, linear, fit, stop, &val, spec, 100 + s as u64, &pool))
                    .collect();
                let (vals, ci) = summarise(&fits);
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let label = format!("(e) {name}: {kind}");
                log(&format!(
                    "  {label}: AUROC {mean:.3} [{:.3}; {:.3}] {vals:?}",
                    ci[0], ci[1]
                ));
                probes.push((label, mean, vals, ci));
            }
            for (kind, use_hid) in [("DN z-scores, last frames stacked", false), ("membrane state V", true)] {
                if use_hid && spec.no_membrane {
                    continue;
                }
                let keep = |r: &&Row| !use_hid || !r.hid.is_empty();
                let (fit_k, stop_k, val_k): (Vec<&Row>, Vec<&Row>, Vec<&Row>) = (
                    fit.iter().copied().filter(keep).collect(),
                    stop.iter().copied().filter(keep).collect(),
                    val.iter().copied().filter(keep).collect(),
                );
                let sel = move |r: &Row| -> Vec<f32> { if use_hid { r.hid[k].clone() } else { r.dn[k].clone() } };
                let fits: Vec<Fit> = (0..spec.seeds)
                    .map(|s| fit_and_score_with(&sel, false, &fit_k, &stop_k, &val_k, spec, 100 + s as u64, &pool))
                    .collect();
                let (vals, ci) = summarise(&fits);
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let label = format!("(d) {name}: {kind}");
                log(&format!(
                    "  {label} ({} / {} rows): AUROC {mean:.3} [{:.3}; {:.3}] {vals:?}",
                    ci[0],
                    ci[1],
                    fit_k.len(),
                    val_k.len()
                ));
                probes.push((label, mean, vals, ci));
            }
        }
        type Select = fn(&Row) -> &Vec<f32>;
        let sets: [(&str, Select); 3] = [
            ("(a) fly encoder input, last frames stacked", |r| &r.a),
            ("(b) privileged exact state", |r| &r.b),
            ("(c) = (a) + derived physics features", |r| &r.c),
        ];
        for (name, sel) in sets.into_iter().filter(|_| !spec.state_only) {
            let fits: Vec<Fit> = (0..spec.seeds)
                .map(|s| fit_and_score(&sel, fit, stop, &val, spec, 100 + s as u64, &pool))
                .collect();
            let (vals, ci) = summarise(&fits);
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            log(&format!(
                "  {name}: AUROC {mean:.3} [{:.3}; {:.3}] {vals:?}",
                ci[0], ci[1]
            ));
            probes.push((name.to_string(), mean, vals, ci));
        }
        let flies_auc: Vec<(String, f64, [f64; 2])> = fly_bundles
            .iter()
            .enumerate()
            .map(|(k, (n, _))| {
                let s: Vec<(f32, bool)> = val.iter().map(|r| (r.fly[k], r.label)).collect();
                let f = fit_of(&s, &val);
                (n.clone(), f.auc, [f.lo, f.hi])
            })
            .collect();
        out.push(ProbeResult {
            latch,
            n_train: fit.len(),
            n_val: val.len(),
            label_rate_train: rate(fit),
            label_rate_val: rate(&val),
            probes,
            flies: flies_auc,
        });
    }
    Ok(out)
}
