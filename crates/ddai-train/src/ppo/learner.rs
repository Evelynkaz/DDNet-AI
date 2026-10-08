//! The PPO update of the fly (task 8.5b): recurrent PPO on the batched BPTT, with a privileged critic, a KL anchor to the start
//! policy and a behaviour-cloning term on teacher windows.
//!
//! One [`PpoLearner::update`] takes the episodes of an iteration (played by the current policy, see [`super::rollout`]) and
//!
//! 1. values every acted decision with the critic and runs GAE ([`super::reward::gae`]) per episode; advantages are normalised over
//!    the iteration;
//! 2. cuts every episode into **windows** of `chunk` scored decisions with a burn-in in front. A window starts from the membrane
//!    state the actor stored (R2D2's stored state, [`super::actor`]); the burn-in decisions run with their loss at zero;
//! 3. runs the **reference** (the start policy, frozen) and the **old** policy (the current one, before the update) forward over every
//!    window once: the KL anchor and the PPO ratio need their logits;
//! 4. `epochs` passes over the shuffled windows in mini-batches, one Adam step each. The loss of a scored decision is
//!    `-min(rho A, clip(rho) A) - c_ent H + beta KL(ref || pi)`, with `rho` the ratio of the played action's probability under the
//!    new and the old policy (all heads, the hook head from the masked view), plus the BC loss of a few teacher windows per mini-batch;
//! 5. trains the critic on the returns.
//!
//! The head-gradient algebra is [`ddai_fly::policy`]; the forward/backward is [`ddai_fly::brain_policy_batched`] (the two views of a
//! two-view fly are lanes of one batch). Nothing here depends on the thread count: every sum is taken in a fixed order.

use std::collections::BTreeMap;

use ddai_fly::batched::{BatchedEngine, BatchedPlan};
use ddai_fly::bc::{HeadLogits, HeadThresholds, HookView, head_loss_and_grad};
use ddai_fly::brain_policy_batched::{PolicyForward, PolicyNet, PolicyWindow, policy_backward, policy_forward};
use ddai_fly::bundle::FlyBundle;
use ddai_fly::encoder::OPPONENT_CHANNELS;
use ddai_fly::policy::{self, HeadEntropy, HeadKl, PolicyAction};
use ddai_fly::rng::SplitMix64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::actor::Decision;
use super::config::{BcAux, PpoParams};
use super::critic::{Critic, INPUT_DIM, MapFields, critic_features};
use super::reward::gae;
use super::rollout::Episode;
use crate::learner::{FlyLearner, FlyTrainConfig, Learner};
use crate::seq::{Corpus, Window};
use crate::trainer::{Adam, clip_norm};

/// What survives a kill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoState {
    pub iteration: u64,
    pub theta: Vec<f32>,
    pub adam: Adam,
    pub critic: Critic,
    pub critic_adam: Adam,
    pub beta_kl: f32,
    /// The best selection score so far (training halls only) and where it was.
    pub best_score: f64,
    pub best_iter: u64,
    /// Bundles of earlier policies (the league), oldest first.
    pub snapshots: Vec<String>,
    /// The reverse curriculum's position.
    pub curriculum: super::curriculum::CurriculumState,
    /// Chunks of the DAgger store loaded into the BC corpus, in the order they were added.
    pub dagger_ids: Vec<usize>,
}

fn mix(a: u64, b: u64, c: u64) -> u64 {
    crate::es::noise::pair_seed(a ^ 0x9907_9907_9907_9907, b, c)
}

/// The constants of one decision's loss.
#[derive(Debug, Clone, Copy)]
pub struct PpoCoefs {
    pub clip: f32,
    pub entropy: f32,
    pub kl: f32,
    pub aim_kappa: f32,
    pub temps: policy::Temperatures,
    pub thresholds: HeadThresholds,
    /// `1 / (scored decisions of the mini-batch)`.
    pub scale: f32,
}

/// What one decision's loss was made of (sums over decisions give the mini-batch's numbers).
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionStats {
    pub pg_loss: f64,
    pub ratio: f64,
    pub clipped: bool,
    pub entropy: HeadEntropy,
    pub kl_ref: HeadKl,
    /// Exact `KL(old || new)`.
    pub kl_old: f64,
}

/// The loss of one scored decision and its gradient with respect to the **raw** head logits of the new policy (a threshold shift is a
/// constant, so the gradient of the shifted logits is the same). `old` and `reference` are raw logits too.
pub fn ppo_decision(
    new: &HeadLogits,
    old: &HeadLogits,
    reference: &HeadLogits,
    a: &PolicyAction,
    adv: f32,
    k: &PpoCoefs,
) -> (HeadLogits, DecisionStats) {
    let (s, o, r) = (
        policy::policy_logits(new, &k.thresholds, &k.temps),
        policy::policy_logits(old, &k.thresholds, &k.temps),
        policy::policy_logits(reference, &k.thresholds, &k.temps),
    );
    let lp_new = policy::log_prob(&s, a, k.aim_kappa).total();
    let lp_old = policy::log_prob(&o, a, k.aim_kappa).total();
    let ratio = (lp_new - lp_old).clamp(-20.0, 20.0).exp();
    let clipped_ratio = ratio.clamp(1.0 - k.clip, 1.0 + k.clip);
    // `min(rho A, clip(rho) A)` is flat exactly where the clipped term is the smaller one.
    let active = (adv >= 0.0 && ratio <= 1.0 + k.clip) || (adv < 0.0 && ratio >= 1.0 - k.clip);
    let mut d = HeadLogits::default();
    if active {
        policy::axpy(&mut d, -adv * ratio, &policy::log_prob_grad(&s, a, k.aim_kappa));
    }
    let ent = policy::entropy(&s);
    policy::axpy(&mut d, -k.entropy, &policy::entropy_grad(&s));
    let aim_on = policy::aim_active(a);
    let kl_ref = policy::kl(&r, &s, k.aim_kappa, aim_on);
    policy::axpy(&mut d, k.kl, &policy::kl_grad(&r, &s, k.aim_kappa, aim_on));
    let kl_old = policy::kl(&o, &s, k.aim_kappa, aim_on).total();
    policy::unscale_temperature(&mut d, &k.temps);
    policy::scale(&mut d, k.scale);
    (
        d,
        DecisionStats {
            pg_loss: -f64::from((adv * ratio).min(adv * clipped_ratio)),
            ratio: f64::from(ratio),
            clipped: !active,
            entropy: ent,
            kl_ref,
            kl_old: f64::from(kl_old),
        },
    )
}

/// The scalar loss `ppo_decision` differentiates (the stats' `pg_loss` already is the policy term): for the finite-difference test.
pub fn ppo_decision_loss(
    new: &HeadLogits,
    old: &HeadLogits,
    reference: &HeadLogits,
    a: &PolicyAction,
    adv: f32,
    k: &PpoCoefs,
) -> f64 {
    let (_, st) = ppo_decision(new, old, reference, a, adv, k);
    f64::from(k.scale)
        * (st.pg_loss - f64::from(k.entropy) * f64::from(st.entropy.total())
            + f64::from(k.kl) * f64::from(st.kl_ref.total()))
}

/// One BPTT window of an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WinPlan {
    pub ep: usize,
    /// First decision of the window (the burn-in starts here), first scored decision, one past the last.
    pub b: usize,
    pub i0: usize,
    pub i1: usize,
}

/// Cuts the episodes into windows on the absolute grid `a = tick / decide_every`, `chunk` scored decisions each, `burn_in` before the
/// first scored one (more for the first window of an episode that starts in the middle of a chunk, down to the first recorded decision).
pub fn plan_windows(episodes: &[Episode], chunk: usize, burn_in: usize, de: i32) -> Vec<WinPlan> {
    let mut plans = Vec::new();
    for (e, ep) in episodes.iter().enumerate() {
        let n = ep.decisions.len();
        let first = ep.first_acted();
        let abs = |i: usize| (ep.decisions[i].obs.tick / de.max(1)) as usize;
        let mut i = first;
        while i < n {
            let c = abs(i) / chunk;
            let mut j = i;
            while j < n && abs(j) / chunk == c {
                j += 1;
            }
            // The snapshot at absolute decision `c * chunk - burn_in` (or the first recorded decision).
            let want = (c * chunk).saturating_sub(burn_in);
            let back = abs(i).saturating_sub(want);
            let b = i.saturating_sub(back);
            plans.push(WinPlan { ep: e, b, i0: i, i1: j });
            i = j;
        }
    }
    plans
}

/// The numbers of one update.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateStats {
    pub decisions: usize,
    pub windows: usize,
    pub minibatches: usize,
    pub epochs_run: usize,
    pub pg_loss: f64,
    pub ratio_mean: f64,
    pub clip_frac: f64,
    pub entropy: [f64; 4],
    pub kl_ref: [f64; 5],
    pub kl_old: f64,
    pub bc_loss: f64,
    pub grad_norm: f64,
    pub grad_norm_new_channels: f64,
    pub value_loss: f64,
    pub explained_variance: f64,
    pub return_mean: f64,
    pub adv_std: f64,
    pub policy_updated: bool,
    pub weight_norm_new_g: f64,
    pub weight_norm_new_c: f64,
    pub max_abs_new: f64,
    pub secs_forward: f64,
    pub secs_train: f64,
    pub secs_critic: f64,
}

/// The teacher windows of the BC term.
pub struct BcData {
    pub corpus: Corpus,
    pub cfg: BcAux,
}

pub struct PpoLearner {
    pub params: PpoParams,
    cur: FlyLearner,
    reference: FlyLearner,
    engine: BatchedEngine,
    engine_ref: BatchedEngine,
    pub theta: Vec<f32>,
    pub adam: Adam,
    lrs: Vec<f32>,
    pub critic: Critic,
    pub critic_adam: Adam,
    pub beta_kl: f32,
    two_view: bool,
    thresholds: HeadThresholds,
    aim_kappa: f32,
    temps: policy::Temperatures,
    /// Flat indices of the encoder weights (`g`, `c`) of the opponent-state channels.
    new_g: Vec<usize>,
    new_c: Vec<usize>,
    rest: Vec<f32>,
    cap: Option<usize>,
    seed: u64,
    bc: Option<BcData>,
}

impl PpoLearner {
    pub fn new(
        params: PpoParams,
        base: &FlyBundle,
        flyg_path: &std::path::Path,
        aim_kappa: f32,
        temps: policy::Temperatures,
        seed: u64,
        bc: Option<BcData>,
    ) -> Result<PpoLearner, String> {
        if !base.neuron_model.is_rate() {
            return Err(format!(
                "PPO (the batched actor and its BPTT) supports the rate model only; this checkpoint has the {} neuron model (task 8.8)",
                base.neuron_model.label()
            ));
        }
        let train_cfg = FlyTrainConfig {
            l2_a: params.l2_a,
            ..FlyTrainConfig::default()
        };
        let cur = FlyLearner::from_bundle(base.clone(), flyg_path, train_cfg.clone())?;
        let reference = FlyLearner::from_bundle(base.clone(), flyg_path, train_cfg)?;
        let theta = cur.params();
        let mut lrs: Vec<f32> = cur.base_lrs().iter().map(|l| l * params.lr_scale).collect();
        // The opponent-state channels' encoder weights: where they live in the flat vector, and a larger step for them (they start at zero).
        let layout = cur.layout();
        let names: Vec<&str> = OPPONENT_CHANNELS.iter().map(|c| c.name()).collect();
        let (g0, c0) = (
            layout.a + layout.b + layout.theta,
            layout.a + layout.b + layout.theta + layout.g,
        );
        let new_ids: Vec<usize> = cur
            .encoder()
            .assignments()
            .iter()
            .filter(|a| names.contains(&a.channel))
            .map(|a| a.param_id as usize)
            .collect();
        let new_g: Vec<usize> = new_ids.iter().map(|&i| g0 + i).collect();
        let new_c: Vec<usize> = new_ids.iter().map(|&i| c0 + i).collect();
        for &i in new_g.iter().chain(&new_c) {
            lrs[i] *= params.lr_new_channel_mult;
        }
        let n_params = theta.len();
        let plan = BatchedPlan::new(cur.model());
        let engine = BatchedEngine::with_plan(plan.clone());
        let engine_ref = BatchedEngine::with_plan(plan);
        let critic = Critic::new(INPUT_DIM, params.critic_hidden, params.critic_hidden, seed);
        let critic_adam = Adam::new(critic.params.len());
        let rest = rest_state(&cur);
        Ok(PpoLearner {
            cap: (params.memory_cap_mb > 0).then(|| params.memory_cap_mb.saturating_mul(1 << 20)),
            beta_kl: params.kl_coef,
            two_view: base.hook_view == HookView::MaskedForHookHead,
            thresholds: base.thresholds,
            aim_kappa,
            temps,
            new_g,
            new_c,
            rest,
            seed,
            bc,
            params,
            cur,
            reference,
            engine,
            engine_ref,
            theta,
            adam: Adam::new(n_params),
            lrs,
            critic,
            critic_adam,
        })
    }

    pub fn from_state(&mut self, s: &PpoState) -> Result<(), String> {
        if s.theta.len() != self.theta.len() {
            return Err("state.bin does not match the bundle's parameter count".into());
        }
        if s.critic.n_in != self.critic.n_in || s.critic.params.len() != self.critic.params.len() {
            return Err("state.bin does not match the critic's shape".into());
        }
        self.theta.clone_from(&s.theta);
        self.cur.set_params(&self.theta)?;
        self.adam = s.adam.clone();
        self.critic = s.critic.clone();
        self.critic_adam = s.critic_adam.clone();
        self.beta_kl = s.beta_kl;
        Ok(())
    }

    /// Adds labelled sequences to the BC corpus (the DAgger rounds); a no-op when the run has no BC term.
    pub fn append_bc(&mut self, seqs: Vec<crate::seq::Seq>) {
        if let Some(bc) = &mut self.bc {
            bc.corpus.append(seqs);
        }
    }

    pub fn has_bc(&self) -> bool {
        self.bc.is_some()
    }

    pub fn to_bundle(&self, meta: ddai_fly::bundle::BundleMeta) -> FlyBundle {
        self.cur.to_bundle(meta)
    }

    pub fn two_view(&self) -> bool {
        self.two_view
    }

    pub fn thresholds(&self) -> HeadThresholds {
        self.thresholds
    }

    /// Norms of the opponent-state channels' encoder weights: `(|g|, |c|, max |w|)`.
    pub fn new_channel_norms(&self) -> (f64, f64, f64) {
        let norm = |idx: &[usize]| {
            idx.iter()
                .map(|&i| f64::from(self.theta[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let max = self
            .new_g
            .iter()
            .chain(&self.new_c)
            .map(|&i| f64::from(self.theta[i]).abs())
            .fold(0.0, f64::max);
        (norm(&self.new_g), norm(&self.new_c), max)
    }

    pub fn num_new_channel_params(&self) -> usize {
        self.new_g.len() + self.new_c.len()
    }

    fn windows_of(&self, episodes: &[Episode], plans: &[WinPlan]) -> Vec<PolicyWindow> {
        plans
            .iter()
            .map(|p| {
                let ep = &episodes[p.ep];
                let snap = ep.decisions[p.b]
                    .snap
                    .as_ref()
                    .expect("a window starts at a decision with a stored state");
                PolicyWindow {
                    v_init: snap.0.clone(),
                    v_init_masked: snap.1.clone(),
                    observations: ep.decisions[p.b..p.i1].iter().map(|d| d.obs.clone()).collect(),
                }
            })
            .collect()
    }

    /// One PPO update from the episodes of an iteration. `de` is the decisions' tick spacing.
    pub fn update(
        &mut self,
        episodes: &[Episode],
        fields: &BTreeMap<String, MapFields>,
        iteration: u64,
        de: i32,
        pool: &rayon::ThreadPool,
    ) -> Result<UpdateStats, String> {
        let p = self.params.clone();
        let mut stats = UpdateStats::default();
        let warmup = iteration < p.critic_warmup_iters;
        // 1. Critic features, values, GAE.
        struct EpData {
            feats: Vec<Vec<f32>>,
            adv: Vec<f32>,
            ret: Vec<f32>,
            values: Vec<f32>,
        }
        let critic = &self.critic;
        let mut data: Vec<EpData> = pool.install(|| {
            episodes
                .par_iter()
                .map(|ep| {
                    let first = ep.first_acted();
                    let f = &fields[&ep.arena];
                    let feats: Vec<Vec<f32>> = ep.decisions[first..]
                        .iter()
                        .map(|d| critic_features(&d.obs, f, &ep.clock))
                        .collect();
                    let values: Vec<f32> = feats.iter().map(|x| critic.value(x)).collect();
                    let (adv, ret) = gae(&ep.rewards[first..], &values, p.gamma, p.lambda);
                    EpData {
                        feats,
                        adv,
                        ret,
                        values,
                    }
                })
                .collect()
        });
        let all_adv: Vec<f64> = data.iter().flat_map(|d| d.adv.iter().map(|&a| f64::from(a))).collect();
        let n_dec = all_adv.len();
        stats.decisions = n_dec;
        if n_dec == 0 {
            return Ok(stats);
        }
        let (mean, var) = {
            let m = all_adv.iter().sum::<f64>() / n_dec as f64;
            (m, all_adv.iter().map(|a| (a - m).powi(2)).sum::<f64>() / n_dec as f64)
        };
        let std = var.sqrt().max(1e-6);
        stats.adv_std = std;
        for d in &mut data {
            for a in &mut d.adv {
                *a = ((f64::from(*a) - mean) / std) as f32;
            }
        }
        let (ret_all, val_all): (Vec<f64>, Vec<f64>) = data
            .iter()
            .flat_map(|d| d.ret.iter().zip(&d.values).map(|(&r, &v)| (f64::from(r), f64::from(v))))
            .unzip();
        stats.return_mean = ret_all.iter().sum::<f64>() / n_dec as f64;
        {
            let m = stats.return_mean;
            let var_r = ret_all.iter().map(|r| (r - m).powi(2)).sum::<f64>() / n_dec as f64;
            let var_e = ret_all.iter().zip(&val_all).map(|(r, v)| (r - v).powi(2)).sum::<f64>() / n_dec as f64;
            stats.explained_variance = if var_r > 1e-9 { 1.0 - var_e / var_r } else { 0.0 };
        }

        // 2. Windows and their logits under the reference and the old policy.
        let plans = plan_windows(episodes, p.chunk, p.burn_in, de);
        stats.windows = plans.len();
        if !warmup {
            let windows = self.windows_of(episodes, &plans);
            let t0 = std::time::Instant::now();
            let (two_view, cap, chunk) = (self.two_view, self.cap, p.forward_chunk);
            let need_ref = self.beta_kl > 0.0 || p.kl_target > 0.0;
            let (cur, refn) = (&self.cur, &self.reference);
            let (eng, eng_ref) = (&mut self.engine, &mut self.engine_ref);
            let (old, reference) = pool.install(|| -> Result<_, String> {
                let old = forward_all(cur.policy_net(), eng, &windows, two_view, cap, chunk)?;
                let reference = if need_ref {
                    forward_all(refn.policy_net(), eng_ref, &windows, two_view, cap, chunk)?
                } else {
                    old.clone()
                };
                Ok((old, reference))
            })?;
            stats.secs_forward = t0.elapsed().as_secs_f64();

            // 3. The epochs.
            let t0 = std::time::Instant::now();
            let advs: Vec<Vec<f32>> = data.iter().map(|d| d.adv.clone()).collect();
            self.train_epochs(
                episodes, &advs, &plans, &windows, &old, &reference, iteration, pool, &mut stats,
            )?;
            stats.secs_train = t0.elapsed().as_secs_f64();
            stats.policy_updated = true;
        }
        // 4. The critic.
        let t0 = std::time::Instant::now();
        let epochs = if warmup { p.critic_epochs * 4 } else { p.critic_epochs };
        let xs: Vec<&[f32]> = data.iter().flat_map(|d| d.feats.iter().map(Vec::as_slice)).collect();
        let rets: Vec<f32> = data.iter().flat_map(|d| d.ret.iter().copied()).collect();
        let mut order: Vec<usize> = (0..xs.len()).collect();
        let mut loss_sum = 0.0f64;
        let mut loss_n = 0usize;
        for e in 0..epochs {
            let mut rng = SplitMix64::new(mix(self.seed, iteration, 0xC000 + e as u64));
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            for mb in order.chunks(p.critic_batch) {
                let bx: Vec<&[f32]> = mb.iter().map(|&i| xs[i]).collect();
                let bt: Vec<f32> = mb.iter().map(|&i| rets[i]).collect();
                let (loss, mut g) = pool.install(|| self.critic.mse_grad(&bx, &bt));
                clip_norm(&mut g, p.critic_grad_clip);
                let lrs = vec![p.critic_lr; g.len()];
                let mut params = self.critic.params.clone();
                if self.critic_adam.step(&mut params, &g, &lrs, 1.0) {
                    self.critic.params = params;
                }
                loss_sum += f64::from(loss) * mb.len() as f64;
                loss_n += mb.len();
            }
        }
        stats.value_loss = loss_sum / loss_n.max(1) as f64;
        stats.secs_critic = t0.elapsed().as_secs_f64();
        let (g, c, m) = self.new_channel_norms();
        (stats.weight_norm_new_g, stats.weight_norm_new_c, stats.max_abs_new) = (g, c, m);
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    fn train_epochs(
        &mut self,
        episodes: &[Episode],
        advs: &[Vec<f32>],
        plans: &[WinPlan],
        windows: &[PolicyWindow],
        old: &[Vec<HeadLogits>],
        reference: &[Vec<HeadLogits>],
        iteration: u64,
        pool: &rayon::ThreadPool,
        stats: &mut UpdateStats,
    ) -> Result<(), String> {
        let p = self.params.clone();
        let nw = plans.len();
        let mut order: Vec<usize> = (0..nw).collect();
        let lrs_mult = 1.0f32;
        let (mut s_pg, mut s_ratio, mut s_clip, mut s_n) = (0.0f64, 0.0f64, 0usize, 0usize);
        let mut s_ent = [0.0f64; 4];
        let mut s_kl = [0.0f64; 5];
        let mut s_klold = 0.0f64;
        let (mut s_bc, mut s_bcw) = (0.0f64, 0.0f64);
        let (mut s_gn, mut s_gnn, mut s_mb) = (0.0f64, 0.0f64, 0usize);
        for epoch in 0..p.epochs {
            let mut rng = SplitMix64::new(mix(self.seed, iteration, epoch as u64));
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            let (mut e_klold, mut e_n) = (0.0f64, 0usize);
            for (mbi, mb) in order.chunks(p.minibatch_windows).enumerate() {
                // The mini-batch: PPO windows, then teacher windows.
                let mut pw: Vec<PolicyWindow> = mb.iter().map(|&w| windows[w].clone()).collect();
                let n_ppo = pw.len();
                let mut bc_targets: Vec<Vec<ddai_fly::bc::StepTargets>> = Vec::new();
                if let (Some(bc), true) = (&self.bc, p.bc_coef > 0.0 && p.bc_windows > 0)
                    && !self.bc.as_ref().is_some_and(|b| b.corpus.is_empty())
                {
                    let mut brng = SplitMix64::new(mix(
                        self.seed ^ bc.cfg.seed,
                        iteration,
                        ((epoch as u64) << 20) | mbi as u64,
                    ));
                    for _ in 0..p.bc_windows {
                        let flip = brng.next_f32_unit() < 0.5;
                        let w: Window = bc.corpus.sample_window(&mut brng, p.chunk + p.burn_in, p.burn_in, flip);
                        pw.push(PolicyWindow {
                            v_init: self.rest.clone(),
                            v_init_masked: self.rest.clone(),
                            observations: w.observations,
                        });
                        bc_targets.push(w.targets);
                    }
                }
                let scored: usize = mb.iter().map(|&w| plans[w].i1 - plans[w].i0).sum();
                let coefs = PpoCoefs {
                    clip: p.clip,
                    entropy: p.entropy_coef,
                    kl: self.beta_kl,
                    aim_kappa: self.aim_kappa,
                    temps: self.temps,
                    thresholds: self.thresholds,
                    scale: 1.0 / scored.max(1) as f32,
                };
                let bc_weight: f32 = bc_targets.iter().flat_map(|t| t.iter().map(|s| s.weight)).sum();
                let bc_scale = if bc_weight > 0.0 { p.bc_coef / bc_weight } else { 0.0 };
                let fwd: PolicyForward = pool
                    .install(|| policy_forward(self.cur.policy_net(), &mut self.engine, &pw, self.two_view, self.cap))
                    .map_err(|e| e.to_string())?;
                // The gradient of the loss on the logits, window by window.
                struct Out {
                    d: Vec<HeadLogits>,
                    pg: f64,
                    ratio: f64,
                    clipped: usize,
                    n: usize,
                    ent: [f64; 4],
                    kl: [f64; 5],
                    klold: f64,
                    bc: f64,
                    bcw: f64,
                }
                let bc_cfg = self.bc.as_ref().map(|b| b.cfg.loss);
                let outs: Vec<Out> = pool.install(|| {
                    (0..pw.len())
                        .into_par_iter()
                        .map(|j| {
                            let logits = &fwd.logits[j];
                            let mut o = Out {
                                d: vec![HeadLogits::default(); logits.len()],
                                pg: 0.0,
                                ratio: 0.0,
                                clipped: 0,
                                n: 0,
                                ent: [0.0; 4],
                                kl: [0.0; 5],
                                klold: 0.0,
                                bc: 0.0,
                                bcw: 0.0,
                            };
                            if j < n_ppo {
                                let w = mb[j];
                                let plan = &plans[w];
                                let advs = &advs[plan.ep];
                                let ep = &episodes[plan.ep];
                                let first = ep.first_acted();
                                for i in plan.i0..plan.i1 {
                                    let t = i - plan.b;
                                    let dec: &Decision = &ep.decisions[i];
                                    let adv = advs[i - first];
                                    let (d, st) = ppo_decision(
                                        &logits[t],
                                        &old[w][t],
                                        &reference[w][t],
                                        &dec.action,
                                        adv,
                                        &coefs,
                                    );
                                    o.d[t] = d;
                                    o.pg += st.pg_loss;
                                    o.ratio += st.ratio;
                                    o.clipped += usize::from(st.clipped);
                                    o.n += 1;
                                    o.ent[0] += f64::from(st.entropy.dir);
                                    o.ent[1] += f64::from(st.entropy.jump);
                                    o.ent[2] += f64::from(st.entropy.hook);
                                    o.ent[3] += f64::from(st.entropy.fire);
                                    o.kl[0] += f64::from(st.kl_ref.dir);
                                    o.kl[1] += f64::from(st.kl_ref.jump);
                                    o.kl[2] += f64::from(st.kl_ref.hook);
                                    o.kl[3] += f64::from(st.kl_ref.fire);
                                    o.kl[4] += f64::from(st.kl_ref.aim);
                                    o.klold += st.kl_old;
                                }
                            } else if let Some(cfg) = &bc_cfg {
                                let targets = &bc_targets[j - n_ppo];
                                for (t, tg) in targets.iter().enumerate() {
                                    if tg.weight <= 0.0 {
                                        continue;
                                    }
                                    let (l, mut d) = head_loss_and_grad(&logits[t], tg, cfg);
                                    policy::scale(&mut d, bc_scale);
                                    o.d[t] = d;
                                    o.bc += f64::from(l.total);
                                    o.bcw += f64::from(tg.weight);
                                }
                            }
                            o
                        })
                        .collect()
                });
                let dls: Vec<Vec<HeadLogits>> = outs.iter().map(|o| o.d.clone()).collect();
                let bwd = pool.install(|| policy_backward(self.cur.policy_net(), &mut self.engine, &fwd, &dls));
                // The flat gradient, in window order.
                let mut grad = vec![0.0f32; self.theta.len()];
                let zero_enc = self.cur.encoder().zero_grads();
                let no_dec = self.cur.policy_net().decoder.zeros_gradients();
                self.cur.add_parts_grads(&bwd.fly, &zero_enc, &no_dec, &mut grad);
                let no_fly = ddai_fly::optim::ParamGradients::zeros_like(self.cur.policy_net().model.params());
                for (e, d) in bwd.encoder.iter().zip(&bwd.decoder) {
                    self.cur.add_parts_grads(&no_fly, e, d, &mut grad);
                }
                let _ = self.cur.regularizer_grad(&mut grad);
                let gn_new = self
                    .new_g
                    .iter()
                    .chain(&self.new_c)
                    .map(|&i| f64::from(grad[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let gn = f64::from(clip_norm(&mut grad, p.grad_clip));
                let mut theta = self.theta.clone();
                if self.adam.step(&mut theta, &grad, &self.lrs, lrs_mult) {
                    self.theta = theta;
                    self.cur.set_params(&self.theta)?;
                }
                for o in &outs {
                    s_pg += o.pg;
                    s_ratio += o.ratio;
                    s_clip += o.clipped;
                    s_n += o.n;
                    for (a, b) in s_ent.iter_mut().zip(&o.ent) {
                        *a += b;
                    }
                    for (a, b) in s_kl.iter_mut().zip(&o.kl) {
                        *a += b;
                    }
                    s_klold += o.klold;
                    e_klold += o.klold;
                    e_n += o.n;
                    s_bc += o.bc;
                    s_bcw += o.bcw;
                }
                s_gn += gn;
                s_gnn += gn_new;
                s_mb += 1;
            }
            stats.epochs_run = epoch + 1;
            if p.target_kl > 0.0 && e_n > 0 && e_klold / e_n as f64 > 1.5 * f64::from(p.target_kl) {
                break;
            }
        }
        let n = s_n.max(1) as f64;
        stats.minibatches = s_mb;
        stats.pg_loss = s_pg / n;
        stats.ratio_mean = s_ratio / n;
        stats.clip_frac = s_clip as f64 / n;
        for (a, b) in stats.entropy.iter_mut().zip(&s_ent) {
            *a = b / n;
        }
        for (a, b) in stats.kl_ref.iter_mut().zip(&s_kl) {
            *a = b / n;
        }
        stats.kl_old = s_klold / n;
        stats.bc_loss = if s_bcw > 0.0 { s_bc / s_bcw } else { 0.0 };
        stats.grad_norm = s_gn / s_mb.max(1) as f64;
        stats.grad_norm_new_channels = s_gnn / s_mb.max(1) as f64;
        // The adaptive KL coefficient.
        if p.kl_target > 0.0 {
            let kl: f64 = stats.kl_ref.iter().sum();
            if kl > 1.5 * f64::from(p.kl_target) {
                self.beta_kl = (self.beta_kl * 1.5).min(p.kl_coef_max);
            } else if kl < f64::from(p.kl_target) / 1.5 {
                self.beta_kl = (self.beta_kl / 1.5).max(p.kl_coef_min);
            }
        }
        Ok(())
    }
}

fn rest_state(l: &FlyLearner) -> Vec<f32> {
    l.rest_state().to_vec()
}

/// The logits of `net` over every window (forward only), in chunks.
fn forward_all(
    net: PolicyNet<'_>,
    engine: &mut BatchedEngine,
    windows: &[PolicyWindow],
    two_view: bool,
    cap: Option<usize>,
    chunk: usize,
) -> Result<Vec<Vec<HeadLogits>>, String> {
    let mut out = Vec::with_capacity(windows.len());
    for ws in windows.chunks(chunk.max(1)) {
        let fwd = policy_forward(net, engine, ws, two_view, cap).map_err(|e| e.to_string())?;
        out.extend(fwd.logits);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_fly::bc::HeadLogits;

    fn logits(seed: u64) -> HeadLogits {
        let mut r = SplitMix64::new(seed);
        let mut g = || (r.next_f32_unit() - 0.5) * 3.0;
        HeadLogits {
            dir: [g(), g(), g()],
            jump: g(),
            hook: g(),
            fire: g(),
            aim_c: g() + 2.0,
            aim_s: g(),
        }
    }

    #[test]
    fn the_decision_gradient_matches_finite_differences() {
        let th = HeadThresholds {
            jump: 0.55,
            hook: 0.63,
            fire: 0.5,
        };
        let k = PpoCoefs {
            clip: 0.2,
            entropy: 0.01,
            kl: 0.7,
            aim_kappa: 8.0,
            temps: policy::Temperatures::uniform(0.4),
            thresholds: th,
            scale: 0.25,
        };
        let a = PolicyAction {
            dir: 2,
            jump: true,
            hook: true,
            fire: false,
            aim: 0.9,
            aim_counts: true,
        };
        for seed in 0..5u64 {
            let (old, reference) = (logits(seed + 10), logits(seed + 20));
            // The new policy close to the old one: the ratio is inside the clip range, where the objective is differentiable.
            let new = {
                let mut n = old;
                n.dir[0] += 0.03;
                n.jump -= 0.02;
                n.aim_s += 0.02;
                n
            };
            for adv in [1.3f32, -0.8] {
                let (d, st) = ppo_decision(&new, &old, &reference, &a, adv, &k);
                assert!(!st.clipped, "the test wants the unclipped branch");
                let eps = 1e-3f32;
                type Edit = fn(&mut HeadLogits) -> &mut f32;
                let fields: [(&str, Edit, f32); 8] = [
                    ("dir0", |l| &mut l.dir[0], d.dir[0]),
                    ("dir1", |l| &mut l.dir[1], d.dir[1]),
                    ("dir2", |l| &mut l.dir[2], d.dir[2]),
                    ("jump", |l| &mut l.jump, d.jump),
                    ("hook", |l| &mut l.hook, d.hook),
                    ("fire", |l| &mut l.fire, d.fire),
                    ("aim_c", |l| &mut l.aim_c, d.aim_c),
                    ("aim_s", |l| &mut l.aim_s, d.aim_s),
                ];
                for (name, edit, analytic) in fields {
                    let (mut x, mut y) = (new, new);
                    *edit(&mut x) += eps;
                    *edit(&mut y) -= eps;
                    let numeric = ((ppo_decision_loss(&x, &old, &reference, &a, adv, &k)
                        - ppo_decision_loss(&y, &old, &reference, &a, adv, &k))
                        / (2.0 * f64::from(eps))) as f32;
                    assert!(
                        (numeric - analytic).abs() < 3e-3 * (1.0 + numeric.abs()),
                        "seed {seed} adv {adv} {name}: analytic {analytic} vs numeric {numeric}"
                    );
                }
            }
        }
    }

    #[test]
    fn clipping_stops_the_gradient_where_ppo_says_so() {
        let k = PpoCoefs {
            clip: 0.2,
            entropy: 0.0,
            kl: 0.0,
            aim_kappa: 8.0,
            temps: policy::Temperatures {
                dir: 0.5,
                jump: 0.4,
                hook: 0.2,
                fire: 0.5,
            },
            thresholds: HeadThresholds::default(),
            scale: 1.0,
        };
        let a = PolicyAction {
            dir: 0,
            jump: false,
            hook: false,
            fire: false,
            aim: 0.0,
            aim_counts: false,
        };
        let old = logits(1);
        // Push the probability of the action well above 1 + clip: a positive advantage gives no gradient, a negative one does.
        let mut new = old;
        new.dir[0] += 1.5;
        let (d_pos, st) = ppo_decision(&new, &old, &old, &a, 1.0, &k);
        assert!(st.ratio > 1.2);
        assert!(d_pos.dir.iter().all(|&x| x == 0.0) && d_pos.jump == 0.0, "{d_pos:?}");
        let (d_neg, _) = ppo_decision(&new, &old, &old, &a, -1.0, &k);
        assert!(d_neg.dir.iter().any(|&x| x != 0.0));
        // And mirrored below 1 - clip.
        let mut low = old;
        low.dir[0] -= 1.5;
        let (d_neg, st) = ppo_decision(&low, &old, &old, &a, -1.0, &k);
        assert!(st.ratio < 0.8);
        assert!(d_neg.dir.iter().all(|&x| x == 0.0));
        let (d_pos, _) = ppo_decision(&low, &old, &old, &a, 1.0, &k);
        assert!(d_pos.dir.iter().any(|&x| x != 0.0));
        // At ratio 1 the gradient is the policy gradient `-A grad logp`.
        let (d, st) = ppo_decision(&old, &old, &old, &a, 2.0, &k);
        assert!((st.ratio - 1.0).abs() < 1e-6 && st.kl_old.abs() < 1e-7);
        let g = policy::log_prob_grad(&policy::policy_logits(&old, &k.thresholds, &k.temps), &a, 8.0);
        for i in 0..3 {
            assert!((d.dir[i] + 2.0 * 2.0 * g.dir[i]).abs() < 1e-5);
        }
    }
}
