//! The reward of a PPO episode, decision by decision (task 8.5b), and GAE.
//!
//! One decision is two world ticks. For the decisions of an episode (the acted ones; the burn-in decisions before the handover carry
//! nothing) the reward is the sum of
//!
//! * **potential-based shaping** `F_i = gamma * Phi(s_{i+1}) - Phi(s_i)` (Ng, Harada & Russell 1999) with the hold margin of the opponent
//!   as the potential ([`hold_potential`]) and `Phi = 0` after the last decision: the episode ends there (the window is over, or the
//!   fly is out), and a terminal state has potential zero. The discounted sum telescopes to `-Phi(s_first)`
//!   ([`shaping_terms`], tested on arbitrary sequences and on real episodes), so shaping adds a constant to every return of an episode
//!   and cannot change which policy is best, only how early the credit arrives. It is scaled by [`PpoReward::shaping`]; the scale
//!   multiplies the same telescoping sum, so it keeps the guarantee;
//! * **the terminal outcome** on the last decision, the held-block reward of 8.5a (`heldblock::RewardConfig`): a held block `+held`, the
//!   fly out in the window `-self_freeze`, otherwise `0`; for full games the credited first freeze pays `+credited` at the decision that
//!   froze the opponent and the rest (`held - credited`, or `-self_freeze - credited` when the fly is out afterwards) at the end, so the
//!   total is the 8.5a return of the game; a lost game `-self_freeze`, a draw or a timeout `-draw_or_timeout`;
//! * **freeing the victim ourselves** `-free_victim`, once per episode: the victim's freeze timer, which stays full while it lies on a
//!   freeze tile, starts to run down while the fly holds it with its hook (it was pulled off the freeze: T8 of the research).

use ddai_brain::HOOK_IDLE;
use ddai_env::stats::GameResult;
use serde::{Deserialize, Serialize};

use super::actor::Decision;
use super::critic::{FIELD_CAP, FREEZE_TICKS, MapFields, hold_potential};
use crate::heldblock::EpisodeOutcome;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpisodeKind {
    /// Starts at a bank handover: the opponent has just been frozen with credit.
    Post,
    /// A full game from the spawn; ends `window` ticks after the first freeze (or at its end).
    Game,
}

/// Weights of the per-decision reward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PpoReward {
    pub held: f32,
    pub self_freeze: f32,
    pub credited: f32,
    pub draw_or_timeout: f32,
    /// Scale of the potential-based shaping (`0` = none).
    pub shaping: f32,
    /// Penalty for pulling the frozen victim off the freeze with the hook (once per episode).
    pub free_victim: f32,
    /// Weight of the fly's own hazard in the potential: `-own_hazard * (1 - d / 6)` for a fly `d < 6` tiles (BFS) from the nearest freeze
    /// cell, `0` beyond (and `0` weight = none). The fly's own freeze is its main failure after the block (48.6% of the validation starts,
    /// E-022); this makes the approach to a freeze cost something before the freeze itself does. A potential like any other part of `Phi`:
    /// it telescopes.
    pub own_hazard: f32,
}

impl Default for PpoReward {
    fn default() -> Self {
        PpoReward {
            held: 1.0,
            self_freeze: 1.0,
            credited: 0.3,
            draw_or_timeout: 0.5,
            shaping: 0.5,
            free_victim: 0.25,
            own_hazard: 0.15,
        }
    }
}

/// The potential of a state: the hold margin of the opponent ([`hold_potential`]) minus the own-hazard term (module docs).
pub fn potential(obs: &ddai_brain::Observation, fields: &MapFields, own_hazard: f32) -> f32 {
    let mut p = hold_potential(obs, fields);
    if own_hazard > 0.0 {
        let d = f32::from(
            fields
                .freeze_distance(obs.self_state.pos.x, obs.self_state.pos.y)
                .min(FIELD_CAP),
        )
        .min(6.0);
        p -= own_hazard * (1.0 - d / 6.0);
    }
    p
}

/// `F_i = gamma * phi[i + 1] - phi[i]`, with `phi[n] = 0` after the last state (terminal): `n` terms for `n` potentials.
pub fn shaping_terms(phi: &[f32], gamma: f32) -> Vec<f32> {
    (0..phi.len())
        .map(|i| gamma * phi.get(i + 1).copied().unwrap_or(0.0) - phi[i])
        .collect()
}

/// What an episode's reward is made of (for the logs).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct RewardParts {
    pub shaping: f32,
    pub terminal: f32,
    pub credited: f32,
    pub free_victim: f32,
    /// `Phi` of the first acted decision.
    pub phi_start: f32,
    /// The victim was pulled off the freeze by the fly.
    pub freed_victim: bool,
}

/// The per-decision rewards of an episode (`decisions` already trimmed at the end of the episode) and what they are made of. The burn-in
/// decisions get `0`. `de` is the decisions' tick spacing, `freeze_tick` the tick of the deciding freeze (for the credited bonus of a game).
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    decisions: &[Decision],
    fields: &MapFields,
    outcome: &EpisodeOutcome,
    kind: EpisodeKind,
    de: i32,
    gamma: f32,
    cfg: &PpoReward,
) -> (Vec<f32>, RewardParts) {
    let n = decisions.len();
    let mut r = vec![0.0f32; n];
    let mut parts = RewardParts::default();
    let Some(first) = decisions.iter().position(|d| d.acted) else {
        return (r, parts);
    };
    let last = n - 1;
    let phi: Vec<f32> = decisions[first..]
        .iter()
        .map(|d| potential(&d.obs, fields, cfg.own_hazard))
        .collect();
    parts.phi_start = phi[0];
    for (k, f) in shaping_terms(&phi, gamma).into_iter().enumerate() {
        r[first + k] += cfg.shaping * f;
        parts.shaping += cfg.shaping * f;
    }
    // Pulled off the freeze: the timer of the victim starts to run down while the fly's hook holds it.
    for i in first..last {
        let (a, b) = (&decisions[i].obs, &decisions[i + 1].obs);
        if let (Some(va), Some(vb)) = (a.others.first(), b.others.first()) {
            let hooking = a.self_state.hooked_player == va.id || b.self_state.hooked_player == vb.id;
            if va.freeze_ticks_remaining as f32 >= FREEZE_TICKS - 1.0
                && vb.freeze_ticks_remaining < va.freeze_ticks_remaining
                && hooking
                && a.self_state.hook_state != HOOK_IDLE
            {
                r[i] -= cfg.free_victim;
                parts.free_victim -= cfg.free_victim;
                parts.freed_victim = true;
                break;
            }
        }
    }
    let focal_out = outcome.focal_out_in_window;
    match kind {
        EpisodeKind::Post => {
            parts.terminal = if focal_out {
                -cfg.self_freeze
            } else if outcome.held_block {
                cfg.held
            } else {
                0.0
            };
        }
        EpisodeKind::Game => {
            let won_credited = outcome.result == GameResult::W && outcome.credited;
            // The credited freeze pays at the decision whose two ticks contain it.
            if won_credited
                && let Some(i) = decisions.iter().rposition(|d| d.obs.tick < outcome.end_tick)
                && outcome.end_tick <= decisions[i].obs.tick + de
            {
                r[i] += cfg.credited;
                parts.credited = cfg.credited;
            }
            parts.terminal = match outcome.result {
                GameResult::W if focal_out => -cfg.self_freeze - parts.credited,
                GameResult::W if outcome.credited && outcome.held_block => cfg.held - parts.credited,
                GameResult::W if outcome.credited => 0.0,
                GameResult::W => 0.0,
                GameResult::L => -cfg.self_freeze,
                GameResult::D | GameResult::T => -cfg.draw_or_timeout,
            };
        }
    }
    r[last] += parts.terminal;
    (r, parts)
}

/// Generalised advantage estimation over one episode's acted decisions: `rewards[i]` follows decision `i`, `values[i]` is `V(s_i)`, and the
/// state after the last decision is terminal (value `0`). Returns `(advantages, returns)` with `returns = advantages + values`.
pub fn gae(rewards: &[f32], values: &[f32], gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(rewards.len(), values.len());
    let n = rewards.len();
    let mut adv = vec![0.0f32; n];
    let mut next_adv = 0.0f32;
    for i in (0..n).rev() {
        let v_next = if i + 1 < n { values[i + 1] } else { 0.0 };
        let delta = rewards[i] + gamma * v_next - values[i];
        next_adv = delta + gamma * lambda * next_adv;
        adv[i] = next_adv;
    }
    let ret = adv.iter().zip(values).map(|(a, v)| a + v).collect();
    (adv, ret)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_fly::rng::SplitMix64;

    #[test]
    fn shaping_telescopes_for_any_potential_and_any_discount() {
        let mut rng = SplitMix64::new(4);
        for gamma in [0.9f32, 0.99, 0.995, 1.0] {
            for n in [1usize, 2, 7, 120] {
                let phi: Vec<f32> = (0..n).map(|_| rng.next_f32_unit()).collect();
                let f = shaping_terms(&phi, gamma);
                // The discounted sum over the whole episode: -Phi(first), because the state after the last one is terminal.
                let sum: f64 = f
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| f64::from(gamma).powi(i as i32) * f64::from(x))
                    .sum();
                assert!(
                    (sum + f64::from(phi[0])).abs() < 1e-4,
                    "gamma {gamma} n {n}: {sum} vs {}",
                    -phi[0]
                );
                // Any prefix of k steps telescopes to gamma^k Phi(s_k) - Phi(s_0) (what a truncated rollout sees).
                for k in 0..n {
                    let part: f64 = f[..k]
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| f64::from(gamma).powi(i as i32) * f64::from(x))
                        .sum();
                    let want = f64::from(gamma).powi(k as i32) * f64::from(phi[k]) - f64::from(phi[0]);
                    assert!((part - want).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn gae_reduces_to_the_discounted_return_at_lambda_one_and_to_td_at_zero() {
        let r = [0.0f32, 0.0, 1.0];
        let v = [0.2f32, 0.4, 0.6];
        let g = 0.9;
        let (a1, ret1) = gae(&r, &v, g, 1.0);
        // lambda = 1: the Monte-Carlo return minus the value.
        let mc = [g * g * 1.0, g * 1.0, 1.0];
        for i in 0..3 {
            assert!((ret1[i] - mc[i]).abs() < 1e-6, "{i}: {} vs {}", ret1[i], mc[i]);
            assert!((a1[i] - (mc[i] - v[i])).abs() < 1e-6);
        }
        // lambda = 0: the one-step TD error, the last state bootstraps from 0.
        let (a0, _) = gae(&r, &v, g, 0.0);
        assert!((a0[0] - (0.0 + g * 0.4 - 0.2)).abs() < 1e-6);
        assert!((a0[2] - (1.0 - 0.6)).abs() < 1e-6);
    }
}
