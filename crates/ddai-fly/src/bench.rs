//! Reusable pieces of the performance-measurement methodology (acceptance criterion 4): the
//! "documented distribution" of synthetic input traffic, and the realistic 25Hz duty-cycle loop
//! (`docs/research/rust-stack.md` §4's phase-0 benchmark methodology: compute a decision, sleep to
//! the next tick, repeat). Shared by `tests/stability.rs` (acceptance criterion 3) and
//! `ddnet-ai fly bench` (acceptance criterion 6) so the two don't duplicate this logic.
//!
//! Deliberately does **not** pin the calling thread to a core — that needs `core_affinity` (or
//! `sched_setaffinity`), which this crate keeps out of its normal dependency graph (a dev-only
//! dependency for its own criterion benches — see the crate README). Callers that want a clean,
//! unpinned-noise-free measurement (the CLI, criterion benches) pin the thread themselves before
//! calling into this module.

use std::time::{Duration, Instant};

use crate::model::FlyModel;
use crate::rng::SplitMix64;
use crate::state::FlyState;

/// The "uniform \[0,1) on 20% of inputs, changing every decision" distribution from acceptance
/// criterion 3: each call, every entry of `buf` is independently redrawn (uniform `[0,1)`) with
/// probability `refresh_prob`; otherwise it keeps its previous value. `buf` doubles as both the
/// running "previous values" state and the output.
pub fn refresh_inputs_partial(rng: &mut SplitMix64, buf: &mut [f32], refresh_prob: f32) {
    for x in buf.iter_mut() {
        if rng.next_f32_unit() < refresh_prob {
            *x = rng.next_f32_unit();
        }
    }
}

/// One `run_duty_cycle` call's timing summary, in milliseconds (except `num_decisions`/
/// `over_5ms`, counts).
#[derive(Debug, Clone, Copy)]
pub struct DutyCycleReport {
    pub num_decisions: u32,
    pub substeps: u32,
    pub median_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    /// Count of decisions whose `step_decision` call alone (not counting the sleep) took more
    /// than 5ms — acceptance criterion 4's "count > 5ms".
    pub over_5ms: u32,
}

fn percentile_ms(sorted_ms: &[f64], q: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = (((sorted_ms.len() - 1) as f64) * q).round() as usize;
    sorted_ms[idx]
}

/// Runs `num_decisions` decisions of synthetic input traffic ([`refresh_inputs_partial`], seeded
/// by `seed`), timing only the `step_decision` call itself, then sleeping to the next `tick`
/// boundary (an absolute deadline that advances by `tick` every iteration, so sleep jitter doesn't
/// accumulate drift across the run — matches `docs/research/rust-stack.md` §4's `duty` tool).
pub fn run_duty_cycle(
    model: &FlyModel,
    state: &mut FlyState,
    num_decisions: u32,
    tick: Duration,
    seed: u64,
    input_refresh_prob: f32,
) -> DutyCycleReport {
    let mut rng = SplitMix64::new(seed);
    let mut inputs = vec![0.0f32; model.num_inputs()];
    let mut durations_ms: Vec<f64> = Vec::with_capacity(num_decisions as usize);

    let mut deadline = Instant::now() + tick;
    for _ in 0..num_decisions {
        refresh_inputs_partial(&mut rng, &mut inputs, input_refresh_prob);

        let start = Instant::now();
        let _ = state.step_decision(model, &inputs);
        durations_ms.push(start.elapsed().as_secs_f64() * 1000.0);

        let now = Instant::now();
        if now < deadline {
            std::thread::sleep(deadline - now);
        }
        deadline += tick;
    }

    let mut sorted = durations_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("duration is never NaN"));
    let over_5ms = durations_ms.iter().filter(|&&d| d > 5.0).count() as u32;
    let mean_ms = durations_ms.iter().sum::<f64>() / durations_ms.len().max(1) as f64;

    DutyCycleReport {
        num_decisions,
        substeps: model.config().substeps_per_decision,
        median_ms: percentile_ms(&sorted, 0.5),
        p99_ms: percentile_ms(&sorted, 0.99),
        max_ms: sorted.last().copied().unwrap_or(0.0),
        mean_ms,
        over_5ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FlyConfig;
    use crate::params::FlyParams;
    use crate::test_fixtures::tiny_chain_flyg;

    #[test]
    fn refresh_inputs_partial_with_zero_prob_never_changes_values() {
        let mut rng = SplitMix64::new(1);
        let mut buf = vec![0.5f32; 100];
        refresh_inputs_partial(&mut rng, &mut buf, 0.0);
        assert!(buf.iter().all(|&x| x == 0.5));
    }

    #[test]
    fn refresh_inputs_partial_with_prob_one_always_redraws() {
        let mut rng = SplitMix64::new(1);
        let mut buf = vec![0.5f32; 100];
        refresh_inputs_partial(&mut rng, &mut buf, 1.0);
        assert!(buf.iter().all(|&x| (0.0..1.0).contains(&x)));
        assert!(
            buf.iter().any(|&x| x != 0.5),
            "redrawing 100 uniforms should not all land on 0.5"
        );
    }

    #[test]
    fn run_duty_cycle_reports_sane_percentiles_on_a_tiny_graph() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        // Tiny tick so this test doesn't actually take 20 decisions * 40ms.
        let report = run_duty_cycle(&model, &mut state, 20, Duration::from_millis(1), 7, 0.2);
        assert_eq!(report.num_decisions, 20);
        assert!(report.median_ms <= report.p99_ms);
        assert!(report.p99_ms <= report.max_ms);
        assert!(report.mean_ms >= 0.0);
    }
}
