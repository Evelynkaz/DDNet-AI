//! Acceptance criterion 3: stability on the real S/M graphs
//! (`~/aiddnet/data/connectome/compiled/fly-{S,M}-v1.flyg`), with the default init and random
//! input currents, for 60s of simulated time: no NaN/inf, a sane (neither dead nor saturated)
//! fraction of active neurons, and DN rates that actually vary with input — **from a converged
//! resting state** (review round 1, F1: the original version started the 60s window from
//! whatever `warm_up`'s old fixed 400ms reached, which for the real graphs' default init was
//! still 90%+ transient, not settled activity; see the crate README's "Warm-up convergence"
//! table). Also checks the step response is actually asymmetric left vs. right (F4: the
//! "variance with random input" metric alone can't tell a genuinely responsive network apart
//! from a barely-alive one with unrelated per-decision noise).
//!
//! `#[ignore]`d: this crate's core dependency (real connectome data) lives outside the repo and
//! isn't guaranteed present in every environment (see `docs/formats.md` §8 / `docs/FLY.md`) — run
//! explicitly with `cargo test -p ddai-fly --test stability -- --ignored --nocapture` when the
//! data is available (as it is on the project's own machine). This mirrors the project's existing
//! convention for hardware/data-dependent tests (see `CLAUDE.md`'s `parity_cpp -- --ignored`).

use std::path::PathBuf;

use ddai_fly::{FlyConfig, FlyModel, FlyParams, FlyState, WarmUpReport};
use ddai_flyg::Side;

fn compiled_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set to find ~/aiddnet/data/connectome/compiled");
    PathBuf::from(home).join("aiddnet/data/connectome/compiled")
}

/// Left-only / right-only step response from a given rest `V`, sustained for `duration_ms` of
/// simulated time (review round 1, F4). `left_input`/`right_input` set every `InputVisual`/
/// `InputAscending` neuron on that side to `1.0` (others `0.0`, including any `Side::M`/`Unknown`
/// input, which this test doesn't try to characterise).
struct StepResponseReport {
    left_dn_l_sum: f64,
    left_dn_r_sum: f64,
    right_dn_l_sum: f64,
    right_dn_r_sum: f64,
    /// Count of output-role neurons whose rate differs by more than `moved_threshold` between
    /// the left-only and right-only steady states.
    num_dn_moved: usize,
    num_dn_total: usize,
}

fn side_input_vector(model: &FlyModel, side: Side) -> Vec<f32> {
    model
        .input_neuron_indices()
        .iter()
        .map(|&dense| {
            if model.flyg().neurons[dense as usize].side == side {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn run_constant_input_from(
    model: &FlyModel,
    state: &mut FlyState,
    rest_v: &[f32],
    input: &[f32],
    duration_ms: f32,
) -> Vec<f32> {
    state.set_v(model, rest_v);
    let decision_ms = model.config().decision_ms();
    let num_decisions = ((duration_ms / decision_ms).round() as u32).max(1);
    let mut last_dn = vec![0.0f32; model.num_outputs()];
    for _ in 0..num_decisions {
        let out = state.step_decision(model, input);
        last_dn.copy_from_slice(out.dn_rates);
    }
    last_dn
}

fn measure_step_response(
    model: &FlyModel,
    state: &mut FlyState,
    rest_v: &[f32],
    duration_ms: f32,
    moved_threshold: f32,
) -> StepResponseReport {
    let left_input = side_input_vector(model, Side::L);
    let right_input = side_input_vector(model, Side::R);
    let dn_left = run_constant_input_from(model, state, rest_v, &left_input, duration_ms);
    let dn_right = run_constant_input_from(model, state, rest_v, &right_input, duration_ms);

    let dn_sides: Vec<Side> = model
        .output_neuron_indices()
        .iter()
        .map(|&d| model.flyg().neurons[d as usize].side)
        .collect();

    let mut left_dn_l_sum = 0.0f64;
    let mut left_dn_r_sum = 0.0f64;
    let mut right_dn_l_sum = 0.0f64;
    let mut right_dn_r_sum = 0.0f64;
    let mut num_dn_moved = 0usize;
    for k in 0..dn_sides.len() {
        match dn_sides[k] {
            Side::L => {
                left_dn_l_sum += f64::from(dn_left[k]);
                right_dn_l_sum += f64::from(dn_right[k]);
            }
            Side::R => {
                left_dn_r_sum += f64::from(dn_left[k]);
                right_dn_r_sum += f64::from(dn_right[k]);
            }
            Side::M | Side::Unknown => {}
        }
        if (dn_left[k] - dn_right[k]).abs() > moved_threshold {
            num_dn_moved += 1;
        }
    }

    StepResponseReport {
        left_dn_l_sum,
        left_dn_r_sum,
        right_dn_l_sum,
        right_dn_r_sum,
        num_dn_moved,
        num_dn_total: dn_sides.len(),
    }
}

struct StabilityReport {
    name: &'static str,
    num_neurons: usize,
    num_decisions: u32,
    warm_up: WarmUpReport,
    /// Mean over decisions of the fraction of neurons with `f(V) > 0` — acceptance criterion 3's
    /// literal metric. Near 0 would mean "dead" (the network never fires at all).
    fraction_rate_positive: f64,
    /// Mean over decisions of the fraction of neurons with `f(V) > 0.9 * r_max` — near 1 would
    /// mean "saturated". Distinct from `fraction_rate_positive`.
    fraction_rate_saturated: f64,
    mean_rate: f64,
    /// Secondary/reported only (F4): variance of DN rates under partially-randomised input.
    /// Doesn't gate pass/fail on its own any more — `step_response` does that.
    dn_rate_variance_mean: f64,
    max_abs_v: f32,
    step_response: StepResponseReport,
}

fn run_stability_check(name: &'static str, flyg_path: PathBuf) -> StabilityReport {
    let flyg = ddai_flyg::load(&flyg_path).unwrap_or_else(|e| panic!("failed to load {}: {e}", flyg_path.display()));
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 42);
    let model = FlyModel::new(flyg, config, params).expect("build FlyModel");
    let mut state = FlyState::new(&model);
    let warm_up_report = state.warm_up(&model);
    let rest_v = state.v().to_vec();

    // 60s of simulated time at 40ms/decision = 1500 decisions.
    let decision_s = f64::from(model.config().decision_ms()) / 1000.0;
    let num_decisions = (60.0 / decision_s).round() as u32;

    let mut rng = ddai_fly::rng::SplitMix64::new(2026);
    let mut inputs = vec![0.0f32; model.num_inputs()];

    let num_neurons = model.num_neurons();
    let num_outputs = model.num_outputs();
    let r_max = model.config().r_max;
    let mut rate_positive_fraction_sum = 0.0f64;
    let mut rate_saturated_fraction_sum = 0.0f64;
    let mut mean_rate_sum = 0.0f64;
    let mut dn_sum = vec![0.0f64; num_outputs];
    let mut dn_sum_sq = vec![0.0f64; num_outputs];
    let mut max_abs_v = 0.0f32;

    for decision in 0..num_decisions {
        ddai_fly::refresh_inputs_partial(&mut rng, &mut inputs, 0.2);
        // `DecisionOutput` borrows `state`, so pull out everything this loop needs from it before
        // touching `state` again (e.g. `state.v()` below) — see `DecisionOutput`'s doc comment.
        let out = state.step_decision(&model, &inputs);
        let dn_rates: Vec<f32> = out.dn_rates.to_vec();
        let per_type_rates: Vec<f32> = out.per_type_mean_rate.to_vec();

        for &v in state.v() {
            assert!(v.is_finite(), "{name}: V went non-finite at decision {decision}: {v}");
            max_abs_v = max_abs_v.max(v.abs());
        }
        for &r in &dn_rates {
            assert!(
                r.is_finite(),
                "{name}: DN rate went non-finite at decision {decision}: {r}"
            );
        }
        for &r in &per_type_rates {
            assert!(
                r.is_finite(),
                "{name}: per-type mean rate went non-finite at decision {decision}: {r}"
            );
        }

        let rates: Vec<f32> = state
            .v()
            .iter()
            .map(|&v| ddai_fly::activation::activation(v, r_max))
            .collect();
        let positive = rates.iter().filter(|&&r| r > 0.0).count();
        let saturated = rates.iter().filter(|&&r| r > 0.9 * r_max).count();
        rate_positive_fraction_sum += positive as f64 / num_neurons as f64;
        rate_saturated_fraction_sum += saturated as f64 / num_neurons as f64;
        mean_rate_sum += rates.iter().map(|&r| f64::from(r)).sum::<f64>() / num_neurons as f64;

        for (k, &r) in dn_rates.iter().enumerate() {
            dn_sum[k] += f64::from(r);
            dn_sum_sq[k] += f64::from(r) * f64::from(r);
        }
    }

    let n = f64::from(num_decisions);
    let dn_rate_variance_mean = dn_sum
        .iter()
        .zip(&dn_sum_sq)
        .map(|(&s, &sq)| (sq / n) - (s / n) * (s / n))
        .sum::<f64>()
        / num_outputs.max(1) as f64;

    let step_response = measure_step_response(&model, &mut state, &rest_v, 1000.0, 0.05);

    StabilityReport {
        name,
        num_neurons,
        num_decisions,
        warm_up: warm_up_report,
        fraction_rate_positive: rate_positive_fraction_sum / n,
        fraction_rate_saturated: rate_saturated_fraction_sum / n,
        mean_rate: mean_rate_sum / n,
        dn_rate_variance_mean,
        max_abs_v,
        step_response,
    }
}

fn assert_report_is_sane(report: &StabilityReport) {
    let sr = &report.step_response;
    println!(
        "[{}] neurons={} decisions={} warm_up=(converged={}, decisions={}, elapsed_ms={:.0}, final_max_dV={:.5}) \
         fraction_rate_positive={:.4} fraction_rate_saturated={:.6} mean_rate={:.4} dn_rate_variance_mean={:.8} \
         max_abs_v={:.3} step_response=(left L/R={:.2}/{:.2}, right L/R={:.2}/{:.2}, moved={}/{})",
        report.name,
        report.num_neurons,
        report.num_decisions,
        report.warm_up.converged,
        report.warm_up.decisions_run,
        report.warm_up.elapsed_ms,
        report.warm_up.final_max_delta_v,
        report.fraction_rate_positive,
        report.fraction_rate_saturated,
        report.mean_rate,
        report.dn_rate_variance_mean,
        report.max_abs_v,
        sr.left_dn_l_sum,
        sr.left_dn_r_sum,
        sr.right_dn_l_sum,
        sr.right_dn_r_sum,
        sr.num_dn_moved,
        sr.num_dn_total,
    );

    assert!(
        report.warm_up.converged,
        "{}: warm_up did not converge within the cap (final max|dV|={}) — the 60s stability window \
         would start from an unconverged transient, not a resting state",
        report.name, report.warm_up.final_max_delta_v
    );
    assert!(
        report.fraction_rate_positive > 0.01,
        "{}: network looks dead (only {:.4}% of neurons ever fire at all)",
        report.name,
        report.fraction_rate_positive * 100.0
    );
    assert!(
        report.fraction_rate_saturated < 0.9,
        "{}: network looks saturated ({:.4}% of neurons pegged above 90% of r_max)",
        report.name,
        report.fraction_rate_saturated * 100.0
    );
    // F4: the step response itself, not just "DN rates vary with unrelated random input" — a
    // left-only stimulus must favour left DNs over right, and vice versa for right-only, and a
    // clear majority of DNs must actually move between the two conditions.
    assert!(
        sr.left_dn_l_sum > sr.left_dn_r_sum,
        "{}: left-only input should favour left DNs: L={:.3} R={:.3}",
        report.name,
        sr.left_dn_l_sum,
        sr.left_dn_r_sum
    );
    assert!(
        sr.right_dn_r_sum > sr.right_dn_l_sum,
        "{}: right-only input should favour right DNs: L={:.3} R={:.3}",
        report.name,
        sr.right_dn_l_sum,
        sr.right_dn_r_sum
    );
    // Threshold, not "a majority": `DEFAULT_ALPHA_INIT` (review round 1, F1b) deliberately trades
    // off *some* of this against settle time — the alpha-sweep table shows 30-40% of DNs moving at
    // the chosen alpha vs. 70%+ at a much slower-settling one. 15% is comfortably above what a
    // barely-alive network shows (alpha=0.5 in that same sweep: ~11-12%) and comfortably below
    // what the chosen alpha actually delivers, so it still catches a real regression.
    assert!(
        sr.num_dn_moved as f64 >= 0.15 * sr.num_dn_total as f64,
        "{}: too few DNs distinguish left-only from right-only input ({}/{} moved by > 0.05)",
        report.name,
        sr.num_dn_moved,
        sr.num_dn_total
    );
}

#[test]
#[ignore = "needs ~/aiddnet/data/connectome/compiled/fly-S-v1.flyg (real connectome data, not in the repo)"]
fn stability_on_real_s_graph() {
    let path = compiled_dir().join("fly-S-v1.flyg");
    assert!(
        path.exists(),
        "expected {} to exist — build it with `ddai-connectome build-subgraph`",
        path.display()
    );
    let report = run_stability_check("S", path);
    assert_report_is_sane(&report);
}

#[test]
#[ignore = "needs ~/aiddnet/data/connectome/compiled/fly-M-v1.flyg (real connectome data, not in the repo)"]
fn stability_on_real_m_graph() {
    let path = compiled_dir().join("fly-M-v1.flyg");
    assert!(
        path.exists(),
        "expected {} to exist — build it with `ddai-connectome build-subgraph`",
        path.display()
    );
    let report = run_stability_check("M", path);
    assert_report_is_sane(&report);
}

/// Not a pass/fail test — a reproducible way to regenerate the crate README's α-sweep table
/// (review round 1, F1b: pick `DEFAULT_ALPHA_INIT` by measuring settle time, active fraction and
/// step-response asymmetry across candidates, not just "healthy activity" as the original
/// tuning pass did). Run with `--nocapture` to see the table; `#[ignore]`d for the same reason as
/// the other tests here (needs real data) plus it's meant to be read, not asserted on.
#[test]
#[ignore = "prints the alpha-init sweep table used to pick DEFAULT_ALPHA_INIT; needs real .flyg data, --nocapture to see output"]
fn alpha_sweep_experiment() {
    println!(
        "{:>10} {:>6} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "graph", "alpha", "converged", "decisions", "elapsed_ms", "active%", "sat%", "moved", "L/R_ok"
    );
    for &(name, path_name) in &[("S", "fly-S-v1.flyg"), ("M", "fly-M-v1.flyg")] {
        let path = compiled_dir().join(path_name);
        if !path.exists() {
            println!("skipping {name}: {} not found", path.display());
            continue;
        }
        let flyg = ddai_flyg::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
        let config = FlyConfig::default();

        for &alpha in &[0.5f32, 1.0, 1.5, 2.0, 2.3, 3.0, 4.0, 5.0] {
            let params = FlyParams::init_default_with_alpha(&flyg, &config, 42, alpha);
            let model = FlyModel::new(flyg.clone(), config, params).expect("build FlyModel");
            let mut state = FlyState::new(&model);
            let warm_up_report = state.warm_up(&model);
            let rest_v = state.v().to_vec();

            let r_max = model.config().r_max;
            let rates: Vec<f32> = state
                .v()
                .iter()
                .map(|&v| ddai_fly::activation::activation(v, r_max))
                .collect();
            let num_neurons = model.num_neurons();
            let active_pct = 100.0 * rates.iter().filter(|&&r| r > 0.0).count() as f64 / num_neurons as f64;
            let sat_pct = 100.0 * rates.iter().filter(|&&r| r > 0.9 * r_max).count() as f64 / num_neurons as f64;

            let sr = measure_step_response(&model, &mut state, &rest_v, 1000.0, 0.05);
            let l_r_ok = sr.left_dn_l_sum > sr.left_dn_r_sum && sr.right_dn_r_sum > sr.right_dn_l_sum;

            println!(
                "{:>10} {:>6.1} {:>10} {:>9} {:>9.0} {:>9.2} {:>9.4} {:>4}/{:<4} {:>7}",
                name,
                alpha,
                warm_up_report.converged,
                warm_up_report.decisions_run,
                warm_up_report.elapsed_ms,
                active_pct,
                sat_pct,
                sr.num_dn_moved,
                sr.num_dn_total,
                l_r_ok,
            );
        }
    }
}
