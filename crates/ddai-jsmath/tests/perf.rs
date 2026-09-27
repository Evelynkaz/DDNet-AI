//! Performance note (task acceptance criterion 6): ns/call for each ported function vs the
//! closest Rust `std` equivalent, so the planner port (task 3.2) knows the cost of V8-bit-exact
//! math vs plain `std`. No target is asserted — this is a report, run manually:
//!
//! `cargo test -p ddai-jsmath --release --test perf -- --ignored --nocapture`
//!
//! F8 fix (review round 1): the input generator used to run *inside* the timed loop
//! (`x.rem_euclid(1000.0)` per call), so a chunk of every measurement was the generator's own
//! cost, not the function under test; `tanh`'s domain was also `[0.001, 1000]`, so it almost
//! always took the `|x| >= 22 -> +-1` saturated fast path, not the interesting branches. Inputs
//! are now precomputed into a `Vec` per function *before* timing starts, each in a domain that
//! actually exercises the function's real work.

use std::hint::black_box;
use std::time::Instant;

const N: usize = 2_000_000;

/// A small, self-contained PRNG for generating the input vectors (same shape as `tests/oracle.rs`'s
/// `Prng` — duplicated rather than shared, since each `tests/*.rs` file is its own crate).
struct Prng(u64);
impl Prng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
}

/// Times `f(inputs[i])` for every `i` (i.e. `inputs.len()` calls total), having already paid for
/// generating `inputs` before this function is called — the timed loop only ever indexes into an
/// already-built `Vec` and calls `f`, nothing else.
fn time_ns_over<F: Fn(f64) -> f64>(inputs: &[f64], f: F) -> f64 {
    // Warm up (page faults, branch predictor, icache), then time.
    for &x in inputs.iter().take(1000) {
        black_box(f(black_box(x)));
    }
    let start = Instant::now();
    let mut acc = 0.0f64;
    for &x in inputs {
        acc += f(black_box(x));
    }
    black_box(acc);
    start.elapsed().as_secs_f64() * 1e9 / inputs.len() as f64
}

fn report_row(name: &str, ns_jsmath: f64, ns_std: f64) {
    println!(
        "{name:<10} jsmath={ns_jsmath:>8.2} ns/call   std={ns_std:>8.2} ns/call   ratio={:.2}x",
        ns_jsmath / ns_std
    );
}

#[test]
#[ignore]
fn perf_report() {
    let mut rng = Prng(0x9E3779B97F4A7C15);
    let wide: Vec<f64> = (0..N).map(|_| rng.uniform(-1.0e5, 1.0e5)).collect();
    let trig_domain: Vec<f64> = (0..N)
        .map(|_| rng.uniform(-4.0 * std::f64::consts::PI, 4.0 * std::f64::consts::PI))
        .collect();
    // F8 fix: was `[0.001, 1000]`, which for `tanh` (saturates to +-1 once `|x| >= 22`) spent
    // almost the entire run on the trivial saturated branch. `[-3, 3]` stays inside `tanh`'s two
    // real computational branches (`expm1(+-2|x|)`-based, split at `|x| == 1`).
    let tanh_domain: Vec<f64> = (0..N).map(|_| rng.uniform(-3.0, 3.0)).collect();
    let exp_domain: Vec<f64> = (0..N).map(|_| rng.uniform(-50.0, 50.0)).collect();
    let log_domain: Vec<f64> = (0..N).map(|_| rng.uniform(0.05, 1.0e5)).collect();
    let pow_exponent: Vec<f64> = (0..N).map(|_| rng.uniform(-100.0, 100.0)).collect();

    println!("\n=== ddai-jsmath perf report (N={N} calls each, release) ===");
    report_row(
        "round",
        time_ns_over(&wide, ddai_jsmath::round),
        time_ns_over(&wide, f64::round),
    );
    report_row(
        "trunc",
        time_ns_over(&wide, ddai_jsmath::trunc),
        time_ns_over(&wide, f64::trunc),
    );
    report_row(
        "floor",
        time_ns_over(&wide, ddai_jsmath::floor),
        time_ns_over(&wide, f64::floor),
    );
    report_row(
        "ceil",
        time_ns_over(&wide, ddai_jsmath::ceil),
        time_ns_over(&wide, f64::ceil),
    );
    report_row(
        "abs",
        time_ns_over(&wide, ddai_jsmath::abs),
        time_ns_over(&wide, f64::abs),
    );
    report_row(
        "sqrt",
        time_ns_over(&log_domain, ddai_jsmath::sqrt),
        time_ns_over(&log_domain, f64::sqrt),
    );
    report_row(
        "sin",
        time_ns_over(&trig_domain, ddai_jsmath::sin),
        time_ns_over(&trig_domain, f64::sin),
    );
    report_row(
        "cos",
        time_ns_over(&trig_domain, ddai_jsmath::cos),
        time_ns_over(&trig_domain, f64::cos),
    );
    report_row(
        "tanh",
        time_ns_over(&tanh_domain, ddai_jsmath::tanh),
        time_ns_over(&tanh_domain, f64::tanh),
    );
    report_row(
        "atan",
        time_ns_over(&trig_domain, ddai_jsmath::atan),
        time_ns_over(&trig_domain, f64::atan),
    );
    report_row(
        "exp",
        time_ns_over(&exp_domain, ddai_jsmath::exp),
        time_ns_over(&exp_domain, f64::exp),
    );
    report_row(
        "log",
        time_ns_over(&log_domain, ddai_jsmath::log),
        time_ns_over(&log_domain, f64::ln),
    );

    let ns_hypot2 = time_ns_over(&wide, |v| ddai_jsmath::hypot2(v, v * 0.5));
    let ns_hypot_std = time_ns_over(&wide, |v| v.hypot(v * 0.5));
    report_row("hypot2", ns_hypot2, ns_hypot_std);

    let ns_atan2 = time_ns_over(&trig_domain, |v| ddai_jsmath::atan2(v, v * 0.5));
    let ns_atan2_std = time_ns_over(&trig_domain, |v| v.atan2(v * 0.5));
    report_row("atan2", ns_atan2, ns_atan2_std);

    // `1.4` is the actual velramp base the old bot uses (`Math.pow(1.4, p)`); see the crate
    // README. `black_box` on both operands here too, for the same reason `pow.rs` needs it
    // internally (F1): without it, `1.4f64.powf(v)` written directly at a call site is just a
    // regular (non-power-of-two, non-integer-exponent) `pow` call with no known LLVM
    // constant-argument rewrite, so this specific perf comparison happens not to be affected
    // either way — but consistently black-boxing avoids relying on that non-rewrite staying true.
    let ns_pow = time_ns_over(&pow_exponent, |v| ddai_jsmath::pow(black_box(1.4), v));
    let ns_pow_std = time_ns_over(&pow_exponent, |v| black_box(1.4f64).powf(v));
    report_row("pow", ns_pow, ns_pow_std);

    let mut gauss_rng = ddai_jsmath::Rng::new(42);
    let start = Instant::now();
    let mut acc = 0.0;
    for _ in 0..N {
        acc += black_box(gauss_rng.next_gaussian());
    }
    black_box(acc);
    let ns_gauss = start.elapsed().as_secs_f64() * 1e9 / N as f64;
    println!(
        "{:<10} Rng::next_gaussian={ns_gauss:.2} ns/call (no std equivalent to compare)",
        "Rng"
    );
}
