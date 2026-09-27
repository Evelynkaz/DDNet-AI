//! Proof that this crate is bit-exact with the real Node 24.21.0 / V8 13.6 binary on this machine.
//!
//! Two tests share every helper below (probe generation, the function table, the golden fixture
//! layout), so "the golden fixture is checked" and "the golden fixture was generated" are
//! provably the same code path (task acceptance criterion 5):
//!
//! - `golden_fixture_matches_v8` (always runs, no Node needed): re-derives the same small probe
//!   set this crate's own inputs were generated from (a fixed PRNG seed, [`FIXTURE_SEED`]),
//!   computes this crate's own results, and compares them against the pre-recorded real-V8
//!   results committed in `tests/fixtures/golden.bin` (+ `golden.sha256`). This is what
//!   `cargo test --workspace` runs, including in CI (no Node there).
//! - `full_oracle_vs_real_v8` (`#[ignore]`, run via `tools/jsmath-oracle/run.sh`): the same
//!   generator, at >= 10^6 probes per function (`JSMATH_PROBES` env var, default 1_000_000),
//!   actually shells out to the real `node` binary (via `tools/jsmath-oracle/probe.mjs` and
//!   `rng_probe.mjs`) and asserts zero mismatches.
//! - `regenerate_golden_fixture` (`#[ignore]`, run by a human when the crate's probe generator or
//!   the old TS source change): the small-scale version of the above, but *writes*
//!   `tests/fixtures/golden.bin`/`.sha256` instead of comparing against them.
//!
//! Comparison rule (per the task spec): two `f64` results match if their bit patterns are equal,
//! *or* if both are NaN (any payload/signaling bit) — see [`f64_matches`].

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use ddai_jsmath as jsmath;

// ---------------------------------------------------------------------------------------------
// A small, self-contained PRNG for generating reproducible probe inputs (this is test-only
// infrastructure, not the `Rng` under test — using the crate's own `Rng` to test itself would be
// circular). `splitmix64`, the 64-bit cousin of the `splitmix32` this crate's `Rng` seeds from;
// same idea, unrelated code path.
// ---------------------------------------------------------------------------------------------

struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        Prng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Raw random bit pattern, interpreted as `f64` — covers every exponent (including
    /// subnormals), both zeros, +-infinity and NaN with arbitrary payloads, each with realistic
    /// probability (the 11-bit exponent field is all-ones, i.e. inf/NaN, on 1/2048 draws).
    fn random_bits_f64(&mut self) -> f64 {
        f64::from_bits(self.next_u64())
    }

    /// Uniform `f64` in `[lo, hi)` (`lo < hi` finite).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        lo + u * (hi - lo)
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next_u64() as usize) % xs.len()]
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

/// Values every function's probe set includes verbatim (each cheaply reachable by pure uniform
/// bits too, but explicit inclusion guarantees they are always exercised regardless of draw luck)
fn special_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        -0.5,
        f64::MIN_POSITIVE, // smallest positive normal
        -f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 2.0, // subnormal
        f64::from_bits(1),       // smallest positive subnormal
        f64::EPSILON,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0x7FF0_0000_0000_0001), // NaN, alternate payload
        f64::from_bits(0xFFF8_0000_0000_0000), // NaN, sign bit set
        std::f64::consts::PI,
        std::f64::consts::PI / 2.0,
        std::f64::consts::PI * 2.0,
        0.49999999999999994, // the double just below 0.5 (round-tie adversarial point)
        4294967296.0,        // 2^32
        4294967295.0,        // 2^32 - 1
        -1.5,
        1.5,
        2.5,
        -2.5,
    ]
}

// ---------------------------------------------------------------------------------------------
// Per-function probe generation: mixes uniform random bit patterns, bot-relevant domain ranges
// and adversarial points, per the task's acceptance criteria (section 4).
// ---------------------------------------------------------------------------------------------

/// One column of a probe row for function `op`, at row `idx` (of `count`), argument position
/// `arg` (0-based, `< arity`). Deterministic in `(op, idx, arg)` given the same `rng` draw
/// sequence, so regenerating with a fresh `Prng::new(seed)` reproduces the exact same probes.
fn probe_value(op: &str, rng: &mut Prng, idx: u64, arg: usize) -> f64 {
    if op == "opp_seed_next" {
        // Scoped to its actual domain: the planner only ever calls this on a genuine u32 state
        // value (already `>>> 0`-masked from the previous step), not an arbitrary `f64` — unlike
        // every other function here, which the old bot can hand any JS `Number`.
        return f64::from(rng.next_u32());
    }
    match idx % 4 {
        0 => rng.random_bits_f64(),
        1 => domain_value(op, rng, arg),
        2 => adversarial_value(op, rng, arg),
        _ => {
            // A special value, lightly jittered on some draws so it also probes the immediate
            // neighborhood (`nextafter`-style) of the exact special point, not only the point
            // itself. Built fresh only on this branch (1 in 4 calls), not on every call.
            let specials = special_values();
            let base = *rng.pick(&specials);
            if rng.next_u64().is_multiple_of(2) || !base.is_finite() {
                base
            } else if rng.next_u64().is_multiple_of(2) {
                bump(base, 1)
            } else {
                bump(base, -1)
            }
        }
    }
}

/// The adjacent representable `f64` in the direction of `dir` (+1 or -1) ULPs — a minimal
/// `nextafter`, used for the round-tie-neighbor adversarial probes.
fn bump(x: f64, dir: i64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits() as i64;
    let bits = if x == 0.0 {
        if dir > 0 { 1i64 } else { (1u64 << 63) as i64 | 1 }
    } else if (x > 0.0) == (dir > 0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(bits as u64)
}

fn domain_value(op: &str, rng: &mut Prng, arg: usize) -> f64 {
    match op {
        "sin" | "cos" | "tanh" | "atan" => rng.uniform(-4.0 * std::f64::consts::PI, 4.0 * std::f64::consts::PI),
        "atan2" => rng.uniform(-1.0e5, 1.0e5),
        "exp" => rng.uniform(-50.0, 50.0),
        // F2 fix: `log`'s domain used to be `uniform(0, 1e5)`, spread thin across many decades of
        // magnitude on a *linear* scale (so almost all draws land in, say, [9e4, 1e5], leaving the
        // mantissa-normalized computation's sensitive region under-sampled). Every positive input
        // gets normalized to a mantissa in roughly [0.7, 1.4] before `log`'s polynomial runs
        // (`src/base/ieee754.cc`'s own frexp-style reduction), so sampling *directly* in a range
        // straddling that normalized region exercises the same code with much finer resolution —
        // confirmed empirically: this domain's mismatch rate against an injected 1-ulp `LG1` bug
        // is ~0.2%, vs ~0.003% for the old `uniform(0, 1e5)` (measured over 500k draws each, see
        // the crate README "Плотность проб для log/sin/cos"). `sqrt` keeps the old wide range
        // (never rounding-ambiguous, so density doesn't matter for it).
        "log" => rng.uniform(0.05, 12.0),
        "sqrt" => rng.uniform(0.0, 1.0e5),
        // F5 fix: `1.4` (the velramp base, `pow(1.4, p)`) as the *base* half the time — it used to
        // appear only in the exponent list below, so `pow(1.4, p)` itself was never actually
        // probed. `p` up to +-100 per the task's own domain description ("velramp").
        "pow" if arg == 0 => {
            if rng.next_u64().is_multiple_of(2) {
                *rng.pick(&[1.4, -1.4, 1.0 / 1.4])
            } else {
                rng.uniform(-1.0e3, 1.0e3)
            }
        }
        "pow" if arg == 1 => rng.uniform(-100.0, 100.0),
        "hypot2" | "hypot_n" => rng.uniform(-1.0e5, 1.0e5),
        "max2" | "min2" | "max_n" | "min_n" | "rem" => rng.uniform(-1.0e5, 1.0e5),
        "shl" | "shr" | "ushr" if arg == 1 => rng.uniform(0.0, 32.0),
        "shl" | "shr" | "ushr" | "imul" => rng.uniform(-1.0e10, 1.0e10),
        "to_int32" | "to_uint32" => rng.uniform(-1.0e10, 1.0e10),
        "round" | "trunc" | "floor" | "ceil" | "abs" | "sign" => rng.uniform(-1.0e5, 1.0e5),
        "opp_seed_next" => f64::from(rng.next_u32()),
        _ => rng.uniform(-1.0e5, 1.0e5),
    }
}

fn adversarial_value(op: &str, rng: &mut Prng, arg: usize) -> f64 {
    match op {
        "sin" | "cos" | "tanh" | "atan" => {
            // F5 fix: was always a coarse (+-1e-6, tens of millions of ULPs) jitter around
            // k*pi/2, which never actually exercises the extra `__kernel_rem_pio2` precision
            // passes (the 2nd/3rd iteration in `ieee754_rem_pio2`'s medium-size branch) that only
            // trigger within a few ULPs of the reduction boundary — see the crate README "Плотность
            // проб для log/sin/cos". Now: half the draws are *small* k (bot angles are bounded,
            // `angle/256` in network units) with a fine ULP-level (1..20 ULP) offset from k*pi/2
            // *and* k*pi/4 (the kernel_sin/kernel_cos branch boundary at |x| <= pi/4 is its own
            // adversarial point); the other half keep the original coarse jitter on huge k, which
            // is what actually stresses Payne-Hanek reduction (a fine ULP offset there would be
            // swamped by the huge base value's own ULP spacing anyway).
            if rng.next_u64().is_multiple_of(2) {
                let k = rng.uniform(-10_000.0, 10_000.0).round();
                let base = if rng.next_u64().is_multiple_of(2) {
                    k * std::f64::consts::FRAC_PI_2
                } else {
                    k * std::f64::consts::FRAC_PI_4
                };
                let ulps = 1 + (rng.next_u64() % 20) as i64;
                let dir = if rng.next_u64().is_multiple_of(2) { 1 } else { -1 };
                let mut v = base;
                for _ in 0..ulps {
                    v = bump(v, dir);
                }
                v
            } else {
                let k = rng.uniform(-1.0e15, 1.0e15).round();
                let near = k * std::f64::consts::FRAC_PI_2;
                let jitter = rng.uniform(-1.0e-6, 1.0e-6);
                near + jitter
            }
        }
        "log" => {
            // Same rationale as `domain_value`'s `"log"` case: doubles the fraction of `log`
            // probes that land in the mantissa-normalized, rounding-sensitive range (this counts
            // as this function's "adversarial" 1-in-4 draws, `domain_value` already covers the
            // other 1-in-4 — together roughly half of all `log` probes, instead of a quarter).
            rng.uniform(0.05, 12.0)
        }
        "round" => {
            // k + 0.5 ties, and their immediate neighbors, for |k| up to 2^53.
            let k = rng.uniform(-9.0e15, 9.0e15).round();
            let tie = k + 0.5;
            match rng.next_u64() % 3 {
                0 => tie,
                1 => bump(tie, 1),
                _ => bump(tie, -1),
            }
        }
        "hypot2" | "hypot_n" => {
            // Huge/tiny ratios and overflow-prone magnitudes.
            *rng.pick(&[
                1.0e300,
                1.0e-300,
                1.0e150,
                1.0e-150,
                f64::MAX / 4.0,
                f64::MIN_POSITIVE * 4.0,
            ])
        }
        "pow" if arg == 0 => *rng.pick(&[
            -1.0,
            1.0,
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0e300,
            1.0e-300,
        ]),
        "pow" if arg == 1 => *rng.pick(&[
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.5,
            2.0,
            f64::NAN,
            1.0e18,
            -1.0e18,
            0.0,
            -0.0,
        ]),
        "atan2" => {
            // Near axis/quadrant boundaries, plus zeros and infinities with both signs.
            *rng.pick(&[0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0])
        }
        "shl" | "shr" | "ushr" if arg == 1 => *rng.pick(&[0.0, 31.0, 32.0, 33.0, -1.0, 63.0, 64.0]),
        _ => rng.random_bits_f64(),
    }
}

// ---------------------------------------------------------------------------------------------
// Function table.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ret {
    F64,
    I32,
    U32,
}

struct FnSpec {
    name: &'static str,
    op: &'static str,
    arity: usize,
    ret: Ret,
    /// Computes this crate's own result for one row of `arity` inputs.
    rust: fn(&[f64]) -> RustVal,
}

enum RustVal {
    F64(f64),
    I32(i32),
    U32(u32),
}

macro_rules! spec {
    ($name:literal, $op:literal, 1, F64, $f:expr) => {
        FnSpec {
            name: $name,
            op: $op,
            arity: 1,
            ret: Ret::F64,
            rust: |a| RustVal::F64($f(a[0])),
        }
    };
    ($name:literal, $op:literal, 2, F64, $f:expr) => {
        FnSpec {
            name: $name,
            op: $op,
            arity: 2,
            ret: Ret::F64,
            rust: |a| RustVal::F64($f(a[0], a[1])),
        }
    };
    ($name:literal, $op:literal, 1, I32, $f:expr) => {
        FnSpec {
            name: $name,
            op: $op,
            arity: 1,
            ret: Ret::I32,
            rust: |a| RustVal::I32($f(a[0])),
        }
    };
    ($name:literal, $op:literal, 2, I32, $f:expr) => {
        FnSpec {
            name: $name,
            op: $op,
            arity: 2,
            ret: Ret::I32,
            rust: |a| RustVal::I32($f(a[0], a[1])),
        }
    };
    ($name:literal, $op:literal, 1, U32, $f:expr) => {
        FnSpec {
            name: $name,
            op: $op,
            arity: 1,
            ret: Ret::U32,
            rust: |a| RustVal::U32($f(a[0])),
        }
    };
    ($name:literal, $op:literal, 2, U32, $f:expr) => {
        FnSpec {
            name: $name,
            op: $op,
            arity: 2,
            ret: Ret::U32,
            rust: |a| RustVal::U32($f(a[0], a[1])),
        }
    };
}

fn function_table() -> Vec<FnSpec> {
    vec![
        spec!("round", "round", 1, F64, jsmath::round),
        spec!("trunc", "trunc", 1, F64, jsmath::trunc),
        spec!("floor", "floor", 1, F64, jsmath::floor),
        spec!("ceil", "ceil", 1, F64, jsmath::ceil),
        spec!("abs", "abs", 1, F64, jsmath::abs),
        spec!("sign", "sign", 1, F64, jsmath::sign),
        spec!("sqrt", "sqrt", 1, F64, jsmath::sqrt),
        spec!("max2", "max2", 2, F64, jsmath::max),
        spec!("min2", "min2", 2, F64, jsmath::min),
        spec!("hypot2", "hypot2", 2, F64, jsmath::hypot2),
        spec!("rem", "rem", 2, F64, jsmath::rem),
        spec!("imul", "imul", 2, I32, jsmath::imul),
        spec!("to_int32", "to_int32", 1, I32, jsmath::to_int32),
        spec!("to_uint32", "to_uint32", 1, U32, jsmath::to_uint32),
        spec!("shl", "shl", 2, I32, jsmath::shl),
        spec!("shr", "shr", 2, I32, jsmath::shr),
        spec!("ushr", "ushr", 2, U32, jsmath::ushr),
        spec!("sin", "sin", 1, F64, jsmath::sin),
        spec!("cos", "cos", 1, F64, jsmath::cos),
        spec!("tanh", "tanh", 1, F64, jsmath::tanh),
        spec!("atan", "atan", 1, F64, jsmath::atan),
        spec!("exp", "exp", 1, F64, jsmath::exp),
        spec!("log", "log", 1, F64, jsmath::log),
        spec!("atan2", "atan2", 2, F64, jsmath::atan2),
        spec!("pow", "pow", 2, F64, jsmath::pow),
        FnSpec {
            name: "opp_seed_next",
            op: "opp_seed_next",
            arity: 1,
            ret: Ret::U32,
            // `a[0]` is always an exact, non-negative, u32-range integer (see `probe_value`'s
            // special case for this op), so `as u32` is a lossless, exact cast, not a truncation.
            rust: |a| RustVal::U32(jsmath::opp_seed_next(a[0] as u32)),
        },
    ]
}

/// Supplementary n-ary probes for `hypot`/`max`/`min` (n != 2), a smaller run (not part of the
/// ">= 10^6 per function" budget, which the 2-argument forms above already satisfy — these check
/// the n-ary *reduction* itself, whose per-step arithmetic is the same already-probed 2-arg code).
struct NArySpec {
    name: &'static str,
    op: &'static str,
    arity: usize,
    rust: fn(&[f64]) -> f64,
}

fn n_ary_table() -> Vec<NArySpec> {
    let mut v = Vec::new();
    for &n in &[0usize, 1, 3, 4, 5, 8] {
        v.push(NArySpec {
            name: Box::leak(format!("hypot_n{n}").into_boxed_str()),
            op: "hypot_n",
            arity: n,
            rust: |xs| jsmath::hypot(xs),
        });
        v.push(NArySpec {
            name: Box::leak(format!("max_n{n}").into_boxed_str()),
            op: "max_n",
            arity: n,
            rust: |xs| jsmath::max_n(xs),
        });
        v.push(NArySpec {
            name: Box::leak(format!("min_n{n}").into_boxed_str()),
            op: "min_n",
            arity: n,
            rust: |xs| jsmath::min_n(xs),
        });
    }
    v
}

fn f64_matches(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn rust_val_matches(rust: &RustVal, expected_bits: u64) -> bool {
    match rust {
        RustVal::F64(v) => f64_matches(*v, f64::from_bits(expected_bits)),
        RustVal::I32(v) => (*v as u32 as u64) == (expected_bits & 0xFFFF_FFFF),
        RustVal::U32(v) => (*v as u64) == (expected_bits & 0xFFFF_FFFF),
    }
}

// ---------------------------------------------------------------------------------------------
// Node invocation.
// ---------------------------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    // crates/ddai-jsmath/tests/oracle.rs -> repo root is 3 levels up from CARGO_MANIFEST_DIR.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn node_bin() -> String {
    std::env::var("JSMATH_NODE").unwrap_or_else(|_| "node".to_string())
}

/// Runs `tools/jsmath-oracle/probe.mjs` over every scalar function in `table` at `count_for(op)`
/// probes each (plus `n_ary` at their own fixed arities, `n_ary_rows` each), writing/reading
/// through `work_dir`. Returns, per function name, the real-V8 result bits (`f64`/`i32`/`u32`
/// widened to `u64`) and the inputs used (so callers can both compare against Rust and, for the
/// fixture writer, persist only the outputs).
fn run_scalar_oracle(
    table: &[FnSpec],
    n_ary: &[NArySpec],
    count_for: impl Fn(&str) -> u64,
    n_ary_rows: u64,
    seed: u64,
    work_dir: &Path,
) -> (
    std::collections::HashMap<String, Vec<f64>>,
    std::collections::HashMap<String, Vec<u64>>,
) {
    fs::create_dir_all(work_dir).unwrap();
    let probes_dir = work_dir.join("probes");
    let results_dir = work_dir.join("results");
    fs::create_dir_all(&probes_dir).unwrap();
    fs::create_dir_all(&results_dir).unwrap();

    let mut inputs_by_fn: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let mut entries: Vec<String> = Vec::new();

    for (i, f) in table.iter().enumerate() {
        let count = count_for(f.op);
        let mut rng = Prng::new(seed ^ (i as u64).wrapping_mul(0x1000_0001));
        let mut inputs = Vec::with_capacity((count as usize) * f.arity);
        for row in 0..count {
            for arg in 0..f.arity {
                inputs.push(probe_value(f.op, &mut rng, row, arg));
            }
        }
        write_f64_file(&probes_dir.join(format!("{}.in.f64", f.name)), &inputs);
        entries.push(fn_manifest_entry(f.name, f.op, f.arity, ret_str(f.ret), count));
        inputs_by_fn.insert(f.name.to_string(), inputs);
    }
    for (i, f) in n_ary.iter().enumerate() {
        let mut rng = Prng::new(seed ^ 0xA5A5_0000 ^ (i as u64).wrapping_mul(0x1000_0001));
        let rows: u64 = n_ary_rows;
        let mut inputs = Vec::with_capacity((rows as usize) * f.arity);
        for row in 0..rows {
            for arg in 0..f.arity {
                inputs.push(probe_value(f.op, &mut rng, row, arg));
            }
        }
        write_f64_file(&probes_dir.join(format!("{}.in.f64", f.name)), &inputs);
        entries.push(fn_manifest_entry(f.name, f.op, f.arity, "f64", rows));
        inputs_by_fn.insert(f.name.to_string(), inputs);
    }
    let manifest = format!("{{\"functions\":[{}]}}", entries.join(","));
    let manifest_path = work_dir.join("manifest.json");
    fs::write(&manifest_path, manifest).unwrap();

    let status = Command::new(node_bin())
        .arg(repo_root().join("tools/jsmath-oracle/probe.mjs"))
        .arg(&manifest_path)
        .arg(&probes_dir)
        .arg(&results_dir)
        .status()
        .expect("failed to run node probe.mjs — is Node 24 on PATH? (see tools/jsmath-oracle/README.md)");
    assert!(status.success(), "probe.mjs exited with {status}");

    let mut outputs_by_fn: std::collections::HashMap<String, Vec<u64>> = std::collections::HashMap::new();
    for f in table.iter() {
        let bits = read_result_file(
            &results_dir.join(format!("{}.out.{}", f.name, ret_str(f.ret))),
            ret_str(f.ret),
        );
        outputs_by_fn.insert(f.name.to_string(), bits);
    }
    for f in n_ary.iter() {
        let bits = read_result_file(&results_dir.join(format!("{}.out.f64", f.name)), "f64");
        outputs_by_fn.insert(f.name.to_string(), bits);
    }

    (inputs_by_fn, outputs_by_fn)
}

fn ret_str(r: Ret) -> &'static str {
    match r {
        Ret::F64 => "f64",
        Ret::I32 => "i32",
        Ret::U32 => "u32",
    }
}

fn fn_manifest_entry(name: &str, op: &str, arity: usize, ret: &str, count: u64) -> String {
    format!("{{\"name\":\"{name}\",\"op\":\"{op}\",\"arity\":{arity},\"ret\":\"{ret}\",\"count\":{count}}}")
}

fn write_f64_file(path: &Path, values: &[f64]) {
    let mut file = fs::File::create(path).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    file.write_all(&bytes).unwrap();
}

fn read_result_file(path: &Path, ret: &str) -> Vec<u64> {
    let bytes = fs::read(path).unwrap();
    match ret {
        "f64" => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect(),
        "i32" | "u32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c) as u64)
            .collect(),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------------------------
// Rng oracle (against the real `src/nn/rng.ts`, run by Node).
// ---------------------------------------------------------------------------------------------

fn run_rng_oracle(num_seeds: usize, draws: usize, seed: u64, work_dir: &Path) -> (Vec<u32>, Vec<f64>) {
    fs::create_dir_all(work_dir).unwrap();
    let mut rng = Prng::new(seed);
    let seeds: Vec<u32> = (0..num_seeds).map(|_| rng.next_u32()).collect();
    let seeds_path = work_dir.join("seeds.u32");
    let mut f = fs::File::create(&seeds_path).unwrap();
    let bytes: Vec<u8> = seeds.iter().flat_map(|s| s.to_le_bytes()).collect();
    f.write_all(&bytes).unwrap();

    let out_path = work_dir.join("rng.out.f64");
    let status = Command::new(node_bin())
        .arg(repo_root().join("tools/jsmath-oracle/rng_probe.mjs"))
        .arg(&seeds_path)
        .arg(draws.to_string())
        .arg(&out_path)
        .status()
        .expect("failed to run node rng_probe.mjs");
    assert!(status.success(), "rng_probe.mjs exited with {status}");

    let bytes = fs::read(&out_path).unwrap();
    let expected: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_bits(u64::from_le_bytes(*c)))
        .collect();
    (seeds, expected)
}

/// Draw-index -> which `Rng` method to call, shared with `rng_probe.mjs`'s comment (must match).
fn rng_op_for_draw(i: usize) -> u8 {
    (i % 5) as u8
}

fn rng_sequence(seed: u32, draws: usize) -> Vec<f64> {
    let mut rng = jsmath::Rng::new(seed);
    (0..draws)
        .map(|i| match rng_op_for_draw(i) {
            0 | 1 => f64::from(rng.next_u32()),
            2 | 3 => rng.next_float(),
            _ => rng.next_gaussian(),
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Golden fixture (compact, committed, no Node needed to check it).
// ---------------------------------------------------------------------------------------------

const FIXTURE_SEED: u64 = 0xD00D_C0DE_1337_F00D;
// F2 fix: the golden fixture is sized in two tiers instead of one flat count (it was 900/700 by
// arity before, which wasn't actually about arity at all — every function's fixture record is one
// 8-byte output regardless of arity, since inputs are regenerated deterministically, not stored).
// A 1-ulp error in a single polynomial-coefficient constant (e.g. `log`'s `LG1`, `sin`/`cos`'s
// `INVPIO2`) only flips the final rounded bit for a small fraction of inputs (review's own
// mutation numbers: ~35/200,000 for `LG1`, ~283/200,000 for `INVPIO2` — roughly 1 in 700-6,000),
// so a flat few-thousand-row fixture has a real chance of drawing zero affected rows purely by
// luck, even though the function-under-test is wrong. Rather than inflate every function's row
// count to the same high number (26 functions at, say, 10,000 rows each would alone be ~1.7 MB,
// blowing the < 1 MB budget), the 8 transcendentals actually built from fdlibm polynomial/table
// constants (where this class of bug lives) get an order of magnitude more rows than the other 18
// "exact" functions (round/trunc/imul/to_int32/...), which have no rounding ambiguity to begin
// with — a wrong constant there would be a logic bug the < 1000-row tier already catches
// trivially, not a last-bit statistical needle. See `fixture_count_for`.
const FIXTURE_COUNT_TRANSCENDENTAL: u64 = 12_000;
const FIXTURE_COUNT_EXACT: u64 = 500;

/// The 8 functions built from fdlibm polynomial/table constants — see [`FIXTURE_COUNT_TRANSCENDENTAL`].
const FIXTURE_RISKY_OPS: [&str; 8] = ["sin", "cos", "tanh", "atan", "exp", "log", "atan2", "pow"];

fn fixture_count_for(op: &str) -> u64 {
    if FIXTURE_RISKY_OPS.contains(&op) {
        FIXTURE_COUNT_TRANSCENDENTAL
    } else {
        FIXTURE_COUNT_EXACT
    }
}
// F2 fix: the golden fixture previously covered *no* n-ary hypot/max/min variant at all, so a
// bug specific to the n>=3 general Kahan-loop path (confirmed by review mutation testing: e.g.
// dropping the `- compensation` term) passed it silently. n = 3, 4, 5 are enough to exercise that
// path (the n<=3 fast-path formulas are covered by the `hypot2`/`max2`/`min2` rows above and are
// provably identical to n<=3 of the general loop anyway, see `hypot`'s doc comment).
const FIXTURE_NARY_COUNT: u64 = 400;
const FIXTURE_RNG_SEEDS: usize = 150;
const FIXTURE_RNG_DRAWS: usize = 60;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The n-ary `hypot`/`max`/`min` variants the golden fixture covers (a subset of
/// [`n_ary_table`]'s n=0,1,3,4,5,8 — the fixture only needs the ones that actually exercise the
/// general (`n > 3`... and, deliberately, `n == 3` too, right at the boundary) Kahan-loop path;
/// n=0/1 are trivial and n=8 adds size without adding coverage beyond what n=3..5 already give).
fn fixture_n_ary_table() -> Vec<NArySpec> {
    n_ary_table().into_iter().filter(|f| matches!(f.arity, 3..=5)).collect()
}

/// Regenerates `tests/fixtures/golden.bin`/`golden.sha256` from real V8. Run by a human with
/// `cargo test -p ddai-jsmath --test oracle -- --ignored regenerate_golden_fixture`.
#[test]
#[ignore]
fn regenerate_golden_fixture() {
    let table = function_table();
    let n_ary = fixture_n_ary_table();
    let work_dir = jsmath_scratch_dir().join("fixture-gen");
    let (_inputs, outputs) = run_scalar_oracle(
        &table,
        &n_ary,
        fixture_count_for,
        FIXTURE_NARY_COUNT,
        FIXTURE_SEED,
        &work_dir,
    );

    let mut blob = Vec::new();
    for f in &table {
        for &b in &outputs[f.name] {
            blob.extend_from_slice(&b.to_le_bytes());
        }
    }
    for f in &n_ary {
        for &b in &outputs[f.name] {
            blob.extend_from_slice(&b.to_le_bytes());
        }
    }

    let (seeds, rng_expected) = run_rng_oracle(FIXTURE_RNG_SEEDS, FIXTURE_RNG_DRAWS, FIXTURE_SEED, &work_dir);
    assert_eq!(seeds.len(), FIXTURE_RNG_SEEDS);
    for v in &rng_expected {
        blob.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    fs::create_dir_all(fixture_dir()).unwrap();
    fs::write(fixture_dir().join("golden.bin"), &blob).unwrap();
    let sha = sha256_hex(&blob);
    fs::write(fixture_dir().join("golden.sha256"), format!("{sha}  golden.bin\n")).unwrap();
    eprintln!("regenerate_golden_fixture: wrote {} bytes, sha256 {sha}", blob.len());
}

#[test]
fn golden_fixture_matches_v8() {
    let table = function_table();
    let n_ary = fixture_n_ary_table();
    let bin_path = fixture_dir().join("golden.bin");
    let sha_path = fixture_dir().join("golden.sha256");
    let blob = fs::read(&bin_path).unwrap_or_else(|e| {
        panic!(
            "reading {}: {e} (run `regenerate_golden_fixture` first)",
            bin_path.display()
        )
    });
    let recorded_sha = fs::read_to_string(&sha_path).unwrap();
    let recorded_sha = recorded_sha.split_whitespace().next().unwrap();
    assert_eq!(
        sha256_hex(&blob),
        recorded_sha,
        "golden.bin does not match golden.sha256 (corrupted?)"
    );

    let mut cursor = 0usize;
    let mut mismatches = Vec::new();
    for f in &table {
        let mut rng = Prng::new(FIXTURE_SEED ^ (fn_index(&table, f.name) as u64).wrapping_mul(0x1000_0001));
        for row in 0..fixture_count_for(f.op) {
            let mut args = Vec::with_capacity(f.arity);
            for arg in 0..f.arity {
                args.push(probe_value(f.op, &mut rng, row, arg));
            }
            let expected_bits = u64::from_le_bytes(blob[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            let got = (f.rust)(&args);
            if !rust_val_matches(&got, expected_bits) {
                mismatches.push(format!("{}({args:?}) row {row}", f.name));
            }
        }
    }
    // n-ary section: same seeding scheme `run_scalar_oracle` uses for its `n_ary` slice
    // (`seed ^ 0xA5A5_0000 ^ (i * 0x10000001)`, `i` = position within *this* subset).
    for (i, f) in n_ary.iter().enumerate() {
        let mut rng = Prng::new(FIXTURE_SEED ^ 0xA5A5_0000 ^ (i as u64).wrapping_mul(0x1000_0001));
        for row in 0..FIXTURE_NARY_COUNT {
            let mut args = Vec::with_capacity(f.arity);
            for arg in 0..f.arity {
                args.push(probe_value(f.op, &mut rng, row, arg));
            }
            let expected_bits = u64::from_le_bytes(blob[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            let got = (f.rust)(&args);
            if !f64_matches(got, f64::from_bits(expected_bits)) {
                mismatches.push(format!("{}({args:?}) row {row}", f.name));
            }
        }
    }
    assert!(
        mismatches.len() <= 20,
        "{} mismatches, first 20: {:#?}",
        mismatches.len(),
        &mismatches[..20]
    );
    assert!(
        mismatches.is_empty(),
        "{} mismatches: {:#?}",
        mismatches.len(),
        mismatches
    );

    let mut rng_mismatches = 0usize;
    let mut seed_rng = Prng::new(FIXTURE_SEED);
    for _ in 0..FIXTURE_RNG_SEEDS {
        let seed = seed_rng.next_u32();
        let got = rng_sequence(seed, FIXTURE_RNG_DRAWS);
        for v in got {
            let expected_bits = u64::from_le_bytes(blob[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            if !f64_matches(v, f64::from_bits(expected_bits)) {
                rng_mismatches += 1;
            }
        }
    }
    assert_eq!(
        rng_mismatches, 0,
        "{rng_mismatches} Rng mismatches against the golden fixture"
    );
    assert_eq!(
        cursor,
        blob.len(),
        "golden.bin has trailing bytes the reader never consumed"
    );
}

fn fn_index(table: &[FnSpec], name: &str) -> usize {
    table.iter().position(|f| f.name == name).unwrap()
}

fn sha256_hex(data: &[u8]) -> String {
    // Minimal, self-contained SHA-256 (no dependency): this crate has zero runtime deps and this
    // is dev/test-only code, but adding a crate just for a fixture checksum still isn't worth it.
    let hash = sha256(data);
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Where the (large, disposable) probe/result files for the full oracle and fixture regeneration
/// go. F7 fix: used to default to a path hardcoded to this one deployment
/// (`~/aiddnet/data/research/jsmath-scratch/`); defaults to the OS temp dir now, so this test
/// file works unmodified on any machine — set `JSMATH_SCRATCH_DIR` to use this project's own
/// scratch convention instead (see `tools/jsmath-oracle/README.md`).
fn jsmath_scratch_dir() -> PathBuf {
    std::env::var("JSMATH_SCRATCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("ddai-jsmath-oracle"))
}

// ---------------------------------------------------------------------------------------------
// The full, >= 10^6-probes-per-function run against real V8. `#[ignore]`d: run explicitly via
// `tools/jsmath-oracle/run.sh` (or `cargo test -p ddai-jsmath --test oracle -- --ignored
// full_oracle_vs_real_v8 --nocapture`).
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore]
fn full_oracle_vs_real_v8() {
    let count: u64 = std::env::var("JSMATH_PROBES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let table = function_table();
    let n_ary = n_ary_table();
    let work_dir = jsmath_scratch_dir().join("full-run");

    let n_ary_rows: u64 = std::env::var("JSMATH_NARY_PROBES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let start = std::time::Instant::now();
    let (inputs, outputs) = run_scalar_oracle(&table, &n_ary, |_op| count, n_ary_rows, 0x5EED_0001, &work_dir);
    eprintln!("full_oracle_vs_real_v8: node probe run took {:?}", start.elapsed());

    let mut report = String::new();
    let mut any_fail = false;
    for f in &table {
        let ins = &inputs[f.name];
        let outs = &outputs[f.name];
        let mut mismatches = 0u64;
        let mut first_examples = Vec::new();
        for row in 0..count as usize {
            let args = &ins[row * f.arity..row * f.arity + f.arity];
            let got = (f.rust)(args);
            if !rust_val_matches(&got, outs[row]) {
                mismatches += 1;
                if first_examples.len() < 5 {
                    first_examples.push(format!("{}({args:?})", f.name));
                }
            }
        }
        report.push_str(&format!(
            "{:<14} probes={:<9} mismatches={}\n",
            f.name, count, mismatches
        ));
        if mismatches > 0 {
            any_fail = true;
            report.push_str(&format!("  examples: {first_examples:?}\n"));
        }
    }
    for f in &n_ary {
        let ins = &inputs[f.name];
        let outs = &outputs[f.name];
        let rows = ins.len().checked_div(f.arity).unwrap_or(outs.len());
        let mut mismatches = 0u64;
        for row in 0..rows {
            let args = if f.arity == 0 {
                &[][..]
            } else {
                &ins[row * f.arity..row * f.arity + f.arity]
            };
            let got = (f.rust)(args);
            if !f64_matches(got, f64::from_bits(outs[row])) {
                mismatches += 1;
            }
        }
        report.push_str(&format!(
            "{:<14} probes={:<9} mismatches={}\n",
            f.name, rows, mismatches
        ));
        any_fail |= mismatches > 0;
    }

    let (seeds, rng_expected) = run_rng_oracle(100_000, 1_000, 0x5EED_0002, &work_dir);
    let mut rng_mismatches: u64 = 0;
    for (s, &seed) in seeds.iter().enumerate() {
        let got = rng_sequence(seed, 1_000);
        let expected = &rng_expected[s * 1_000..s * 1_000 + 1_000];
        for i in 0..1_000 {
            if !f64_matches(got[i], expected[i]) {
                rng_mismatches += 1;
            }
        }
    }
    report.push_str(&format!(
        "Rng            seeds=100000 draws/seed=1000 mismatches={rng_mismatches}\n"
    ));
    any_fail |= rng_mismatches > 0;

    eprintln!("{report}");
    assert!(!any_fail, "oracle found mismatches:\n{report}");
}
