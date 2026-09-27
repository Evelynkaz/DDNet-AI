//! Regression test for review finding F1: `f64::powf` compiles to the `llvm.pow.f64` intrinsic,
//! which LLVM's `TargetLibraryInfo`/instcombine can rewrite for certain *constant* operands even
//! without `-ffast-math` — `pow(2^n, y) -> exp2(n*y)`, `pow(x, -1.0) -> 1.0/x` — differing from a
//! genuine runtime call into glibc's `pow` (which is what both V8 and an unprotected Rust
//! `powf` call would otherwise perform). If a caller writes `jsmath::pow(2.0, p)` and this
//! function gets inlined into them (plausible: it is small, and the workspace release profile
//! enables thin LTO across crates), the compiler can see `x == 2.0` after inlining just as it
//! would from a literal written directly in `pow.rs`, and apply the same rewrite. `pow.rs` fixes
//! this with `core::hint::black_box` on both operands; this file proves the fix at the exact call
//! shape that exposed the bug (a literal constant written directly at the call site, so nothing
//! stops the compiler from inlining and constant-propagating).
//!
//! - `spot_values_match_real_v8_recorded_bits` (always runs, no Node): a handful of `(base, y)` /
//!   `(x, -1)` pairs the review found actually differ without the fix, checked against bits
//!   recorded from real V8 (`node -e 'Buffer...writeDoubleLE(Math.pow(...))'`, Node 24.21.0 / V8
//!   13.6.233.17-node.53).
//! - `literal_args_vs_real_v8` (`#[ignore]`): the same call shapes at 10^6 probes each, calling
//!   `node` directly.
//!
//! **Review finding F9 — this file's dev-profile blind spot, and why the real guard lives
//! elsewhere too:** every test in *this* file calls `ddai_jsmath::pow` from a separate crate (an
//! integration test in `tests/` is its own crate, linking `ddai-jsmath` as a compiled
//! dependency). `pow` has no `#[inline]` attribute, so a plain `cargo test` (dev profile, no
//! LTO) never inlines it across that crate boundary at all, regardless of `opt-level` — so
//! `spot_values_match_real_v8_recorded_bits` above passes whether or not `pow.rs`'s `black_box`
//! fix is present, in the dev profile (confirmed: reverting the fix and running plain `cargo test
//! --test pow_literal_args` still passes; only `--release`, which turns on thin LTO for the whole
//! workspace, makes cross-crate inlining happen and the same revert fail here). That is a real
//! gap for a Node-free, no-`--release`-needed guard — closed by `crates/ddai-jsmath/src/pow.rs`'s
//! own `#[cfg(test)]` unit test (`literal_argument_regression_survives_dev_profile`), which calls
//! `pow` from *inside* its own crate (ordinary same-crate inlining, no LTO needed) and is
//! confirmed to fail without the fix under plain `cargo test` — see that test's doc comment and
//! the workspace `Cargo.toml`'s `[profile.dev.package.ddai-jsmath]` override. This file's own
//! tests remain valuable regardless: they are what actually matches the real-world risk shape
//! (a *different* crate calling `jsmath::pow` with a literal, exactly like `ddai-planner` would),
//! at the same release+LTO settings the shipped binary uses, and at far higher probe counts.

use std::process::Command;

/// `(y, expected_bits)` for `ddai_jsmath::pow(2.0, y)`, `2.0` written as a literal at the call
/// site below — recorded from real V8. Includes the review's own repro point
/// (`y = -49.99615121981689`).
const BASE2: &[(f64, u64)] = &[
    (-49.996_151_219_816_89, 0x3cd0_0af1_1870_3c44),
    (0.3, 0x3ff3_b2c4_7bff_8329),
    (12.7, 0x40b9_fdf8_bcce_533a),
    (-0.0001, 0x3fef_ff6e_a43b_d8ec),
    (33.333, 0x4204_2771_c40e_f69f),
    (-700.5, 0x1426_a09e_667f_3bcd),
    (700.25, 0x6bb3_06fe_0a31_b715),
    (1e-10, 0x3ff0_0000_0004_c366),
    (-1e-10, 0x3fef_ffff_fff6_7935),
    (0.5, 0x3ff6_a09e_667f_3bcd),
    (-0.5, 0x3fe6_a09e_667f_3bcd),
];

/// Same, for `ddai_jsmath::pow(4.0, y)`.
const BASE4: &[(f64, u64)] = &[
    (-49.996_151_219_816_89, 0x39b0_15e9_ac6f_f52d),
    (0.3, 0x3ff8_4060_03b2_ae5c),
    (12.7, 0x4185_1cb4_53b9_5366),
    (-0.0001, 0x3fef_fedd_4b0b_fa80),
    (33.333, 0x4419_62fd_a7ea_1c35),
    (-700.5, 0x0000_0000_0000_0000),
    (700.25, 0x7ff0_0000_0000_0000), // +Infinity
    (1e-10, 0x3ff0_0000_0009_86cb),
    (-1e-10, 0x3fef_ffff_ffec_f269),
    (0.5, 0x4000_0000_0000_0000),
    (-0.5, 0x3fe0_0000_0000_0000),
];

/// Same, for `ddai_jsmath::pow(0.5, y)`.
const BASE_HALF: &[(f64, u64)] = &[
    (-49.996_151_219_816_89, 0x430f_ea2c_bc09_8734),
    (0.3, 0x3fe9_fdf8_bcce_533e),
    (12.7, 0x3f23_b2c4_7bff_832c),
    (-0.0001, 0x3ff0_0048_af2c_3dba),
    (33.333, 0x3dd9_677f_4605_7be5),
    (-700.5, 0x6bb6_a09e_667f_3bcd),
    (700.25, 0x142a_e89f_995a_d3ad),
    (1e-10, 0x3fef_ffff_fff6_7935),
    (-1e-10, 0x3ff0_0000_0004_c366),
    (0.5, 0x3fe6_a09e_667f_3bcd),
    (-0.5, 0x3ff6_a09e_667f_3bcd),
];

/// `(x, expected_bits)` for `ddai_jsmath::pow(x, -1.0)`, `-1.0` written as a literal at the call
/// site below. Includes the review's own repro point (`x = 1.4194738499730272e-21`).
const EXP_NEG1: &[(f64, u64)] = &[
    (3.0, 0x3fd5_5555_5555_5555),
    (7.0, 0x3fc2_4924_9249_2492),
    (1.419_473_849_973_027_2e-21, 0x4443_185b_34f0_8d5d),
    (-3.0, 0xbfd5_5555_5555_5555),
    (0.1, 0x4024_0000_0000_0000),
    (-0.1, 0xc024_0000_0000_0000),
    (1e300, 0x01a5_6e1f_c2f8_f359),
    (1e-300, 0x7e37_e43c_8800_759b),
    (-7.0, 0xbfc2_4924_9249_2492),
];

#[test]
fn spot_values_match_real_v8_recorded_bits() {
    let mut failures = Vec::new();
    for &(y, expected) in BASE2 {
        let got = ddai_jsmath::pow(2.0, y).to_bits(); // `2.0` is a literal right here.
        if got != expected {
            failures.push(format!("pow(2.0, {y}) = {got:#x}, expected {expected:#x}"));
        }
    }
    for &(y, expected) in BASE4 {
        let got = ddai_jsmath::pow(4.0, y).to_bits();
        if got != expected {
            failures.push(format!("pow(4.0, {y}) = {got:#x}, expected {expected:#x}"));
        }
    }
    for &(y, expected) in BASE_HALF {
        let got = ddai_jsmath::pow(0.5, y).to_bits();
        if got != expected {
            failures.push(format!("pow(0.5, {y}) = {got:#x}, expected {expected:#x}"));
        }
    }
    for &(x, expected) in EXP_NEG1 {
        let got = ddai_jsmath::pow(x, -1.0).to_bits(); // `-1.0` is a literal right here.
        if got != expected {
            failures.push(format!("pow({x}, -1.0) = {got:#x}, expected {expected:#x}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------------------------
// Full-scale (#[ignore]) run against real V8, at the same literal-argument call shape.
// ---------------------------------------------------------------------------------------------

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

fn node_bin() -> String {
    std::env::var("JSMATH_NODE").unwrap_or_else(|_| "node".to_string())
}

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Runs `op` (a `probe.mjs` op name taking exactly one `f64`) over `inputs` via real Node, and
/// returns V8's `f64` results.
fn run_node_1arg(op: &str, inputs: &[f64], work_dir: &std::path::Path) -> Vec<f64> {
    std::fs::create_dir_all(work_dir).unwrap();
    let probes_dir = work_dir.join("probes");
    let results_dir = work_dir.join("results");
    std::fs::create_dir_all(&probes_dir).unwrap();
    std::fs::create_dir_all(&results_dir).unwrap();

    let bytes: Vec<u8> = inputs.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    std::fs::write(probes_dir.join(format!("{op}.in.f64")), &bytes).unwrap();

    let manifest = format!(
        "{{\"functions\":[{{\"name\":\"{op}\",\"op\":\"{op}\",\"arity\":1,\"ret\":\"f64\",\"count\":{}}}]}}",
        inputs.len()
    );
    let manifest_path = work_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest).unwrap();

    let status = Command::new(node_bin())
        .arg(repo_root().join("tools/jsmath-oracle/probe.mjs"))
        .arg(&manifest_path)
        .arg(&probes_dir)
        .arg(&results_dir)
        .status()
        .expect("failed to run node probe.mjs");
    assert!(status.success(), "probe.mjs exited with {status}");

    let out_bytes = std::fs::read(results_dir.join(format!("{op}.out.f64"))).unwrap();
    out_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_bits(u64::from_le_bytes(*c)))
        .collect()
}

fn f64_matches(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[test]
#[ignore]
fn literal_args_vs_real_v8() {
    let count: usize = std::env::var("JSMATH_PROBES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    // F7 fix (see oracle.rs's `jsmath_scratch_dir`): OS temp dir by default, not a path hardcoded
    // to one deployment.
    let work_dir = std::env::var("JSMATH_SCRATCH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("ddai-jsmath-oracle-pow-literal"));

    // `y` domain: a mix of the review's own trouble ranges (a wide swing including the exact
    // repro point) and bot-relevant magnitudes.
    let mut rng = Prng(0xF1F1_F1F1_0000_0001);
    let ys: Vec<f64> = (0..count)
        .map(|i| match i % 3 {
            0 => rng.uniform(-100.0, 100.0),
            1 => rng.uniform(-1000.0, 1000.0),
            _ => rng.uniform(-1e-6, 1e-6),
        })
        .collect();
    let mut rng2 = Prng(0xF1F1_F1F1_0000_0002);
    let xs: Vec<f64> = (0..count)
        .map(|i| match i % 4 {
            0 => rng2.uniform(-1e5, 1e5),
            1 => rng2.uniform(-1.0, 1.0),
            2 => f64::from_bits(rng2.next_u64()), // raw bits: subnormals, huge, tiny, etc.
            _ => rng2.uniform(-1e-20, 1e-20),
        })
        .filter(|x| *x != 0.0) // pow(0, -1) is a defined special case, not the bug under test
        .collect();

    let mut report = String::new();
    let mut any_fail = false;

    // Four separate, straight-line loops, each with the base written as a literal directly at
    // the `ddai_jsmath::pow` call site — deliberately not factored into one generic
    // helper/closure per base, so nothing stands between the literal and the call that could give
    // the compiler a reason not to treat it as a compile-time constant.
    macro_rules! check_base {
        ($name:literal, $base:literal) => {{
            let expected = run_node_1arg($name, &ys, &work_dir);
            let mut mismatches = 0u64;
            for (i, &y) in ys.iter().enumerate() {
                let got = ddai_jsmath::pow($base, y);
                if !f64_matches(got, expected[i]) {
                    mismatches += 1;
                }
            }
            report.push_str(&format!(
                "{:<16} probes={:<9} mismatches={mismatches}\n",
                $name,
                ys.len()
            ));
            any_fail |= mismatches > 0;
        }};
    }
    check_base!("pow_base2", 2.0);
    check_base!("pow_base4", 4.0);
    check_base!("pow_base_half", 0.5);
    check_base!("pow_base8", 8.0);

    let expected_neg1 = run_node_1arg("pow_exp_neg1", &xs, &work_dir);
    let mut mismatches_neg1 = 0u64;
    for (i, &x) in xs.iter().enumerate() {
        let got = ddai_jsmath::pow(x, -1.0); // `-1.0` is a literal right here.
        if !f64_matches(got, expected_neg1[i]) {
            mismatches_neg1 += 1;
        }
    }
    report.push_str(&format!(
        "pow_exp_neg1     probes={:<9} mismatches={mismatches_neg1}\n",
        xs.len()
    ));
    any_fail |= mismatches_neg1 > 0;

    eprintln!("{report}");
    assert!(!any_fail, "literal-argument pow mismatches:\n{report}");
}
