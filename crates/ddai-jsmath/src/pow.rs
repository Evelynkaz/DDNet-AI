//! `Math.pow`, ported from `deps/v8/src/numbers/ieee754.cc` (V8 13.6 / Node 24.21.0), **not**
//! from fdlibm — see the module doc comment on why this one function's port is different from
//! every other transcendental in this crate.
//!
//! ```cc
//! double pow(double x, double y) {
//!   if (v8_flags.use_std_math_pow) {
//!     if (std::isnan(y)) return NaN;
//!     if (std::isinf(y) && (x == 1 || x == -1)) return NaN;
//!     if (std::isnan(x)) x = NaN;  // quiet
//!     if (y == 2) return x * x;
//!     if (y == 0.5) {
//!       if (std::isinf(x)) return Infinity;
//!       return std::sqrt(x + 0);
//!     }
//!     return std::pow(x, y);
//!   }
//!   return base::ieee754::legacy::pow(x, y);
//! }
//! ```
//!
//! `v8_flags.use_std_math_pow` defaults to `true`
//! (`deps/v8/src/flags/flag-definitions.h:1029`, unconditional — no build-time guard), so real V8
//! takes the `std::pow` branch — so, unlike `sin`/`cos`/`log`/`atan`/`atan2`/`exp`/`tanh`, this
//! function needs no fdlibm port at all, only the three pre-checks and two special cases V8 adds
//! on top of the raw `pow` call.
//!
//! **Not the same symbol, verified empirically instead (review finding, round 1):** `std::pow`
//! (Node) and `f64::powf` (Rust) do **not** resolve to the identical versioned glibc symbol on
//! this machine — `objdump -T` shows Node linking `pow@GLIBC_2.2.5` (glibc's ABI-compat wrapper)
//! and this crate's own binaries linking `pow@GLIBC_2.29`. Bit-exactness here rests on the
//! oracle's empirical result (0/10⁶ mismatches, both for general `pow(x, y)` and for the
//! literal-constant-argument shapes in `tests/pow_literal_args.rs`), not on a "same symbol"
//! argument — see the crate README "Почему `pow` — не fdlibm" for the full correction.
//!
//! **A second, unrelated risk this function *does* need to guard against:** `f64::powf` compiles
//! to the `llvm.pow.f64` intrinsic, which LLVM can rewrite for constant operands even without
//! `-ffast-math` (`pow(2^n, y) -> exp2(n*y)`, `pow(x, -1.0) -> 1.0/x`), diverging from a genuine
//! runtime glibc call — see the `black_box` calls below and their doc comment.

/// `Math.pow(x, y)`.
pub fn pow(x: f64, y: f64) -> f64 {
    if y.is_nan() {
        // 1. If exponent is NaN, return NaN.
        return f64::NAN;
    }
    if y.is_infinite() && (x == 1.0 || x == -1.0) {
        // base**(+-Infinity) is NaN when |base| == 1 (historical ECMAScript behavior, kept for
        // compatibility even though later IEEE 754-2008 editions specify 1 here instead).
        return f64::NAN;
    }
    // `std::pow` distinguishes signaling/quiet NaN; JS doesn't, so V8 quiets `x` first. Rust
    // has no separate signaling-NaN input path to normalize here (there's nothing upstream of
    // this function that could hand us a signaling NaN bit pattern that behaves differently from
    // a quiet one in the arithmetic below), so this step is a no-op in practice, kept only for a
    // literal correspondence with the source.
    let x = if x.is_nan() { f64::NAN } else { x };

    if y == 2.0 {
        // x ** 2 ==> x * x (matches what optimizing compilers do instead of calling `pow`).
        return x * x;
    }
    if y == 0.5 {
        // x ** 0.5 ==> sqrt(x), except that sqrt(-Infinity) would be NaN, not +Infinity.
        if x.is_infinite() {
            return f64::INFINITY;
        }
        // The `+ 0` gives `+0` for `(-0) ** 0.5` rather than `-0` (sqrt(-0) == -0 otherwise).
        return (x + 0.0).sqrt();
    }
    // `black_box` on both operands is load-bearing, not defensive styling: `f64::powf` compiles
    // to the `llvm.pow.f64` intrinsic, and LLVM's `TargetLibraryInfo`/instcombine recognize it (by
    // *name*, the same way it would recognize a hand-written `extern "C" fn pow`, not something
    // special to Rust) and rewrite it for certain constant operands — `pow(2^n, y)` becomes
    // `exp2(n*y)`, `pow(x, -1.0)` becomes `1.0/x`, `pow(x, 2.0)` becomes `x*x` — *without* any
    // `-ffast-math`/fast-math flag, because these rewrites are (mostly) value-preserving in exact
    // arithmetic, just not bit-identical to a real runtime call into glibc's `pow`. `x`/`y` here
    // are function parameters with no literal of their own, but if a caller writes something like
    // `jsmath::pow(2.0, p)` and this function gets inlined into it (plausible: it is small, and
    // the workspace release profile enables thin LTO across crates), the compiler can see `x ==
    // 2.0` post-inlining just as well as if the literal were written right here, and apply the
    // same rewrite — independently confirmed against real V8: unprotected, `pow(2.0, y)` disagreed
    // with `Math.pow(2, y)` on 216/200,000 random `y`; wrapping both operands in `black_box`
    // (which is specifically designed to defeat exactly this kind of interprocedural constant
    // propagation into an intrinsic) brought that to 0/200,000. See
    // `tests/pow_literal_args.rs` for the regression test and the crate README for the numbers.
    core::hint::black_box(x).powf(core::hint::black_box(y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn matches_known_points() {
        assert_eq!(pow(2.0, 10.0), 1024.0);
        assert_eq!(pow(0.0, 0.0), 1.0);
        // `pow(x, +-0)` is `1` for *any* `x`, even `NaN` — an IEEE-754 `pow` special case that
        // both `std::pow` and `f64::powf` implement without V8 needing an explicit check for it
        // (verified against real Node: `Math.pow(NaN, 0) === 1`).
        assert_eq!(pow(f64::NAN, 0.0), 1.0);
        assert_eq!(pow(f64::NAN, -0.0), 1.0);
        assert!(pow(2.0, f64::NAN).is_nan());
    }

    #[test]
    fn base_one_or_minus_one_to_infinite_exponent_is_nan() {
        assert!(pow(1.0, f64::INFINITY).is_nan());
        assert!(pow(-1.0, f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn special_cased_exponents() {
        assert_eq!(pow(3.0, 2.0), 9.0);
        assert_eq!(pow(4.0, 0.5), 2.0);
        assert_eq!(pow(f64::NEG_INFINITY, 0.5), f64::INFINITY);
        assert!(!pow(-0.0, 0.5).is_sign_negative());
    }

    /// Review finding F9: `tests/pow_literal_args.rs`'s oracle only ever runs `--release`
    /// (`cargo test`'s default dev profile is `opt-level = 1` for workspace crates — see the
    /// workspace `Cargo.toml` — at which LLVM doesn't inline `pow` into its callers, so it never
    /// gets the chance to see a literal operand and rewrite the `llvm.pow.f64` intrinsic; F1's bug
    /// (and this guard against it) both need actual inlining+constant-folding to reproduce). This
    /// test lives *inside* the crate specifically so `cargo test` (no `--release`, no explicit
    /// target selection) exercises it too — which only works because the workspace `Cargo.toml`
    /// now has `[profile.dev.package.ddai-jsmath] opt-level = 3` (dev-profile *default* is
    /// opt-level 1 for workspace members; the "*" dependency override doesn't reach workspace
    /// members at all — see that file's comment). Values recorded from real V8 (Node 24.21.0 / V8
    /// 13.6.233.17-node.53); same values as `tests/pow_literal_args.rs`'s `BASE2`/`EXP_NEG1`
    /// tables, including the exact points review round 1 found broken.
    #[test]
    fn literal_argument_regression_survives_dev_profile() {
        // The "risky" operand (the base, for the `2^n` cases; the exponent, for the `-1` cases)
        // is a literal right here, at the same call site as the arithmetic — nothing between it
        // and `pow`'s own `black_box` calls that could give the compiler a reason not to treat it
        // as a compile-time constant after inlining. The *other* operand is wrapped in
        // `black_box` right here in the test, deliberately: with both operands as plain literals,
        // LLVM's constant folder evaluates the entire call at compile time (by invoking the same
        // host libm `pow` that a real runtime call would — this was checked empirically: it does
        // *not* reproduce F1 at all, passing even with the crate's own `black_box` fix removed,
        // since there's nothing left to inline-and-then-see-as-constant — there is no runtime
        // call to rewrite). Blackboxing the non-risky operand here forces genuine codegen for the
        // `pow` call while still leaving the risky operand visible to the optimizer as a
        // constant post-inlining, which is exactly the shape a real caller like
        // `jsmath::pow(2.0, p)` (`p` a genuine runtime value) has. `y == 0.5`/`y == 2.0` are
        // excluded from this set: `pow` special-cases those *before* reaching the
        // `black_box`-guarded call (see the function above), so they'd pass regardless of the
        // guard — this test is specifically about the general-case call.
        assert_eq!(
            pow(2.0, black_box(-49.996_151_219_816_89)).to_bits(),
            0x3cd0_0af1_1870_3c44
        );
        assert_eq!(
            pow(4.0, black_box(-49.996_151_219_816_89)).to_bits(),
            0x39b0_15e9_ac6f_f52d
        );
        assert_eq!(pow(0.5, black_box(12.7)).to_bits(), 0x3f23_b2c4_7bff_832c);
        assert_eq!(
            pow(black_box(1.419_473_849_973_027_2e-21), -1.0).to_bits(),
            0x4443_185b_34f0_8d5d
        );
        assert_eq!(pow(black_box(3.0), -1.0).to_bits(), 0x3fd5_5555_5555_5555);
        assert_eq!(pow(black_box(1e300), -1.0).to_bits(), 0x01a5_6e1f_c2f8_f359);
    }
}
