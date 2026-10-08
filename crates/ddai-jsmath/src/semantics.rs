//! JS numeric primitives whose *values* coincide with plain IEEE-754 (so no port from V8 source
//! is needed for the arithmetic itself), but whose *edge-case semantics* — tie-breaking, NaN
//! propagation, signed zero, integer coercion width and multi-argument reduction order — are
//! specific to ECMA-262 and V8's implementation of it. Each function below cites the exact V8
//! source it was derived from; see the crate README for how the citations were found and how
//! they were checked against the real V8/Node 24 binary (`tools/jsmath-oracle`).

/// `Math.PI`. There is exactly one `f64` nearest to the mathematical constant π, so this is
/// trivially bit-identical to V8's `Math.PI` (and to Rust's own `std::f64::consts::PI`, re-quoted
/// here as a `jsmath` constant so callers that want "the V8 constant" don't have to know that).
pub const PI: f64 = std::f64::consts::PI;

/// `Math.round`. V8 does **not** use "round half away from zero" or "round half to even" — it
/// rounds ties *toward +Infinity*, and preserves `-0` for inputs in `(-1, -0.5]`.
///
/// Ported from `CodeStubAssembler::Float64Round`
/// (`deps/v8/src/codegen/code-stub-assembler.cc:408-421` at the Node 24.21.0 / V8 13.6 tag; see
/// `deps/v8/src/builtins/math.tq`'s `MathRound` builtin, which calls this via the `Float64Round`
/// extern macro): `let r = x.ceil(); if r - 0.5 <= x { r } else { r - 1.0 }`.
///
/// Examples the acceptance criteria call out explicitly: `round(-0.4) == -0.0` (`ceil(-0.4) ==
/// -0.0`, and `-0.0 - 0.5 == -0.5 <= -0.4`, so the `-0.0` from `ceil` is returned unmodified —
/// this is *why* `-0` survives); `round(0.49999999999999994) == 0.0` (`ceil` gives `1.0`, but
/// `1.0 - 0.5 == 0.5` is *not* `<= 0.49999999999999994`, so `1.0 - 1.0 == 0.0` is returned).
pub fn round(x: f64) -> f64 {
    let r = x.ceil();
    if r - 0.5 <= x { r } else { r - 1.0 }
}

/// `Math.trunc`. Plain IEEE-754 truncation (round toward zero) — identical to Rust's
/// [`f64::trunc`] for every input, including signed zero, infinities and NaN. `math.tq`'s
/// `MathTrunc` builtin calls the `Float64Trunc` extern macro, which lowers to a hardware
/// round-toward-zero instruction (or an equivalent bit-exact software fallback) with no JS-level
/// quirk on top; there is nothing to port.
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

/// `Math.floor`. Same story as [`trunc`]: plain IEEE-754 floor, identical to [`f64::floor`].
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// `Math.ceil`. Same story as [`trunc`]: plain IEEE-754 ceiling, identical to [`f64::ceil`].
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// `Math.abs`. Plain IEEE-754 absolute value (clear the sign bit) — identical to [`f64::abs`],
/// including for NaN (the sign bit is cleared, the payload is preserved) and `-0` (`abs(-0) ==
/// +0`). `math.tq`'s `MathAbs` calls the `Float64Abs` extern macro.
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// `Math.sign`. NaN-preserving and signed-zero-preserving: unlike a naive `if x < 0 { -1 } else
/// if x > 0 { 1 } else { 0 }`, this returns the *original* value (not a fresh `+0.0`/`-0.0`
/// literal) when `x` is `0`, `-0` or `NaN`, so `sign(-0.0).is_sign_negative()` and
/// `sign(f64::NAN).is_nan()` both hold.
///
/// Ported from `math.tq`'s `MathSign` builtin (`deps/v8/src/builtins/math.tq`, "ES6
/// #sec-math.sign"): `if (value < 0) { -1 } else if (value > 0) { 1 } else { num }` (`num` is the
/// ToNumber-converted but otherwise unmodified input).
pub fn sign(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        x
    }
}

/// `Math.sqrt`. IEEE-754 square root is correctly rounded by definition (there is only one
/// correct result), so this is trivially bit-identical to V8's `Float64Sqrt` for every input —
/// identical to Rust's [`f64::sqrt`].
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// The two-argument `Math.max` reduction step. NaN-propagating (if either operand is NaN, the
/// result is NaN); on a tie (`a == b`, which includes `+0.0 == -0.0`), returns `b` if `a` is the
/// negative one of the pair, `a` otherwise — so `js_max(-0.0, 0.0) == 0.0` and `js_max(0.0, -0.0)
/// == 0.0` (i.e. `Math.max` always prefers `+0` over `-0`, regardless of argument order).
///
/// Ported from the x64 codegen for the `Float64Max` machine op
/// (`deps/v8/src/compiler/backend/x64/code-generator-x64.cc`, case `kSSEFloat64Max`: `ucomisd`
/// then, on the equal case, check the *first* operand's sign bit) — this op is what
/// `math.tq`'s `MathMax` builtin folds over its arguments (`result = Float64Max(result,
/// doubleValue)`, starting `result` at `-Infinity`), so `a` below is always the accumulator
/// (earlier arguments) and `b` the new one, matching that fold's argument order exactly. See
/// [`max_n`] for the full left-to-right multi-argument reduction (equivalent to direct 2-argument
/// `Math.max(x, y)` too, since folding from `-Infinity` is a no-op for the first real argument).
pub fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else if a < b || a.is_sign_negative() {
        // `a < b`, or a tie (`a == b`) where `a` is the negative one of the pair.
        b
    } else {
        a
    }
}

/// The two-argument `Math.min` reduction step — the mirror image of [`js_max`]: on a tie, returns
/// `b` if `b` is the negative one of the pair (so `js_min` always prefers `-0` over `+0`,
/// regardless of argument order: `js_min(0.0, -0.0) == -0.0` and `js_min(-0.0, 0.0) == -0.0`).
///
/// Ported from the x64 codegen for `Float64Min` (same file as [`js_max`], case `kSSEFloat64Min`:
/// on the equal case, checks the *second* operand's sign bit — asymmetric with `Float64Max` on
/// purpose, that is what the hardware sequence does). See [`min_n`] for the full reduction.
pub fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else if a > b || b.is_sign_negative() {
        // `a > b`, or a tie (`a == b`) where `b` is the negative one of the pair.
        b
    } else {
        a
    }
}

/// `Math.max(...xs)` for any number of arguments (including zero, matching `Math.max() ===
/// -Infinity`), by left-to-right folding [`js_max`] starting from `-Infinity` — exactly
/// `math.tq`'s `MathMax` builtin's loop.
pub fn max_n(xs: &[f64]) -> f64 {
    xs.iter().fold(f64::NEG_INFINITY, |acc, &x| js_max(acc, x))
}

/// `Math.min(...xs)` for any number of arguments (including zero, matching `Math.min() ===
/// +Infinity`), by left-to-right folding [`js_min`] starting from `+Infinity` — exactly
/// `math.tq`'s `MathMin` builtin's loop.
pub fn min_n(xs: &[f64]) -> f64 {
    xs.iter().fold(f64::INFINITY, |acc, &x| js_min(acc, x))
}

/// `x | 0` / the ECMA-262 `ToInt32` abstract operation. NaN, `±Infinity` and out-of-range
/// magnitudes map to the low 32 bits of the (conceptually infinite-precision) integer part of
/// `x`, taken modulo 2^32 and reinterpreted as signed; values already in `i32` range convert
/// directly (truncating toward zero).
///
/// Ported from `DoubleToInt32` (`deps/v8/src/numbers/conversions-inl.h:142-161`), which V8's
/// bytecode/Torque `ToInt32` coercions call. `base::Double::Exponent()`/`Significand()`/`Sign()`
/// (`deps/v8/src/base/numbers/double.h`) are inlined here as plain bit masks on `x.to_bits()`
/// instead of going through a `Double` wrapper type — same operations, no behavioral difference.
pub fn to_int32(x: f64) -> i32 {
    // Fast path, identical to V8's: values that already fit in `i32` range convert directly.
    // `i32::MIN`/`MAX` as `f64` are exact (both well within the 53-bit mantissa), so this
    // comparison is exact and `as i32` below is a plain truncating conversion, matching C++'s
    // `static_cast<int32_t>` for values known to be in range.
    if x.is_finite() && x >= i32::MIN as f64 && x <= i32::MAX as f64 {
        return x as i32;
    }

    let bits = x.to_bits();
    let sign: i64 = if (bits >> 63) == 0 { 1 } else { -1 };
    // Biased IEEE-754 double exponent (11 bits) minus the bias (1023) minus the mantissa width
    // (52) = `base::Double::kExponentBias` (1075); `base::Double::Exponent()` special-cases
    // denormals (biased exponent field 0) to the fixed value `kDenormalExponent = -1074`.
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;
    let is_denormal = biased_exp == 0;
    let exponent = if is_denormal { -1074 } else { biased_exp - 1075 };
    let significand: u64 = if is_denormal {
        bits & 0x000F_FFFF_FFFF_FFFF
    } else {
        (bits & 0x000F_FFFF_FFFF_FFFF) | (1u64 << 52)
    };
    // `kSignificandSize` = 53 (52 explicit mantissa bits + the implicit leading 1).
    let result_bits: u64 = if exponent < 0 {
        if exponent <= -53 { 0 } else { significand >> (-exponent) }
    } else if exponent > 31 {
        0
    } else {
        (significand << exponent) & 0xFFFF_FFFF
    };
    (sign * (result_bits as i64)) as i32
}

/// `x >>> 0` / the ECMA-262 `ToUint32` abstract operation: [`to_int32`]'s bits, reinterpreted as
/// unsigned. Ported from `DoubleToUint32` (`deps/v8/src/numbers/conversions-inl.h:336-337`),
/// which is literally `static_cast<uint32_t>(DoubleToInt32(x))`.
pub fn to_uint32(x: f64) -> u32 {
    to_int32(x) as u32
}

/// `Math.imul(x, y)`: both operands go through [`to_int32`], multiply modulo 2^32 with wrapping
/// (no overflow trap — JS numbers can't overflow, this is exactly what the spec's "multiplication
/// modulo 2^32" wording means), and the `i32` result is exact (always representable as a
/// double). Ported from `math.tq`'s `MathImul` builtin ("ES6 #sec-math.imul"):
/// `Convert<int32>(...) * Convert<int32>(...)`, which is 32-bit wrapping multiplication on the
/// CodeStubAssembler `int32` type.
pub fn imul(x: f64, y: f64) -> i32 {
    to_int32(x).wrapping_mul(to_int32(y))
}

/// `x << y` (ECMA-262 `<<`): [`to_int32`] on `x`, shifted left by `ToUint32(y) & 31`. Always
/// returns a value that is already a valid `i32` (never needs wraparound handling beyond the
/// shift itself, since shifting an `i32` left by 0..31 bits and keeping the low 32 bits — which
/// `<<` on `i32` already does — is exactly the spec's `x << (shiftCount)` modulo-2^32 result
/// reinterpreted as signed).
pub fn shl(x: f64, y: f64) -> i32 {
    let shift = to_uint32(y) & 31;
    to_int32(x) << shift
}

/// `x >> y` (ECMA-262 `>>`, arithmetic/sign-extending): [`to_int32`] on `x`, shifted right by
/// `ToUint32(y) & 31`, sign bit replicated (Rust's `>>` on `i32` is already arithmetic).
pub fn shr(x: f64, y: f64) -> i32 {
    let shift = to_uint32(y) & 31;
    to_int32(x) >> shift
}

/// `x >>> y` (ECMA-262 `>>>`, logical/zero-filling): [`to_uint32`] on `x`, shifted right by
/// `ToUint32(y) & 31` with zero fill (Rust's `>>` on `u32` is already logical). Always
/// non-negative, matching that `>>>`'s result is a `Uint32`, never a negative `Number`.
pub fn ushr(x: f64, y: f64) -> u32 {
    let shift = to_uint32(y) & 31;
    to_uint32(x) >> shift
}

/// `x % y` on two `f64` (ECMA-262 `Number::remainder`, *not* Euclidean/Python-style modulo): the
/// IEEE-754 `fmod`-style remainder (result has the sign of `x`; `x % Infinity == x` for finite
/// `x`; `x % 0 == NaN`; `Infinity % y == NaN`).
///
/// V8's Turbofan JIT computes this with a `fprem` loop on x64
/// (`deps/v8/src/compiler/backend/x64/code-generator-x64.cc`, case `kSSEFloat64Mod`); the
/// non-JIT/runtime fallback is `Modulo()` (`deps/v8/src/utils/utils.h:105`), which on
/// non-Windows/AIX platforms is `std::fmod(x, y)` directly. Both compute the same
/// infinitely-precise remainder (`x87 fprem`, repeated until the reduction is exact, is
/// mathematically equivalent to `fmod` for finite operands), and Rust's `%` operator on `f64` is
/// specified to be IEEE-754 `fmod` too (same as C's `fmod`) — so no port is needed, only this
/// name, for symmetry with the rest of the crate's API and so probes can find it by name.
pub fn rem(x: f64, y: f64) -> f64 {
    x % y
}

/// `Math.hypot(...xs)` for any number of arguments (including zero, matching `Math.hypot() ===
/// 0`), scaled by the largest `|xs[i]|` to avoid premature overflow/underflow, with Kahan
/// summation of the squared, scaled terms to control rounding error, and `±Infinity` beating NaN
/// (`Math.hypot(NaN, Infinity) === Infinity`).
///
/// This single implementation is bit-identical to *both* of V8's two code paths for every
/// argument count (see the crate README "hypot: один путь вместо трёх" for the full derivation):
///
/// - `FastMathHypot` (`deps/v8/src/builtins/math.tq:399-461`), used for 0..3 arguments: separate
///   closed-form formulas per arity. For 2 arguments there is no compensation term at all; for 3,
///   a single compensation term `(powerA + powerB) - powerA - powerB` is subtracted from the
///   third squared term before summing. Both are what you get by running the Kahan loop below
///   with 2 or 3 elements: the compensation is provably `0.0` after the first iteration (adding
///   to a `0.0` accumulator is exact, and the squared term is never negative or infinite here,
///   since it's already divided by the running max), so the loop's output collapses to exactly
///   those closed forms, term for term, same associativity.
/// - The general (`length > 3`) path (`deps/v8/src/builtins/math.tq:462-497`): tracks the running
///   max of `|xs[i]|` ignoring NaNs (so a `NaN` argument never updates `max`, but is recorded in a
///   separate flag) and Infinity/NaN precedence exactly as coded below, then Kahan-sums
///   `(|xs[i]| / max)^2`.
pub fn hypot(xs: &[f64]) -> f64 {
    let mut max = 0.0f64;
    let mut any_nan = false;
    for &x in xs {
        let ax = x.abs();
        if ax.is_nan() {
            any_nan = true;
        } else if ax > max {
            max = ax;
        }
    }
    if max.is_infinite() {
        return f64::INFINITY;
    }
    if any_nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for &x in xs {
        let n = x.abs() / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

/// `Math.hypot(a, b)`, the overwhelmingly common 2-argument case (the old bot's ~37 call sites are
/// all 2-argument distance computations) — bit-identical to [`hypot`]`(&[a, b])`.
///
/// Task 4.13: the closed form of the general loop for two finite, not both zero arguments. With
/// `max = max(|a|, |b|)` the loop computes `n1 = |a| / max`, `n2 = |b| / max`, and its Kahan
/// compensation is `+0.0` after the first term (`(s - 0.0) - s` for the first summand `s >= 0`), so
/// the sum is `n1 * n1 + n2 * n2` and the result `sqrt(sum) * max`. The larger argument divided by
/// itself is exactly `1.0`, so its division is skipped. NaN, infinities and the all-zero case take
/// the general path (their precedence rules are in [`hypot`]).
#[inline]
pub fn hypot2(a: f64, b: f64) -> f64 {
    let (aa, ab) = (a.abs(), b.abs());
    // `<= MAX` is false for NaN and for infinity.
    if !(aa <= f64::MAX && ab <= f64::MAX) || (aa == 0.0 && ab == 0.0) {
        return hypot(&[a, b]);
    }
    if aa >= ab {
        let n2 = ab / aa;
        (1.0 + n2 * n2).sqrt() * aa
    } else {
        let n1 = aa / ab;
        (n1 * n1 + 1.0).sqrt() * ab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_ties_toward_positive_infinity() {
        assert_eq!(round(-2.5).to_bits(), (-2.0f64).to_bits());
        assert_eq!(round(2.5).to_bits(), 3.0f64.to_bits());
        assert_eq!(round(0.5).to_bits(), 1.0f64.to_bits());
        // Math.round(-0.5) === -0 in real V8 (verified with `node -e`): ceil(-0.5) == -0.0, and
        // -0.0 - 0.5 == -0.5 <= -0.5 holds, so the -0.0 from ceil is returned unmodified.
        let r = round(-0.5);
        assert_eq!(r, 0.0);
        assert!(r.is_sign_negative());
    }

    #[test]
    fn round_preserves_minus_zero_below_half() {
        let r = round(-0.4);
        assert_eq!(r, 0.0);
        assert!(r.is_sign_negative(), "Math.round(-0.4) must be -0");
    }

    #[test]
    fn round_double_just_below_half_rounds_down() {
        // The largest f64 strictly less than 0.5.
        let x = 0.49999999999999994_f64;
        assert_eq!(round(x), 0.0);
        assert!(!round(x).is_sign_negative());
    }

    #[test]
    fn max_min_prefer_signed_zero_regardless_of_order() {
        assert!(!js_max(-0.0, 0.0).is_sign_negative());
        assert!(!js_max(0.0, -0.0).is_sign_negative());
        assert!(js_min(-0.0, 0.0).is_sign_negative());
        assert!(js_min(0.0, -0.0).is_sign_negative());
    }

    #[test]
    fn max_min_propagate_nan() {
        assert!(js_max(f64::NAN, 1.0).is_nan());
        assert!(js_max(1.0, f64::NAN).is_nan());
        assert!(js_min(f64::NAN, 1.0).is_nan());
        assert!(min_n(&[1.0, f64::NAN, 2.0]).is_nan());
    }

    #[test]
    fn max_n_min_n_empty_are_infinities() {
        assert_eq!(max_n(&[]), f64::NEG_INFINITY);
        assert_eq!(min_n(&[]), f64::INFINITY);
    }

    #[test]
    fn to_int32_wraps_like_js() {
        assert_eq!(to_int32(4294967296.0), 0); // 2^32 | 0 === 0
        assert_eq!(to_int32(4294967295.0), -1); // (2^32 - 1) | 0 === -1
        assert_eq!(to_int32(f64::NAN), 0);
        assert_eq!(to_int32(f64::INFINITY), 0);
        assert_eq!(to_int32(f64::NEG_INFINITY), 0);
        assert_eq!(to_int32(-1.5), -1); // truncates toward zero, not floor
    }

    #[test]
    fn to_uint32_matches_shr_zero() {
        assert_eq!(to_uint32(-1.0), 0xFFFF_FFFFu32);
        assert_eq!(to_uint32(4294967296.0), 0);
    }

    #[test]
    fn hypot_special_cases() {
        assert_eq!(hypot(&[]), 0.0);
        assert_eq!(hypot2(3.0, 4.0), 5.0);
        assert_eq!(hypot(&[f64::NAN, f64::INFINITY]), f64::INFINITY);
        assert!(hypot(&[f64::NAN, 1.0]).is_nan());
        assert_eq!(hypot(&[0.0, 0.0, 0.0]), 0.0);
    }

    /// Task 4.13: the closed form of `hypot2` is the general loop, bit for bit, including the special values.
    #[test]
    fn hypot2_closed_form_equals_the_general_loop() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            3.0,
            4.0,
            1e-310,
            -1e-310,
            5e-324,
            1e-160,
            1e160,
            1e300,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &a in &specials {
            for &b in &specials {
                let (x, y) = (hypot2(a, b), hypot(&[a, b]));
                assert!(
                    x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                    "hypot2({a}, {b}) = {x} vs {y}"
                );
            }
        }
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200_000 {
            let scale = |bits: u64| {
                let mant = (bits >> 11) as f64 / (1u64 << 53) as f64;
                let exp = ((bits & 0x7ff) as i32 - 1023) / 4;
                let v = mant * 2f64.powi(exp);
                if bits & 0x800 != 0 { -v } else { v }
            };
            let (a, b) = (scale(next()), scale(next()));
            let (x, y) = (hypot2(a, b), hypot(&[a, b]));
            assert_eq!(x.to_bits(), y.to_bits(), "hypot2({a}, {b})");
            // The common case of the planner: two comparable magnitudes.
            let (c, d) = (a, a * 0.5 + b * 1e-3);
            assert_eq!(hypot2(c, d).to_bits(), hypot(&[c, d]).to_bits(), "hypot2({c}, {d})");
        }
    }

    #[test]
    fn imul_wraps_to_i32() {
        assert_eq!(imul(0xFFFFFFFFu32 as f64, 5.0), -5);
        assert_eq!(imul(2.0, 4.0), 8);
    }

    #[test]
    fn sign_preserves_zero_and_nan() {
        assert!(sign(0.0) == 0.0 && !sign(0.0).is_sign_negative());
        assert!(sign(-0.0).is_sign_negative());
        assert!(sign(f64::NAN).is_nan());
        assert_eq!(sign(5.0), 1.0);
        assert_eq!(sign(-5.0), -1.0);
    }
}
