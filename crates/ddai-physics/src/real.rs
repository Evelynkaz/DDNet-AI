// DDNet-AI: original Rust code (not derived from DDNet's C++), GPL-3.0-only like the rest of
// this repository. It exists to let `ddai-physics` be written once and instantiated over either
// `f32` (bit-exact parity with DDNet 20.1's C++, see `docs/DECISIONS.md` D-002/D-003) or `f64`
// (a compiling, running, but not-yet-parity-checked instantiation for later TS-compat work).

//! The [`Real`] trait: the minimal set of scalar operations `ddai-physics` needs, implemented for
//! `f32` and `f64`. The `f32` transcendental functions (`powf`/`sin`/`cos`/`atan`/`atan2`) are
//! [`ddai_libm`]'s ports of glibc 2.39's algorithms, **the same code on every platform**: DDNet 20.1's
//! C++ on Linux calls glibc's `powf`/`sinf`/`cosf`/`atanf`/`atan2f`, and the Oracle A/B traces this crate
//! must reproduce bit for bit were recorded against those (D-002/D-004). Before D-127 these methods
//! called `std`, which reaches glibc on `linux-gnu` only; on Windows (UCRT) the last bits differ, so the
//! physics would have silently diverged from the server. `ddai-libm` is probed against glibc (10^7+ random
//! inputs per function, all 2^32 `f32` inputs for `sinf`/`cosf`/`atanf`) and its results are pinned by a
//! golden-hash test that runs on every platform (see its crate docs); the `libm` crate is never used
//! (D-004: its `powf` differs from glibc's in 9.7% of probes).
//!
//! The `f64` instantiation makes no parity claim (D-002/D-003: it exists for later TS-compat work and is
//! only instantiated by tests) and keeps calling `std`: its `sin`/`cos`/`atan`/`atan2`/`powf` are
//! whatever the platform's C library gives.
//!
//! **Do not add `mul_add`, fast-math, or any other operation that could fuse/reorder floating
//! point arithmetic** to this trait or to the arithmetic in this crate. DDNet 20.1's C++ never contracts
//! float operations (no `-ffast-math`, no FMA in the reference x86-64 Linux build — see
//! `docs/research/ddnet-physics.md` §4), so this port must not either: every arithmetic expression
//! elsewhere in this crate is written to mirror the C++ source's exact operation order and relies on the
//! plain (non-fused) `+`/`*` this trait's operators provide. (The fused operations inside `ddai-libm` are
//! the ones glibc's own FMA-variant `libm` code contains; they are part of the glibc port, not of this
//! crate's arithmetic.)
//!
//! **Constant folding of libm calls (task 1.6, lesson from the 3.1a review).** LLVM rewrites
//! `powf`/`pow` calls whose exponent (or, for some patterns, base) is a compile-time constant
//! *even without fast-math*: `pow(x, -1)` becomes `1/x`, `pow(2^n, y)` becomes `exp2(n*y)`, and
//! `pow(x, 2)` may become `x*x` — all algebraically equal to the libm call in exact arithmetic,
//! but not bit-for-bit equal to what glibc's `pow`/`powf` actually returns at run time. That hazard is
//! gone for the `f32` methods: they call ordinary Rust functions made of IEEE operations, which LLVM may
//! inline and constant-fold but cannot rewrite into anything with a different value. The `f64` methods
//! still pass their operands through [`std::hint::black_box`] to keep `std`'s calls opaque to LLVM.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A scalar DDNet's physics can be instantiated over. Implemented for `f32` (production; bit-exact
/// with DDNet 20.1's C++, see `docs/DECISIONS.md` D-002) and `f64` (compiles and runs, for later
/// TS-compat parity work — no bit-exactness claim yet, see the task spec).
///
/// This is a thin wrapper trait: every method here is a one-line call into the matching `std`
/// method (or a trivial conversion), never a reimplementation. Callers elsewhere in this crate
/// are responsible for matching DDNet's exact expression *shape* (order of operations, which
/// sub-expressions are computed in `f32` vs `f64`) — this trait only supplies the primitive
/// operations, generic over which concrete width is in use.
pub trait Real:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Debug
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Send
    + Sync
    + 'static
{
    /// Additive identity (`0`).
    const ZERO: Self;
    /// Multiplicative identity (`1`).
    const ONE: Self;
    /// `base/math.h`'s `constexpr float pi = 3.1415926535897932384626433f;`, at whatever
    /// precision `Self` is — for `f32` this is bit-for-bit the same constant DDNet's C++ uses
    /// (both round the same decimal digit string to the nearest `f32`).
    const PI: Self;

    /// Widens an `i32` exactly (every `i32` value is exactly representable in both `f32` and
    /// `f64`... actually not `f32` for large magnitudes, but every value this crate ever widens
    /// this way — tile coordinates, tuning fixed-point integers, network fields — fits well
    /// within `f32`'s 24-bit mantissa; DDNet's own C++ makes the same assumption via `int`->
    /// `float` implicit conversions throughout `gamecore.cpp`/`collision.cpp`).
    fn from_i32(v: i32) -> Self;
    /// Widens/narrows an `f64` literal to `Self` — used for constants written once in the port
    /// (e.g. `R::from_f64(0.75)`) that must produce the exact same `f32` bit pattern `0.75f`
    /// would in the C++ source when `Self = f32`.
    fn from_f64(v: f64) -> Self;
    /// Converts to `f64` (e.g. for bridging into an explicitly-double computation such as
    /// DDNet's `std::atan2(int, int)` call, which promotes both arguments to `double` — see
    /// `core::angle_from_target`).
    fn to_f64(self) -> f64;
    /// `static_cast<int>(self)` semantics **on x86-64** (`cvttss2si`/`cvttsd2si`): truncates
    /// toward zero for values whose truncated result fits in `i32` (`-2147483648.0 <= self <
    /// 2147483648.0`); otherwise (NaN, ±infinity, or a magnitude too large to fit) returns
    /// `i32::MIN` — the hardware's "integer indefinite" value, per the SSE2 instruction set
    /// reference. This does **not** match Rust's own `as i32` (which saturates: NaN → `0`, `+inf`
    /// → `i32::MAX`, `-inf` → `i32::MIN`) — that mismatch is exactly review round 1's finding F1:
    /// DDNet's C++ can and does hit this path (e.g. an extreme `velramp_range`/`velramp_curvature`
    /// tuning override makes `VelocityRamp` return exactly `0.0`, and `m_Vel.x * (1.0f /
    /// RampValue)` — `gamecore.cpp`'s `Move()` — becomes `0.0 * inf = NaN`), and the resulting
    /// quantized position/velocity is observable in the trace, so every truncating float→int
    /// conversion in this crate must go through this method, not a bare `as i32`.
    fn to_i32_trunc(self) -> i32;

    /// `std::floor(Self)`. Not currently called by any ported `collision`/`core` function
    /// (DDNet's `/32` tile-index arithmetic truncates via integer division, never `std::floor`),
    /// provided for API completeness per the task spec ("floor/trunc conversions").
    fn floor(self) -> Self;

    /// `std::sqrt(Self)`.
    fn sqrt(self) -> Self;
    /// `std::pow(Self, Self)`, i.e. C++'s `float pow(float, float)`/`double pow(double, double)`
    /// overload — used only by [`crate::core::velocity_ramp`], matching `VelocityRamp`'s
    /// `std::pow(Curvature, (Value - Start) / Range)` (`gamecore.cpp`), whose *both* arguments
    /// are genuinely `float` in the C++ source, so this exact-type overload applies with no
    /// promotion. **Not** a match for `std::pow(SomeFloat, 2)` (a bare `int` literal exponent,
    /// as the DDRace old-type speedup port's `character.cpp:1558` has): with a mismatched
    /// argument-type pair, overload resolution instead picks the generic `<cmath>` "additional
    /// overload" that promotes *both* arguments (and the result) to `double` regardless of `R`
    /// — see `crate::world::apply_speedup`'s own `to_f64()`/`powi`/`sqrt` for why that call
    /// site does not use this method (an earlier revision of this crate did, and this doc
    /// comment used to claim that was correct — found empirically: the `f32` path differed from
    /// a `double`-throughout one on 16% of 1M random inputs). See the module doc comment for
    /// where the `f32` implementation comes from (`ddai_libm::powf`, glibc's `powf` ported).
    fn powf(self, exp: Self) -> Self;
    // Deliberately **no** `ln`/`log` method here: this crate's one ported `log(...)` call
    // (`crate::world::max_ramp_speed`, `character.cpp:1580`) is the *bare*, unqualified C
    // library function — not `std::log` — which has no `float` overload at all (unlike
    // `std::pow`'s `float pow(float, float)`), so it *always* computes in `double` regardless of
    // its argument's original type, with the caller responsible for widening first and
    // narrowing the final result back once. A generic `Real::ln` returning `Self` would invite
    // exactly the bug review round 2 found: an earlier revision of this crate had one, computing
    // natively in `R` (`f32::ln`/`f64::ln`) — wrong for the `f32` instantiation, since the real
    // call is `double`-only. `max_ramp_speed` calls `f64::ln()` directly instead.
    /// `std::sin(Self)`.
    fn sin(self) -> Self;
    /// `std::cos(Self)`.
    fn cos(self) -> Self;
    /// `std::atan(Self)`.
    fn atan(self) -> Self;
    /// `std::atan2(Self, Self)`.
    fn atan2(self, x: Self) -> Self;
    /// `std::abs(Self)`.
    fn abs(self) -> Self;

    /// `std::isfinite(Self)` — task 1.10b, speed-up 2: guards the `intersect_line`/
    /// `intersect_character` bounding-box early-outs (a `NaN`/`±inf` endpoint makes "the segment's
    /// bounding box" undefined, so those early-outs fall back to the exact per-sample loop
    /// instead of reasoning about it). Never used by any bit-exactness-relevant computation
    /// itself — only to decide whether a *skip* is safe to take.
    fn is_finite(self) -> bool;

    /// Rust's `f32`/`f64::min` — **not** C++ `std::min` (IEEE-754 `minNum`-style: if exactly one
    /// operand is `NaN`, returns the *other* one; C++'s `std::min` is comparison-based and would
    /// return its first argument for any `NaN` involved). No ported call site's bit-exactness
    /// depends on this distinction ([`Real::clamp`]'s default impl below no longer uses `min`/
    /// `max` at all, precisely to avoid it) — kept only for API completeness.
    fn min(self, other: Self) -> Self;
    /// See [`Real::min`].
    fn max(self, other: Self) -> Self;
    /// `std::clamp(self, lo, hi)` (`collision.cpp`'s `MoveBox`: `std::clamp(Elasticity.x, -1.0f,
    /// 1.0f)`) — direct comparisons, exactly matching `std::clamp`'s own definition (`v < lo ?
    /// lo : (hi < v ? hi : v)`) bit-for-bit for every input `libstdc++`'s `std::clamp` accepts,
    /// including `NaN` (both comparisons are false, so `v` itself is returned unchanged) —
    /// deliberately not expressed via `self.max(lo).min(hi)`, since [`Real::min`]/[`Real::max`]
    /// use different (IEEE `minNum`/`maxNum`) `NaN` handling that would silently diverge from
    /// `std::clamp` for a `NaN` input (review round 1, finding F6).
    fn clamp(self, lo: Self, hi: Self) -> Self {
        if self < lo {
            lo
        } else if hi < self {
            hi
        } else {
            self
        }
    }

    /// `std::isnan(Self)`.
    fn is_nan(self) -> bool;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    // `std::f32::consts::PI` is bit-for-bit the same `f32` value `base/math.h`'s
    // `constexpr float pi = 3.1415926535897932384626433f;` is (both are the correctly-rounded
    // `f32` nearest to real pi) — using the named `std` constant instead of retyping the digit
    // string keeps clippy's `approx_constant`/`excessive_precision` lints happy without losing
    // that bit-exactness.
    const PI: Self = std::f32::consts::PI;

    fn from_i32(v: i32) -> Self {
        v as f32
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn to_i32_trunc(self) -> i32 {
        // Exact bounds: `-2147483648.0f32`/`2147483648.0f32` (`±2^31`) are both exactly
        // representable in `f32` (a single significant bit), so this comparison is exact, not an
        // approximation — see the trait method's doc comment for why `i32::MIN` (not Rust's
        // saturating `as i32`) is the correct fallback here.
        if (-2147483648.0..2147483648.0).contains(&self) {
            self as i32
        } else {
            i32::MIN
        }
    }

    fn floor(self) -> Self {
        f32::floor(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    // The transcendental functions are `ddai-libm`'s ports of glibc 2.39's (D-127), the same code on every
    // platform; no `black_box` is needed (see the module doc comment's "constant folding" note).
    fn powf(self, exp: Self) -> Self {
        ddai_libm::powf(self, exp)
    }
    fn sin(self) -> Self {
        ddai_libm::sinf(self)
    }
    fn cos(self) -> Self {
        ddai_libm::cosf(self)
    }
    fn atan(self) -> Self {
        ddai_libm::atanf(self)
    }
    fn atan2(self, x: Self) -> Self {
        ddai_libm::atan2f(self, x)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    // No parity requirement for the `f64` instantiation (see the task spec), so this uses the
    // full-precision `f64` pi rather than the `f32`-rounded value `R::PI` is for `f32` — a
    // strictly better constant for genuine `f64` physics, and there's nothing to be bit-exact
    // with on this side.
    const PI: Self = std::f64::consts::PI;

    fn from_i32(v: i32) -> Self {
        v as f64
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn to_i32_trunc(self) -> i32 {
        // `±2^31` are exactly representable in `f64` too; see the `f32` impl's comment.
        if (-2147483648.0..2147483648.0).contains(&self) {
            self as i32
        } else {
            i32::MIN
        }
    }

    fn floor(self) -> Self {
        f64::floor(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn powf(self, exp: Self) -> Self {
        // See `impl Real for f32`'s `powf` and the module doc comment.
        // libm-census: `Real<f64>` is the instantiation without a parity claim (tests, D-002/D-003); D-127 lists f64 math as not ported
        f64::powf(std::hint::black_box(self), std::hint::black_box(exp))
    }
    fn sin(self) -> Self {
        // libm-census: as above (Real<f64>)
        f64::sin(std::hint::black_box(self))
    }
    fn cos(self) -> Self {
        // libm-census: as above (Real<f64>)
        f64::cos(std::hint::black_box(self))
    }
    fn atan(self) -> Self {
        // libm-census: as above (Real<f64>)
        f64::atan(std::hint::black_box(self))
    }
    fn atan2(self, x: Self) -> Self {
        // libm-census: as above (Real<f64>)
        f64::atan2(std::hint::black_box(self), std::hint::black_box(x))
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_basic<R: Real>() {
        assert_eq!(R::ZERO + R::ONE, R::ONE);
        assert_eq!(R::from_i32(4).sqrt(), R::from_i32(2));
        assert!(R::from_f64(f64::NAN).is_nan());
        assert_eq!(R::from_i32(-5).to_i32_trunc(), -5);
        assert_eq!(R::from_f64(3.9).to_i32_trunc(), 3);
        assert_eq!(R::from_f64(-3.9).to_i32_trunc(), -3);
    }

    #[test]
    fn f32_basic_ops() {
        check_basic::<f32>();
    }

    #[test]
    fn f64_basic_ops() {
        check_basic::<f64>();
    }

    #[test]
    fn from_f64_rounds_to_nearest_f32() {
        // 0.1 is not exactly representable in either width; from_f64 must round like a plain
        // `as f32` cast, not truncate.
        let v: f32 = Real::from_f64(0.1);
        assert_eq!(v, 0.1f32);
    }

    #[test]
    fn clamp_matches_std_clamp_for_ordered_bounds() {
        assert_eq!(Real::clamp(5.0f32, 0.0, 10.0), 5.0f32);
        assert_eq!(Real::clamp(-5.0f32, 0.0, 10.0), 0.0f32);
        assert_eq!(Real::clamp(50.0f32, 0.0, 10.0), 10.0f32);
    }
}
