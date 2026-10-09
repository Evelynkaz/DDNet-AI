//! Small helpers shared by the ports: the fused multiply-add and the special-case results of
//! glibc's `math_err.c` / `math_errf.c` (errno and the floating-point exception flags are not
//! reproduced; only the returned values are).

/// The fused multiply-add a port runs with, chosen once per public call (see [`dispatch`]) so that the choice is a branch per
/// call of `sinf`, `powf`, ... and not one per fused operation.
pub(crate) trait Fma {
    /// `a * b + c` with a single, correct rounding (C `fma`, `__builtin_fma`, and what GCC emits when it contracts `a * b + c`
    /// while compiling for `-mfma`).
    ///
    /// This is the one place the glibc **FMA variants** (`__sinf_fma`, `__powf_fma`, `__log_fma`, ...) are reproduced: on every
    /// x86-64 CPU with FMA3 glibc's `ifunc` picks these variants, whose polynomial evaluations GCC compiled to fused operations,
    /// and the results differ from the plain-SSE2 variants in the last bit for a few inputs. All implementations give the same
    /// bits; which one runs is decided in [`crate::softfma`]. See the crate docs.
    fn fma(a: f64, b: f64, c: f64) -> f64;
}

/// The instruction, or the C library's `fma` after it passed the self-check of [`crate::softfma`].
pub(crate) struct Fused;

/// The crate's own integer `fma` ([`crate::softfma::soft_fma`]).
pub(crate) struct Soft;

impl Fma for Fused {
    #[inline(always)]
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }
}

impl Fma for Soft {
    #[inline(always)]
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        crate::softfma::soft_fma(a, b, c)
    }
}

/// `dispatch!(name_impl(args))`: calls `name_impl::<Soft>` or `name_impl::<Fused>` as [`crate::softfma::use_soft`] says.
macro_rules! dispatch {
    ($f:ident($($arg:expr),* $(,)?)) => {
        if $crate::softfma::use_soft() {
            $f::<$crate::util::Soft>($($arg),*)
        } else {
            $f::<$crate::util::Fused>($($arg),*)
        }
    };
}
pub(crate) use dispatch;

/// `x + y` where at least one operand is a NaN: the result is the first NaN operand, quieted (the x86 SSE
/// rule for `addss xmm_x, xmm_y`, which is how GCC compiled glibc's `x + y`). Written out so the payload of
/// a NaN result does not depend on which operand order the Rust compiler picks for the commutative add.
#[inline(always)]
pub(crate) fn nan_sum_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(x.to_bits() | 0x0040_0000)
    } else {
        f32::from_bits(y.to_bits() | 0x0040_0000)
    }
}

/// See [`nan_sum_f32`].
#[inline(always)]
pub(crate) fn nan_sum_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000)
    } else {
        f64::from_bits(y.to_bits() | 0x0008_0000_0000_0000)
    }
}

/// `__math_invalidf (x)` / `__math_invalid (x)`: `(x - x) / (x - x)`. NaN for an infinite `x` (the x86
/// "default NaN", sign bit set) and `x` itself, quieted, for a NaN `x`.
// `x - x` is glibc's idiom for "NaN, and raise the invalid-operation exception": not a bug.
#[allow(clippy::eq_op)]
#[inline(always)]
pub(crate) fn invalid_f32(x: f32) -> f32 {
    let d = x - x;
    d / d
}

/// See [`invalid_f32`].
#[allow(clippy::eq_op)]
#[inline(always)]
pub(crate) fn invalid_f64(x: f64) -> f64 {
    let d = x - x;
    d / d
}

/// `__math_divzero (sign)`: `±inf`.
#[inline(always)]
pub(crate) fn divzero_f64(sign: bool) -> f64 {
    if sign { f64::NEG_INFINITY } else { f64::INFINITY }
}

/// `__math_oflow (sign)`: `(sign ? -0x1p769 : 0x1p769) * 0x1p769` is `±inf`.
#[inline(always)]
pub(crate) fn oflow_f64(sign: bool) -> f64 {
    if sign { f64::NEG_INFINITY } else { f64::INFINITY }
}

/// `__math_uflow (sign)`: `(sign ? -0x1p-767 : 0x1p-767) * 0x1p-767` underflows to `±0`.
#[inline(always)]
pub(crate) fn uflow_f64(sign: bool) -> f64 {
    if sign { -0.0 } else { 0.0 }
}

/// `__math_divzerof (sign)`: `±inf`.
#[inline(always)]
pub(crate) fn divzero_f32(sign: bool) -> f32 {
    if sign { f32::NEG_INFINITY } else { f32::INFINITY }
}

/// `__math_oflowf (sign)`: `±inf`.
#[inline(always)]
pub(crate) fn oflow_f32(sign: bool) -> f32 {
    if sign { f32::NEG_INFINITY } else { f32::INFINITY }
}

/// `__math_uflowf (sign)`: `±0`.
#[inline(always)]
pub(crate) fn uflow_f32(sign: bool) -> f32 {
    if sign { -0.0 } else { 0.0 }
}

/// `__math_may_uflowf (sign)`: `(sign ? -y : y) * y` with `y = 0x1.4p-75f`, i.e. `1.5625 * 2^-150`,
/// which rounds to the smallest subnormal `±2^-149`.
#[inline(always)]
pub(crate) fn may_uflow_f32(sign: bool) -> f32 {
    let tiny = f32::from_bits(1);
    if sign { -tiny } else { tiny }
}
