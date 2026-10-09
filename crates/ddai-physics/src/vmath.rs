// Ported from DDNet 20.1 `src/base/vmath.h` and `src/base/math.h` (`vector2_base<float>`/`vec2`
// and the free functions this crate's physics actually uses), generalized over the scalar type
// `R: Real` (see `crate::real`) instead of being hardcoded to `float`. DDNet's own license
// notice for the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real` (`f32`/`f64`) instead of
// `float`; only the subset of `vmath.h`/`math.h` that `collision`/`core`/`character` actually use
// is ported (see each item's doc comment for the exact `vmath.h`/`math.h` source it mirrors);
// `fx2f`/`f2fx` and the fixed-point `fxp` type are intentionally omitted — neither
// `collision.cpp` nor `gamecore.cpp` uses them (verified by reading both files in full).

use crate::real::Real;

/// Mirrors `vector2_base<float>` / `vec2` (`base/vmath.h`), generalized over `R`. Only the
/// operators `collision`/`core` actually use are implemented (`+`, `-`, unary `-`, `* R`).
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Vec2<R: Real> {
    /// `x`/`u`.
    pub x: R,
    /// `y`/`v`.
    pub y: R,
}

impl<R: Real> Vec2<R> {
    /// `vector2_base(T nx, T ny)`.
    pub const fn new(x: R, y: R) -> Self {
        Vec2 { x, y }
    }

    /// `(0, 0)`.
    pub fn zero() -> Self {
        Vec2::new(R::ZERO, R::ZERO)
    }
}

impl<R: Real> std::ops::Add for Vec2<R> {
    type Output = Vec2<R>;
    fn add(self, o: Vec2<R>) -> Vec2<R> {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}

impl<R: Real> std::ops::Sub for Vec2<R> {
    type Output = Vec2<R>;
    fn sub(self, o: Vec2<R>) -> Vec2<R> {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}

impl<R: Real> std::ops::Neg for Vec2<R> {
    type Output = Vec2<R>;
    fn neg(self) -> Vec2<R> {
        Vec2::new(-self.x, -self.y)
    }
}

/// `vector2_base::operator*(const T rhs)`: scalar multiplication, vector on the left.
impl<R: Real> std::ops::Mul<R> for Vec2<R> {
    type Output = Vec2<R>;
    fn mul(self, s: R) -> Vec2<R> {
        Vec2::new(self.x * s, self.y * s)
    }
}

impl<R: Real> std::ops::AddAssign for Vec2<R> {
    fn add_assign(&mut self, o: Vec2<R>) {
        self.x += o.x;
        self.y += o.y;
    }
}

impl<R: Real> std::ops::MulAssign<R> for Vec2<R> {
    fn mul_assign(&mut self, s: R) {
        self.x *= s;
        self.y *= s;
    }
}

/// `dot(a, b)` (`vmath.h`, the `Numeric T` template): `a.x*b.x + a.y*b.y`.
pub fn dot<R: Real>(a: Vec2<R>, b: Vec2<R>) -> R {
    a.x * b.x + a.y * b.y
}

/// `length(vector2_base<float>&)` (`vmath.h`): `sqrt(dot(a, a))`. Unlike the C++ header (whose
/// `std::floating_point T` overload is declared to always return `float` even for a
/// hypothetical `vector2_base<double>` — a quirk DDNet's own code never exercises, since `vec2`
/// is always `vector2_base<float>` there), this returns `R` — the natural generalization for a
/// port that genuinely instantiates both widths.
pub fn length<R: Real>(a: Vec2<R>) -> R {
    dot(a, a).sqrt()
}

/// `length_squared(const vector2_base<float>&)` (`vmath.h`).
pub fn length_squared<R: Real>(a: Vec2<R>) -> R {
    dot(a, a)
}

/// `distance(vector2_base<T>, const vector2_base<T>&)` (`vmath.h`): `length(a - b)`.
pub fn distance<R: Real>(a: Vec2<R>, b: Vec2<R>) -> R {
    length(a - b)
}

/// `distance_squared` (`vmath.h`).
pub fn distance_squared<R: Real>(a: Vec2<R>, b: Vec2<R>) -> R {
    length_squared(a - b)
}

/// `normalize(const vector2_base<float>&)` (`vmath.h`): explicit zero check (returns the zero
/// vector rather than dividing by zero and propagating `NaN`/`inf`), otherwise multiplies by the
/// reciprocal (matching the source's `1.0f / divisor` then `v * l`, not a direct `v / divisor` —
/// the exact operation order matters for bit-exactness, see `docs/DECISIONS.md` D-002).
pub fn normalize<R: Real>(v: Vec2<R>) -> Vec2<R> {
    let divisor = length(v);
    if divisor == R::ZERO {
        return Vec2::zero();
    }
    let l = R::ONE / divisor;
    Vec2::new(v.x * l, v.y * l)
}

/// `mix(a, b, amount)` (`base/math.h`): `a + (b - a) * amount`, specialized to `Vec2` (the only
/// instantiation `collision`/`core` use).
pub fn mix<R: Real>(a: Vec2<R>, b: Vec2<R>, amount: R) -> Vec2<R> {
    a + (b - a) * amount
}

/// `closest_point_on_line` (`vmath.h`): the point on segment `a..b` closest to `target`, clamped
/// to the segment (`t` in `0..=1`). Returns `None` when `a == b` (zero-length segment, matching
/// the C++ `SquaredMagnitudeAB > 0` check returning `false`) instead of the C++ out-parameter's
/// `false` return.
pub fn closest_point_on_line<R: Real>(a: Vec2<R>, b: Vec2<R>, target: Vec2<R>) -> Option<Vec2<R>> {
    let ab = b - a;
    let squared_magnitude_ab = dot(ab, ab);
    if squared_magnitude_ab > R::ZERO {
        let ap = target - a;
        let t = dot(ap, ab) / squared_magnitude_ab;
        Some(a + ab * t.clamp(R::ZERO, R::ONE))
    } else {
        None
    }
}

/// `direction(float angle)` (`vmath.h`): `(cos(angle), sin(angle))`.
pub fn direction<R: Real>(angle: R) -> Vec2<R> {
    Vec2::new(angle.cos(), angle.sin()) // libm-census: `Real::cos`/`sin` (ddai_libm for f32)
}

/// `angle(const vector2_base<float>&)` (`vmath.h`).
pub fn angle<R: Real>(a: Vec2<R>) -> R {
    if a.x == R::ZERO && a.y == R::ZERO {
        R::ZERO
    } else if a.x == R::ZERO {
        if a.y < R::ZERO {
            -R::PI / R::from_i32(2)
        } else {
            R::PI / R::from_i32(2)
        }
    } else {
        let mut result = (a.y / a.x).atan(); // libm-census: `Real::atan` (ddai_libm for f32)
        if a.x < R::ZERO {
            result += R::PI;
        }
        result
    }
}

/// `round_to_int(float f)` (`base/math.h`): `f > 0 ? (int)(f + 0.5f) : (int)(f - 0.5f)`. The
/// addition/subtraction of `0.5` happens in `R` (never widened), exactly like the C++ source —
/// this is one of the two porting rules Phase-0 identified as necessary for bit-exactness (see
/// `docs/DECISIONS.md` D-002/D-004).
pub fn round_to_int<R: Real>(f: R) -> i32 {
    let half = R::from_f64(0.5);
    if f > R::ZERO {
        (f + half).to_i32_trunc()
    } else {
        (f - half).to_i32_trunc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_and_length_match_pythagoras() {
        let v = Vec2::new(3.0f32, 4.0f32);
        assert_eq!(dot(v, v), 25.0);
        assert_eq!(length(v), 5.0);
    }

    #[test]
    fn distance_is_symmetric() {
        let a = Vec2::new(1.0f32, 2.0f32);
        let b = Vec2::new(4.0f32, 6.0f32);
        assert_eq!(distance(a, b), 5.0);
        assert_eq!(distance(a, b), distance(b, a));
    }

    #[test]
    fn normalize_of_zero_vector_is_zero_not_nan() {
        let v: Vec2<f32> = Vec2::zero();
        let n = normalize(v);
        assert_eq!(n, Vec2::zero());
    }

    #[test]
    fn normalize_produces_unit_length() {
        let v = Vec2::new(3.0f32, 4.0f32);
        let n = normalize(v);
        assert_eq!(n, Vec2::new(0.6, 0.8));
        assert!((length(n) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_f64_zero_vector_is_zero() {
        let v: Vec2<f64> = Vec2::zero();
        assert_eq!(normalize(v), Vec2::zero());
    }

    #[test]
    fn mix_interpolates_linearly() {
        let a = Vec2::new(0.0f32, 0.0f32);
        let b = Vec2::new(10.0f32, 20.0f32);
        assert_eq!(mix(a, b, 0.0), a);
        assert_eq!(mix(a, b, 1.0), b);
        assert_eq!(mix(a, b, 0.5), Vec2::new(5.0, 10.0));
    }

    #[test]
    fn closest_point_on_line_clamps_to_segment() {
        let a = Vec2::new(0.0f32, 0.0f32);
        let b = Vec2::new(10.0f32, 0.0f32);
        // Perpendicular from the middle: closest point is the projection, inside the segment.
        assert_eq!(
            closest_point_on_line(a, b, Vec2::new(5.0, 3.0)),
            Some(Vec2::new(5.0, 0.0))
        );
        // Beyond `b`: clamps to `b`.
        assert_eq!(closest_point_on_line(a, b, Vec2::new(15.0, 3.0)), Some(b));
        // Before `a`: clamps to `a`.
        assert_eq!(closest_point_on_line(a, b, Vec2::new(-5.0, 3.0)), Some(a));
    }

    #[test]
    fn closest_point_on_line_returns_none_for_a_degenerate_segment() {
        let a = Vec2::new(1.0f32, 1.0f32);
        assert_eq!(closest_point_on_line(a, a, Vec2::new(5.0, 5.0)), None);
    }

    #[test]
    fn direction_and_angle_round_trip() {
        let d = direction(0.0f32);
        assert_eq!(d, Vec2::new(1.0, 0.0));
        let a = angle(Vec2::new(1.0f32, 0.0f32));
        assert_eq!(a, 0.0);
    }

    #[test]
    fn angle_of_zero_vector_is_zero() {
        assert_eq!(angle(Vec2::new(0.0f32, 0.0f32)), 0.0);
    }

    #[test]
    fn angle_handles_vertical_vectors() {
        assert_eq!(angle(Vec2::new(0.0f32, 5.0f32)), <f32 as Real>::PI / 2.0);
        assert_eq!(angle(Vec2::new(0.0f32, -5.0f32)), -<f32 as Real>::PI / 2.0);
    }

    // `round_to_int` (`base/math.h`): `f > 0 ? (int)(f + 0.5f) : (int)(f - 0.5f)`.
    #[test]
    fn round_to_int_at_positive_half_boundary() {
        assert_eq!(round_to_int(0.5f32), 1); // 0.5+0.5=1.0 -> 1
        assert_eq!(round_to_int(0.499f32), 0);
        assert_eq!(round_to_int(2.5f32), 3);
        assert_eq!(round_to_int(2.499f32), 2);
    }

    #[test]
    fn round_to_int_negative_values() {
        assert_eq!(round_to_int(-2.5f32), -3);
        assert_eq!(round_to_int(-2.499f32), -2);
        assert_eq!(round_to_int(-0.5f32), -1);
    }

    #[test]
    fn round_to_int_at_zero() {
        assert_eq!(round_to_int(0.0f32), 0);
        // -0.0 is not > 0.0, so it takes the `f - 0.5` branch: -0.0 - 0.5 = -0.5 -> 0.
        assert_eq!(round_to_int(-0.0f32), 0);
    }

    #[test]
    fn round_to_int_works_for_f64_too() {
        assert_eq!(round_to_int(2.5f64), 3);
        assert_eq!(round_to_int(-2.5f64), -3);
    }
}
