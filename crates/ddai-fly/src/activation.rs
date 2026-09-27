//! The model's two small nonlinearities (FLY.md §4): the firing-rate activation `f` and the
//! `softplus`/inverse-`softplus` pair used to keep `alpha` (connection-strength scale) and `tau`
//! (time constant) positive while still being able to learn them as unconstrained reals `a`/`theta`.
//! All numerically stable (no overflow for large inputs, no needless precision loss for small
//! ones) — see each function's doc comment.

/// `f(V) = r_max * tanh(relu(V) / r_max)`: ReLU with a soft ceiling at `r_max` (FLY.md §4).
///
/// Short-circuits to `0.0` for `v <= 0` without calling `tanh` at all — mathematically identical
/// (`relu(v) == 0` there, and `tanh(0) == 0`), but `tanh` is a real transcendental-function call,
/// and review round 1 (F3) measured a meaningful fraction of neurons sitting at `V <= 0` at any
/// given time even in the tuned network (see the README's stability table's "fraction with
/// r > 0"), so this branch is a free win on every one of those.
#[inline(always)]
pub fn activation(v: f32, r_max: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    r_max * (v / r_max).tanh()
}

/// `softplus(x) = ln(1 + e^x)`, computed the numerically stable way (never overflows for large
/// `x`, never loses precision for very negative `x`): `max(x, 0) + ln1p(exp(-|x|))`.
#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Inverse of [`softplus`] for `y > 0`: `x = ln(e^y - 1) = ln(expm1(y))`. Used only to turn a
/// *target* value (e.g. "I want tau = 50ms, what theta gives that?") into the unconstrained
/// parameter at initialisation time — not on any hot path, so the direct `exp_m1().ln()` form is
/// fine (no special-casing needed for the tiny-`y` limit: `ln` of a very small positive number is
/// just a large negative, finite result, which is exactly the intended parameter value).
#[inline]
pub fn inverse_softplus(y: f32) -> f32 {
    debug_assert!(y > 0.0, "inverse_softplus is only defined for y > 0, got {y}");
    y.exp_m1().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_clamps_negative_v_to_zero() {
        assert_eq!(activation(-5.0, 10.0), 0.0);
        assert_eq!(activation(0.0, 10.0), 0.0);
    }

    #[test]
    fn activation_saturates_towards_r_max_for_large_v() {
        // tanh saturates to (f32-indistinguishable-from) 1.0 well before v=1000, so this only
        // checks the ceiling is reached and never exceeded, not that it's approached strictly.
        let r = activation(1000.0, 10.0);
        assert!(r <= 10.0);
        assert!(r > 9.99);
    }

    #[test]
    fn activation_is_linear_ish_for_small_v_relative_to_r_max() {
        // tanh(x) ~= x for small x, so f(v) ~= v when v << r_max.
        let r = activation(0.01, 10.0);
        assert!((r - 0.01).abs() < 1e-5, "r={r}");
    }

    #[test]
    fn softplus_round_trips_through_inverse_softplus() {
        for target in [0.001f32, 0.01, 0.04, 0.5, 1.0, 5.0, 20.0] {
            let x = inverse_softplus(target);
            let back = softplus(x);
            let rel = (back - target).abs() / target;
            assert!(rel < 1e-4, "target={target} back={back} rel={rel}");
        }
    }

    #[test]
    fn softplus_never_overflows_for_large_positive_x() {
        let y = softplus(1e6);
        assert!(y.is_finite());
        assert!((y - 1e6).abs() < 1.0);
    }

    #[test]
    fn softplus_is_positive_everywhere() {
        for x in [-1e6f32, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6] {
            assert!(softplus(x) >= 0.0, "softplus({x}) went negative");
        }
    }
}
