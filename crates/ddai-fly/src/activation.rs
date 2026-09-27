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

/// `d softplus(x) / dx = sigmoid(x) = 1 / (1 + e^-x)`, computed the numerically stable way (never
/// overflows for very negative `x`: for `x < 0` this uses `e^x / (1 + e^x)` instead, which keeps
/// the exponential's argument non-positive either way). Used by the backward pass (task 7.2) to
/// convert a gradient with respect to `alpha = softplus(a)` (or `tau`'s `softplus(theta)` term)
/// into a gradient with respect to the underlying unconstrained parameter.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `d f(v) / dv` where `f(v) = r_max * tanh(relu(v) / r_max)` (task 7.2's backward pass). Exactly
/// `0` for `v <= 0` — **the relu kink's subgradient is defined as `0` here** (the same convention
/// [`activation`] itself already uses for the value at `v == 0`: that point belongs to the "zero"
/// branch, not the smooth one, so its derivative is 0 too, not some value in `[0, 1]`).
///
/// For `v > 0`: `d/dv [r_max * tanh(v / r_max)] = sech^2(v / r_max) = 1/cosh^2(v / r_max)`.
/// Deliberately **not** the algebraically-equivalent `1 - tanh^2(v / r_max)` (whether from a fresh
/// `tanh` call or, as an earlier version of this function did, reusing an already-computed `r =
/// f(v)` via `1 - (r / r_max)^2`): review round 1 (F7a) found that form loses real precision in
/// `f32` for `V` in the saturated region, where `tanh(v/r_max)` is close to `1` and squaring it
/// then subtracting from `1` is textbook catastrophic cancellation (measured ~1e-2 *relative*
/// error at `r_max = 1`, moderately into saturation — far more than `f32`'s own `~1.2e-7` epsilon
/// would suggest, precisely because the cancellation amplifies it). `1/cosh^2` has no subtraction
/// of nearly-equal quantities at all: `cosh` grows smoothly and unboundedly with `|v|`, so its
/// reciprocal square just shrinks smoothly towards `0` — no cancellation, no loss.
#[inline(always)]
pub fn activation_derivative(v: f32, r_max: f32) -> f32 {
    if v <= 0.0 {
        0.0
    } else {
        let c = (v / r_max).cosh();
        1.0 / (c * c)
    }
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

    #[test]
    fn sigmoid_matches_softplus_derivative_by_central_finite_difference() {
        let h = 1e-3f32;
        for x in [-20.0f32, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0] {
            let fd = (softplus(x + h) - softplus(x - h)) / (2.0 * h);
            let analytic = sigmoid(x);
            assert!((fd - analytic).abs() < 1e-3, "x={x}: fd={fd} analytic={analytic}");
        }
    }

    #[test]
    fn sigmoid_stays_in_unit_interval_and_never_overflows() {
        for x in [-1e6f32, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6] {
            let s = sigmoid(x);
            assert!(s.is_finite(), "sigmoid({x}) not finite: {s}");
            assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s} out of [0,1]");
        }
        assert!(sigmoid(-1e6) < 1e-6);
        assert!(sigmoid(1e6) > 1.0 - 1e-6);
    }

    #[test]
    fn activation_derivative_is_zero_at_and_below_the_relu_kink() {
        for v in [-5.0f32, -0.001, 0.0] {
            assert_eq!(activation_derivative(v, 10.0), 0.0, "v={v}");
        }
    }

    #[test]
    fn activation_derivative_matches_central_finite_difference_for_positive_v() {
        let r_max = 10.0f32;
        let h = 1e-3f32;
        for v in [0.001f32, 0.1, 1.0, 5.0, 20.0, 100.0] {
            let analytic = activation_derivative(v, r_max);
            let fd = (activation(v + h, r_max) - activation(v - h, r_max)) / (2.0 * h);
            assert!((analytic - fd).abs() < 1e-3, "v={v}: analytic={analytic} fd={fd}");
        }
    }

    #[test]
    fn activation_derivative_is_close_to_one_for_small_positive_v() {
        // f(v) ~= v for v << r_max, so f'(v) ~= 1 there.
        let d = activation_derivative(1e-4, 10.0);
        assert!((d - 1.0).abs() < 1e-3, "d={d}");
    }

    /// Review round 1, F7a: the earlier `1 - (r/r_max)^2` form lost real precision deep in
    /// saturation (measured ~1e-2 relative at `r_max = 1`); `1/cosh^2` must not. `r_max = 1` here
    /// specifically (not this crate's usual `10.0`) to reproduce the reviewer's own measurement
    /// exactly.
    #[test]
    fn activation_derivative_stays_precise_deep_in_saturation() {
        let r_max = 1.0f32;
        for v in [3.0f32, 5.0, 8.0, 12.0] {
            let analytic = f64::from(activation_derivative(v, r_max));
            // An independent, `f64` "exact" reference for the same quantity, `1/cosh(x)^2`,
            // computed once in wide precision so this test doesn't just re-derive the same `f32`
            // cancellation it's trying to catch.
            let x = f64::from(v) / f64::from(r_max);
            let exact = 1.0 / x.cosh().powi(2);
            let rel = (analytic - exact).abs() / exact.max(1e-300);
            assert!(rel < 1e-4, "v={v}: analytic={analytic} exact={exact} rel={rel}");
        }
    }
}
