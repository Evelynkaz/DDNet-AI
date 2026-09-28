//! Literal port of `src/core/vmath.ts` (TS, `Wranked1/DDNet-AI`, GPL-3.0). `f64` throughout —
//! see the crate's top-level doc comment for why this crate uses `f64` (not `f32`, unlike
//! `ddai-physics`) and always goes through `ddai_jsmath` for JS-specific numeric semantics.
//!
//! Every function here is a direct, line-for-line translation of its TS counterpart; no
//! shortcuts, no "obviously equivalent" Rust idiom substituted for a JS operation whose exact
//! semantics (rounding tie-breaks, `NaN`/signed-zero propagation) can differ from what looks
//! like the same code in Rust. See each function's doc comment for the exact `vmath.ts` line.

use ddai_jsmath as js;

/// `type Vec2 = { x: number; y: number }` (`vmath.ts:1`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

/// `vec2(x, y)` (`vmath.ts:3-5`).
pub const fn vec2(x: f64, y: f64) -> Vec2 {
    Vec2 { x, y }
}

/// `vadd(a, b)` (`vmath.ts:7-9`).
pub fn vadd(a: Vec2, b: Vec2) -> Vec2 {
    Vec2 {
        x: a.x + b.x,
        y: a.y + b.y,
    }
}

/// `vsub(a, b)` (`vmath.ts:11-13`).
pub fn vsub(a: Vec2, b: Vec2) -> Vec2 {
    Vec2 {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

/// `vmul(v, f)` (`vmath.ts:15-17`).
pub fn vmul(v: Vec2, f: f64) -> Vec2 {
    Vec2 { x: v.x * f, y: v.y * f }
}

/// `vdiv(v, f)` (`vmath.ts:19-21`). Unused by the ported core files (kept for API parity with
/// `vmath.ts` — a future caller in `src/plan`/`src/env` might reach it through `WorldView`).
pub fn vdiv(v: Vec2, f: f64) -> Vec2 {
    Vec2 { x: v.x / f, y: v.y / f }
}

/// `vlength(a)` (`vmath.ts:23-25`).
pub fn vlength(a: Vec2) -> f64 {
    js::sqrt(vdot(a, a))
}

/// `vdistance(a, b)` (`vmath.ts:27-31`).
pub fn vdistance(a: Vec2, b: Vec2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    js::sqrt(dx * dx + dy * dy)
}

/// `vdot(a, b)` (`vmath.ts:33-35`).
pub fn vdot(a: Vec2, b: Vec2) -> f64 {
    a.x * b.x + a.y * b.y
}

/// `vnormalize(v)` (`vmath.ts:37-44`). Zero-length input returns `(0, 0)`, matching the `=== 0.0`
/// check in TS (not `< epsilon`).
pub fn vnormalize(v: Vec2) -> Vec2 {
    let divisor = vlength(v);
    if divisor == 0.0 {
        return Vec2 { x: 0.0, y: 0.0 };
    }
    let l = 1.0 / divisor;
    Vec2 { x: v.x * l, y: v.y * l }
}

/// `vmix(a, b, amount)` (`vmath.ts:46-48`).
pub fn vmix(a: Vec2, b: Vec2, amount: f64) -> Vec2 {
    Vec2 {
        x: a.x + (b.x - a.x) * amount,
        y: a.y + (b.y - a.y) * amount,
    }
}

/// `direction(angle)` (`vmath.ts:50-52`). Unused by the ported core files directly (kept for API
/// parity — `wireAngleRad` callers in `src/env`/`src/plan` may reach it).
pub fn direction(angle: f64) -> Vec2 {
    Vec2 {
        x: js::cos(angle),
        y: js::sin(angle),
    }
}

/// `getAngle(dir)` (`vmath.ts:54-60`). Unused by the ported core files directly; kept for API
/// parity (the inverse of `direction`).
pub fn get_angle(dir: Vec2) -> f64 {
    if dir.x == 0.0 && dir.y == 0.0 {
        return 0.0;
    }
    if dir.x == 0.0 {
        return if dir.y < 0.0 { -js::PI / 2.0 } else { js::PI / 2.0 };
    }
    let mut result = js::atan(dir.y / dir.x);
    if dir.x < 0.0 {
        result += js::PI;
    }
    result
}

/// `closestPointOnLineOrNull(lineA, lineB, targetPoint)` (`vmath.ts:62-72`).
pub fn closest_point_on_line_or_null(line_a: Vec2, line_b: Vec2, target_point: Vec2) -> Option<Vec2> {
    let ab = vsub(line_b, line_a);
    let squared_magnitude_ab = vdot(ab, ab);
    if squared_magnitude_ab > 0.0 {
        let ap = vsub(target_point, line_a);
        let ap_dot_ab = vdot(ap, ab);
        let t = ap_dot_ab / squared_magnitude_ab;
        return Some(vadd(line_a, vmul(ab, clamp(t, 0.0, 1.0))));
    }
    None
}

/// `closestPointOnLine(lineA, lineB, targetPoint)` (`vmath.ts:74-76`).
pub fn closest_point_on_line(line_a: Vec2, line_b: Vec2, target_point: Vec2) -> Vec2 {
    closest_point_on_line_or_null(line_a, line_b, target_point).unwrap_or(line_a)
}

/// `clamp(val, min, max)` (`vmath.ts:78-82`). Deliberately `val < min` / `val > max` in that
/// order, exactly as TS wrote it (not `f64::clamp`, whose `NaN`/argument-order panics and
/// tie-breaking are Rust's own, not JS's — though no call site in the ported files ever passes a
/// `NaN`).
pub fn clamp(val: f64, min: f64, max: f64) -> f64 {
    if val < min {
        return min;
    }
    if val > max {
        return max;
    }
    val
}

/// `roundToInt(f)` (`vmath.ts:84-86`). **Not** `Math.round`: ties (`f == 0`, i.e. the `f > 0`
/// branch is false) go through `Math.trunc(f - 0.5)`, so `roundToInt(0) === -0`. This is one of
/// three distinct "round a coordinate to a tile-grid index" idioms `collision.ts` uses (the
/// other two are `Math.trunc(x)` directly in `indexAt`/`getMapIndex` and `>> 5` bit-shifting in
/// `indexAt`) — they are **not** unified here even though they look similar; see the crate
/// README "Известные причуды TS" for why keeping them distinct matters for negative coordinates.
pub fn round_to_int(f: f64) -> f64 {
    if f > 0.0 {
        js::trunc(f + 0.5)
    } else {
        js::trunc(f - 0.5)
    }
}

/// `sign(f)` (`vmath.ts:88-92`). Unused by the ported core files directly; kept for API parity.
pub fn sign(f: f64) -> f64 {
    if f > 0.0 {
        1.0
    } else if f < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vnormalize_zero_is_zero_not_nan() {
        let v = vnormalize(vec2(0.0, 0.0));
        assert_eq!(v, vec2(0.0, 0.0));
    }

    #[test]
    fn round_to_int_ties_go_negative_at_zero() {
        // f > 0 is false for f == 0.0, so it takes the trunc(f - 0.5) branch: trunc(-0.5) == -0.0.
        assert_eq!(round_to_int(0.0).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(round_to_int(0.4), 0.0);
        assert_eq!(round_to_int(0.5), 1.0);
        assert_eq!(round_to_int(-0.4).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(round_to_int(-0.5), -1.0);
        assert_eq!(round_to_int(1.5), 2.0);
        assert_eq!(round_to_int(-1.5), -2.0);
    }

    #[test]
    fn closest_point_on_line_degenerate_returns_line_a() {
        let a = vec2(5.0, 5.0);
        assert_eq!(closest_point_on_line(a, a, vec2(100.0, -3.0)), a);
        assert_eq!(closest_point_on_line_or_null(a, a, vec2(100.0, -3.0)), None);
    }

    #[test]
    fn closest_point_on_line_clamps_to_segment() {
        let p = closest_point_on_line(vec2(0.0, 0.0), vec2(10.0, 0.0), vec2(-5.0, 3.0));
        assert_eq!(p, vec2(0.0, 0.0));
        let p = closest_point_on_line(vec2(0.0, 0.0), vec2(10.0, 0.0), vec2(15.0, 3.0));
        assert_eq!(p, vec2(10.0, 0.0));
        let p = closest_point_on_line(vec2(0.0, 0.0), vec2(10.0, 0.0), vec2(4.0, 3.0));
        assert_eq!(p, vec2(4.0, 0.0));
    }

    #[test]
    fn sign_matches_ts() {
        assert_eq!(sign(5.0), 1.0);
        assert_eq!(sign(-5.0), -1.0);
        assert_eq!(sign(0.0), 0.0);
    }
}
