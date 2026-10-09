//! Small `f64` 2D vector type + free functions, mirroring `src/core/vmath.ts` exactly (the same
//! semantics `ddai-tsworld::vmath` ports, duplicated here so this crate does not have to depend on
//! `ddai-tsworld` outside the `ts-parity` feature — see the crate's top-level doc comment).
//! Scoring/search always happens in `f64` here (acceptance criterion 2), even when the backing
//! [`crate::plan_world::PlanWorld`] is `ddai_physics::World<f32>`.

use ddai_jsmath as js;

/// `type Vec2 = { x: number; y: number }` (`vmath.ts:1`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

pub const fn vec2(x: f64, y: f64) -> Vec2 {
    Vec2 { x, y }
}

pub fn vadd(a: Vec2, b: Vec2) -> Vec2 {
    Vec2 {
        x: a.x + b.x,
        y: a.y + b.y,
    }
}

pub fn vsub(a: Vec2, b: Vec2) -> Vec2 {
    Vec2 {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

pub fn vdistance(a: Vec2, b: Vec2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    js::sqrt(dx * dx + dy * dy)
}

pub fn vdot(a: Vec2, b: Vec2) -> f64 {
    a.x * b.x + a.y * b.y
}

/// `closestPointOnLineOrNull` (`vmath.ts:62-72`).
pub fn closest_point_on_line_or_null(line_a: Vec2, line_b: Vec2, target_point: Vec2) -> Option<Vec2> {
    let ab = vsub(line_b, line_a);
    let squared_magnitude_ab = vdot(ab, ab);
    if squared_magnitude_ab > 0.0 {
        let ap = vsub(target_point, line_a);
        let t = vdot(ap, ab) / squared_magnitude_ab;
        let clamped = clamp(t, 0.0, 1.0);
        return Some(Vec2 {
            x: line_a.x + ab.x * clamped,
            y: line_a.y + ab.y * clamped,
        });
    }
    None
}

/// `clamp(val, min, max)` (`vmath.ts:78-82`) — deliberately `val < min`/`val > max` in that order,
/// not `f64::clamp` (whose panics/tie-breaking are Rust's own, not JS's).
pub fn clamp(val: f64, min: f64, max: f64) -> f64 {
    if val < min {
        return min;
    }
    if val > max {
        return max;
    }
    val
}

/// `v.floor() as i32` without the libm call (task 4.13): for `|v|` below 2^31 - 647 the truncating cast is in range, and
/// one less than it when the truncation rounded up (a negative non-integer). NaN and larger magnitudes take the plain route
/// (saturating cast, as before), so the result is the same for every input.
#[inline(always)]
pub fn floor_i32(v: f64) -> i32 {
    if v > -2_147_483_000.0 && v < 2_147_483_000.0 {
        let t = v as i32;
        t - i32::from(f64::from(t) > v)
    } else {
        v.floor() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_i32_equals_floor_then_cast() {
        let mut vals = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            31.999999999,
            -31.999999999,
            2_147_482_999.5,
            -2_147_482_999.5,
            2_147_483_000.0,
            2_147_483_647.0,
            2_147_483_648.0,
            -2_147_483_648.0,
            -2_147_483_649.0,
            1e300,
            -1e300,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        for _ in 0..500_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            vals.push((u - 0.5) * 2.0 * [1.0, 40.0, 1e4, 1e9, 4e9][(state % 5) as usize]);
            vals.push(f64::from_bits(state));
        }
        for v in vals {
            assert_eq!(floor_i32(v), v.floor() as i32, "{v}");
        }
    }

    #[test]
    fn distance_matches_pythagoras() {
        assert_eq!(vdistance(vec2(0.0, 0.0), vec2(3.0, 4.0)), 5.0);
    }

    #[test]
    fn closest_point_degenerate_segment_is_none() {
        let a = vec2(1.0, 1.0);
        assert_eq!(closest_point_on_line_or_null(a, a, vec2(9.0, 9.0)), None);
    }
}
