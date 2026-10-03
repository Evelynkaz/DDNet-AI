//! Memoised `sin`/`cos`/`atan2` for the planner's per-plan-step hot path (task 3.6).
//!
//! The JS-exact transcendental functions (`ddai_jsmath`, V8's fdlibm) cost tens of nanoseconds each, and
//! one rollout step calls half a dozen of them (the aim decode, the executed aim, the hook gate) with
//! arguments that come back again and again: the same plan is rolled out under several model
//! combinations, and an aim turns by whole integer targets. Each function below is a small per-thread
//! direct-mapped cache in front of the very same `ddai_jsmath` function, keyed by the exact bit pattern
//! of the argument(s), so a hit returns the value the function would compute, bit for bit.

use std::cell::Cell;

use ddai_jsmath as js;

const SLOTS: usize = 256;

type Slot1 = Cell<(u64, u64, bool)>;
type Slot2 = Cell<(u64, u64, u64, bool)>;

thread_local! {
    static COS: [Slot1; SLOTS] = const { [const { Cell::new((0, 0, false)) }; SLOTS] };
    static SIN: [Slot1; SLOTS] = const { [const { Cell::new((0, 0, false)) }; SLOTS] };
    static ATAN2: [Slot2; SLOTS] = const { [const { Cell::new((0, 0, 0, false)) }; SLOTS] };
}

fn slot1(bits: u64) -> usize {
    (bits.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize
}

fn memo1(table: &'static std::thread::LocalKey<[Slot1; SLOTS]>, x: f64, f: fn(f64) -> f64) -> f64 {
    let bits = x.to_bits();
    table.with(|t| {
        let slot = &t[slot1(bits)];
        let (key, value, valid) = slot.get();
        if valid && key == bits {
            return f64::from_bits(value);
        }
        let v = f(x);
        slot.set((bits, v.to_bits(), true));
        v
    })
}

/// `ddai_jsmath::cos`, memoised.
pub fn cos(x: f64) -> f64 {
    memo1(&COS, x, js::cos)
}

/// `ddai_jsmath::sin`, memoised.
pub fn sin(x: f64) -> f64 {
    memo1(&SIN, x, js::sin)
}

/// `ddai_jsmath::atan2`, memoised.
pub fn atan2(y: f64, x: f64) -> f64 {
    let (yb, xb) = (y.to_bits(), x.to_bits());
    let i = slot1(yb ^ xb.rotate_left(29).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    ATAN2.with(|t| {
        let slot = &t[i];
        let (ky, kx, value, valid) = slot.get();
        if valid && ky == yb && kx == xb {
            return f64::from_bits(value);
        }
        let v = js::atan2(y, x);
        slot.set((yb, xb, v.to_bits(), true));
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hit returns exactly what the function computes: random arguments (including repeats, collisions
    /// within a slot, NaN, infinities, signed zeros) against the plain `ddai_jsmath` functions.
    #[test]
    fn memoised_trig_is_bit_identical_to_jsmath() {
        let mut s = 0xDEAD_BEEF_0BAD_F00Du64;
        let mut next = || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let specials = [
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            -1.0,
            300.0,
            1e-300,
            1e300,
        ];
        let mut recent: Vec<f64> = Vec::new();
        for k in 0..100_000u32 {
            let mut pick = |recent: &mut Vec<f64>| {
                let v = match next() % 5 {
                    0 if !recent.is_empty() => recent[(next() % recent.len() as u64) as usize],
                    1 => specials[(next() % specials.len() as u64) as usize],
                    2 => (next() % 2001) as f64 - 1000.0,
                    _ => (next() as f64 / u64::MAX as f64 - 0.5) * 20.0,
                };
                recent.push(v);
                if recent.len() > 50 {
                    recent.remove(0);
                }
                v
            };
            let (a, b) = (pick(&mut recent), pick(&mut recent));
            assert_eq!(cos(a).to_bits(), js::cos(a).to_bits(), "cos {k}: {a}");
            assert_eq!(sin(a).to_bits(), js::sin(a).to_bits(), "sin {k}: {a}");
            assert_eq!(atan2(a, b).to_bits(), js::atan2(a, b).to_bits(), "atan2 {k}: {a} {b}");
        }
    }
}
