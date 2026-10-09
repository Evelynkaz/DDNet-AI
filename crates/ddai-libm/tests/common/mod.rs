//! Deterministic, platform-independent probe inputs shared by the glibc probe harness (Linux) and the golden
//! test (every platform). Only integer operations, bit casts and basic IEEE arithmetic are used here, never
//! a `libm` function, so the same seed produces bit-identical inputs on every platform.
#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;

/// splitmix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    pub fn f32_any(&mut self) -> f32 {
        f32::from_bits(self.next() as u32)
    }

    pub fn f64_any(&mut self) -> f64 {
        f64::from_bits(self.next())
    }

    /// Random sign, binary exponent uniform in `lo..hi` (within the normal range), random mantissa: a
    /// log-uniform magnitude.
    pub fn log_uniform_f32(&mut self, lo: i32, hi: i32) -> f32 {
        let e = lo + (self.next() % (hi - lo) as u64) as i32;
        let sign = ((self.next() & 1) as u32) << 31;
        f32::from_bits(sign | (((127 + e) as u32) << 23) | (self.next() as u32 & 0x007f_ffff))
    }

    pub fn log_uniform_f64(&mut self, lo: i32, hi: i32) -> f64 {
        let e = lo + (self.next() % (hi - lo) as u64) as i32;
        let sign = self.next() & 1 << 63;
        f64::from_bits(sign | (((1023 + e) as u64) << 52) | (self.next() & 0x000f_ffff_ffff_ffff))
    }

    /// Log-uniform and positive.
    pub fn log_uniform_pos_f32(&mut self, lo: i32, hi: i32) -> f32 {
        self.log_uniform_f32(lo, hi).abs()
    }

    pub fn log_uniform_pos_f64(&mut self, lo: i32, hi: i32) -> f64 {
        self.log_uniform_f64(lo, hi).abs()
    }
}

/// `2^e` exactly (`e` in the normal range).
pub fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// Special `f32` values: zeros, subnormals, extremes, infinities, NaNs (quiet, signalling, payloads, both
/// signs), powers of two with their neighbours, and the neighbourhood of every threshold the functions
/// branch on.
pub fn special_f32() -> Vec<f32> {
    let mut v: Vec<f32> = Vec::new();
    let bits: &[u32] = &[
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x0000_0001, // min subnormal
        0x8000_0001,
        0x0000_0100,
        0x007f_ffff, // max subnormal
        0x807f_ffff,
        0x0080_0000, // min normal
        0x8080_0000,
        0x0080_0001,
        0x7f7f_ffff, // max finite
        0xff7f_ffff,
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_0000, // quiet NaN
        0xffc0_0000, // quiet NaN, negative (x86 default NaN)
        0x7fa0_0000, // signalling NaN
        0xffa0_0000,
        0x7f80_0001, // signalling NaN, smallest payload
        0x7fff_ffff,
        0xffff_ffff,
        0x7fc0_1234, // quiet NaN with payload
    ];
    v.extend(bits.iter().map(|&b| f32::from_bits(b)));
    // Powers of two (subnormal ones included) and the values on either side of them.
    for e in -149..=127i32 {
        for m in [1.0f32, -1.0] {
            let x = if e >= -126 {
                f32::from_bits(((127 + e) as u32) << 23)
            } else {
                f32::from_bits(1 << (e + 149))
            } * m;
            v.push(x);
            v.push(f32::from_bits(x.to_bits().wrapping_add(1)));
            v.push(f32::from_bits(x.to_bits().wrapping_sub(1)));
        }
    }
    // Thresholds: pi/4, pi/2 multiples, 120, 2^25, atan breakpoints, 1.0, 0.5, 2^-12, 2^-29, 126/127/128, 150.
    for t in [
        std::f32::consts::FRAC_PI_4,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        std::f32::consts::TAU,
        9.424_778,
        120.0,
        119.999_99,
        120.000_01,
        33_554_432.0,
        0.4375,
        0.6875,
        1.1875,
        2.4375,
        1.0,
        0.5,
        2.0,
        3.0,
        10.0,
        100.0,
        1e5,
        1e10,
        1e20,
        1e30,
        1e38,
        0.000_244_140_62,
        1.862_645_1e-9,
        126.0,
        127.0,
        128.0,
        129.0,
        149.0,
        150.0,
        -149.0,
        -150.0,
        0.1,
        0.9,
        1.01,
        1.1,
    ] {
        for m in [1.0f32, -1.0] {
            let x = t * m;
            v.push(x);
            v.push(f32::from_bits(x.to_bits().wrapping_add(1)));
            v.push(f32::from_bits(x.to_bits().wrapping_sub(1)));
        }
    }
    // All small integers and half-integers.
    for i in -300..=300 {
        v.push(i as f32);
        v.push(i as f32 + 0.5);
    }
    v
}

pub fn special_f64() -> Vec<f64> {
    let mut v: Vec<f64> = Vec::new();
    let bits: &[u64] = &[
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x800f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x8010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0x7ff4_0000_0000_0000, // signalling NaN
        0xfff4_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7fff_ffff_ffff_ffff,
        0xffff_ffff_ffff_ffff,
        0x7ff8_0000_dead_beef,
    ];
    v.extend(bits.iter().map(|&b| f64::from_bits(b)));
    for e in -1074..=1023i32 {
        for m in [1.0f64, -1.0] {
            let x = if e >= -1022 {
                pow2(e)
            } else {
                f64::from_bits(1u64 << (e + 1074))
            } * m;
            v.push(x);
            v.push(f64::from_bits(x.to_bits().wrapping_add(1)));
            v.push(f64::from_bits(x.to_bits().wrapping_sub(1)));
        }
    }
    for t in [
        1.0 - 1.0 / 16.0,
        1.0 + 1.0 / 16.0,
        1.0 + 0.0645,
        0.5,
        2.0,
        3.0,
        std::f64::consts::E,
        std::f64::consts::PI,
        std::f64::consts::FRAC_PI_2,
        10.0,
        100.0,
        1e5,
        1e10,
        1e100,
        1e300,
        1.01,
        1.0625,
        0.9375,
        1.0 / 16.0,
        0.0625,
        1.0 + pow2(-52),
        1.0 - pow2(-53),
        709.0,
        709.78,
        710.0,
        -745.0,
        -746.0,
        1024.0,
        1075.0,
        -1075.0,
    ] {
        for m in [1.0f64, -1.0] {
            let x = t * m;
            v.push(x);
            v.push(f64::from_bits(x.to_bits().wrapping_add(1)));
            v.push(f64::from_bits(x.to_bits().wrapping_sub(1)));
        }
    }
    for i in -300..=300 {
        v.push(f64::from(i));
        v.push(f64::from(i) + 0.5);
    }
    v
}

// ---------------------------------------------------------------------------------------------
// Input generators, one per function family. `i` is the probe index; the generators cycle through their
// input classes so that every class gets an equal share of any prefix of probes.
// ---------------------------------------------------------------------------------------------

/// sinf, cosf, atanf.
pub fn gen_f32_unary(rng: &mut Rng, i: u64) -> f32 {
    match i % 8 {
        0 | 1 => rng.f32_any(),
        2 => rng.uniform(-8.0, 8.0) as f32,
        3 => rng.uniform(-130.0, 130.0) as f32,
        4 | 5 => rng.log_uniform_f32(-40, 40),
        6 => {
            // Next to a multiple of pi/2 (worst cases of the range reduction).
            let k = (rng.next() % 200_000) as f64 - 100_000.0;
            let base = (k * FRAC_PI_2) as f32;
            f32::from_bits(base.to_bits().wrapping_add((rng.next() % 9) as u32).wrapping_sub(4))
        }
        // Large arguments, the slow range reduction.
        _ => rng.log_uniform_f32(6, 127),
    }
}

pub fn gen_powf(rng: &mut Rng, i: u64) -> (f32, f32) {
    match i % 9 {
        0 => (rng.f32_any(), rng.f32_any()),
        1 => (rng.log_uniform_pos_f32(-30, 30), rng.uniform(-30.0, 30.0) as f32),
        2 => (rng.uniform(0.0, 4.0) as f32, rng.uniform(-8.0, 8.0) as f32),
        // x close to 1 and large exponents (the log2 table edge, and the |y log2 x| ~ 126 boundary).
        3 => (rng.uniform(0.99, 1.01) as f32, rng.uniform(-5000.0, 5000.0) as f32),
        // negative base, integer exponents (even/odd handling).
        4 => (
            -(rng.uniform(0.0, 50.0) as f32),
            rng.uniform(-40.0, 40.0).round_ties_even() as f32,
        ),
        // negative base, arbitrary exponent (domain error -> NaN).
        5 => (-(rng.uniform(0.0, 50.0) as f32), rng.uniform(-40.0, 40.0) as f32),
        // DDNet's VelocityRamp: curvature in [1.01, 3], exponent (value - start) / range, mostly small.
        6 => (rng.uniform(1.01, 3.0) as f32, rng.uniform(-2.0, 8.0) as f32),
        // overflow / underflow boundaries: y chosen so that |y log2 x| is about 127.9 or 149.5.
        7 => {
            let x = rng.log_uniform_pos_f32(-20, 20);
            let target = if rng.next() & 1 == 0 { 127.9 } else { -149.5 };
            let y = (target / log2_approx(f64::from(x))) as f32;
            (
                x,
                f32::from_bits(y.to_bits().wrapping_add((rng.next() % 5) as u32).wrapping_sub(2)),
            )
        }
        // subnormal bases
        _ => (
            f32::from_bits((rng.next() as u32) & 0x00ff_ffff),
            rng.uniform(-3.0, 3.0) as f32,
        ),
    }
}

/// log2 of a positive finite number to a few digits, from the exponent bits and a short series: only used
/// to place probes near a boundary, so it need not be exact, only deterministic (pure IEEE arithmetic).
fn log2_approx(x: f64) -> f64 {
    let b = x.to_bits();
    let e = ((b >> 52) & 0x7ff) as i64 - 1023;
    let m = f64::from_bits((b & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000); // [1, 2)
    // ln(m) via atanh series: ln m = 2 atanh((m-1)/(m+1))
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let ln_m = 2.0 * t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0)))));
    e as f64 + ln_m * std::f64::consts::LOG2_E
}

pub fn gen_atan2f(rng: &mut Rng, i: u64) -> (f32, f32) {
    match i % 8 {
        0 => (rng.f32_any(), rng.f32_any()),
        1 => (rng.uniform(-1000.0, 1000.0) as f32, rng.uniform(-1000.0, 1000.0) as f32),
        // whole numbers like DDNet's aim vectors
        2 => (
            rng.uniform(-3000.0, 3000.0).round_ties_even() as f32,
            rng.uniform(-3000.0, 3000.0).round_ties_even() as f32,
        ),
        3 => (rng.log_uniform_f32(-60, 60), rng.log_uniform_f32(-60, 60)),
        // |y/x| around 2^60 (the shortcut branches)
        4 => (rng.log_uniform_f32(-10, 10), rng.log_uniform_f32(-70, -50)),
        5 => (rng.log_uniform_f32(-70, -50), rng.log_uniform_f32(-10, 10)),
        // x == 1.0 delegates to atanf
        6 => (rng.log_uniform_f32(-30, 30), 1.0),
        // y/x near the atan breakpoints
        _ => {
            let x = rng.uniform(-100.0, 100.0) as f32;
            let t = [0.4375f64, 0.6875, 1.1875, 2.4375][(rng.next() % 4) as usize];
            let y = f64::from(x) * t;
            (
                f32::from_bits(
                    (y as f32)
                        .to_bits()
                        .wrapping_add((rng.next() % 5) as u32)
                        .wrapping_sub(2),
                ),
                x,
            )
        }
    }
}

pub fn gen_hypotf(rng: &mut Rng, i: u64) -> (f32, f32) {
    match i % 6 {
        0 => (rng.f32_any(), rng.f32_any()),
        1 => (rng.uniform(-1000.0, 1000.0) as f32, rng.uniform(-1000.0, 1000.0) as f32),
        // pixel-like positions and differences of them
        2 => (
            rng.uniform(-100_000.0, 100_000.0) as f32,
            rng.uniform(-100_000.0, 100_000.0) as f32,
        ),
        3 => (rng.log_uniform_f32(-126, 127), rng.log_uniform_f32(-126, 127)),
        // very different magnitudes (the sum of squares rounds to the larger square)
        4 => (rng.log_uniform_f32(0, 12), rng.log_uniform_f32(-30, -10)),
        // overflow of x*x + y*y in float, not in double
        _ => (rng.log_uniform_f32(60, 127), rng.log_uniform_f32(60, 127)),
    }
}

pub fn gen_hypot(rng: &mut Rng, i: u64) -> (f64, f64) {
    match i % 8 {
        0 => (rng.f64_any(), rng.f64_any()),
        1 => (rng.uniform(-1000.0, 1000.0), rng.uniform(-1000.0, 1000.0)),
        2 => (f64::from(rng.next() as i32 >> 8), f64::from(rng.next() as i32 >> 8)),
        3 => (rng.log_uniform_f64(-1022, 1023), rng.log_uniform_f64(-1022, 1023)),
        // |y| / |x| around 2^-54 (the shortcut) and the huge / tiny scaling thresholds (2^511, 2^-459)
        4 => {
            let x = rng.log_uniform_f64(-100, 100);
            (
                x,
                x * f64::from_bits(((1023 - 54 + (rng.next() % 5) as i32 - 2) as u64) << 52),
            )
        }
        5 => (rng.log_uniform_f64(505, 520), rng.log_uniform_f64(400, 520)),
        6 => (rng.log_uniform_f64(-470, -450), rng.log_uniform_f64(-470, -440)),
        // x <= y and x > y both, nearly equal operands
        _ => {
            let x = rng.uniform(1.0, 2.0);
            (
                x,
                f64::from_bits(x.to_bits().wrapping_add(rng.next() % 9).wrapping_sub(4)),
            )
        }
    }
}

pub fn gen_log(rng: &mut Rng, i: u64) -> f64 {
    match i % 8 {
        0 => rng.f64_any(),
        1 => rng.uniform(0.0, 100.0),
        2 => rng.log_uniform_pos_f64(-1022, 1023),
        // close to 1.0 (the special polynomial): [1 - 2^-4, 1 + 0x1.09p-4] and a bit beyond
        3 | 4 => rng.uniform(0.92, 1.07),
        5 => 1.0 + rng.log_uniform_f64(-60, -3),
        // ln of the clamped f32 curvature DDNet feeds it
        6 => f64::from(rng.uniform(1.01, 5.0) as f32),
        // subnormals
        _ => f64::from_bits(rng.next() & 0x000f_ffff_ffff_ffff),
    }
}

pub fn gen_atan2(rng: &mut Rng, i: u64) -> (f64, f64) {
    match i % 10 {
        0 => (rng.f64_any(), rng.f64_any()),
        // integer pairs (`std::atan2(int, int)` promotes both to double): all of +-2^16 and beyond
        1 | 2 => (f64::from(rng.next() as i32 >> 16), f64::from(rng.next() as i32 >> 16)),
        3 => (f64::from(rng.next() as i32), f64::from(rng.next() as i32)),
        4 => (rng.uniform(-1000.0, 1000.0), rng.uniform(-1000.0, 1000.0)),
        5 => (rng.log_uniform_f64(-1000, 1000), rng.log_uniform_f64(-1000, 1000)),
        // |y/x| near the 1/16 threshold, near 1, and near the table cell edges
        6 | 7 => {
            let x = rng.uniform(-50.0, 50.0);
            let t = match rng.next() % 4 {
                0 => 1.0 / 16.0,
                1 => 1.0,
                2 => 16.0,
                _ => 1.0 / 16.0 + (rng.next() % 241) as f64 / 256.0,
            };
            let y = x * t;
            (
                f64::from_bits(y.to_bits().wrapping_add(rng.next() % 5).wrapping_sub(2)),
                x,
            )
        }
        // extreme exponent differences and scaling thresholds (2^+-500)
        8 => (rng.log_uniform_f64(-1022, 1023), rng.log_uniform_f64(-1022, 1023)),
        _ => (rng.log_uniform_f64(-520, -480), rng.log_uniform_f64(-520, 520)),
    }
}

pub fn gen_pow(rng: &mut Rng, i: u64) -> (f64, f64) {
    match i % 10 {
        0 => (rng.f64_any(), rng.f64_any()),
        1 => (rng.log_uniform_pos_f64(-100, 100), rng.uniform(-100.0, 100.0)),
        2 => (rng.uniform(0.0, 4.0), rng.uniform(-8.0, 8.0)),
        // x close to 1 and large exponents
        3 => (rng.uniform(0.999, 1.001), rng.uniform(-50_000.0, 50_000.0)),
        4 => (-rng.uniform(0.0, 50.0), rng.uniform(-60.0, 60.0).round_ties_even()),
        5 => (-rng.uniform(0.0, 50.0), rng.uniform(-60.0, 60.0)),
        // JS-style: small integers and fractions (the planner's scoring)
        6 => (rng.uniform(0.0, 20.0).round_ties_even(), rng.uniform(0.0, 6.0)),
        // overflow / underflow boundaries
        7 => {
            let x = rng.log_uniform_pos_f64(-30, 30);
            let target = match rng.next() % 3 {
                0 => 1023.9,
                1 => -1074.5,
                _ => -1022.5,
            };
            let y = target / log2_approx(x);
            (
                x,
                f64::from_bits(y.to_bits().wrapping_add(rng.next() % 5).wrapping_sub(2)),
            )
        }
        8 => (
            f64::from_bits(rng.next() & 0x001f_ffff_ffff_ffff),
            rng.uniform(-3.0, 3.0),
        ),
        _ => (rng.uniform(0.0, 3.0), rng.log_uniform_f64(-70, 70)),
    }
}

/// Second operands every special value is paired with.
pub const F32_PARTNERS: &[f32] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    3.0,
    0.5,
    -0.5,
    1.5,
    10.0,
    -10.0,
    100.5,
    126.0,
    -126.0,
    127.0,
    128.0,
    -149.0,
    -150.0,
    1e30,
    -1e30,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(0x7fa0_0000),
    24.0,
    25.0,
    16_777_216.0,
    33_554_432.0,
    1.0e-30,
    -1.0e-30,
];

pub const F64_PARTNERS: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    3.0,
    0.5,
    -0.5,
    1.5,
    10.0,
    -10.0,
    100.5,
    1023.0,
    -1074.0,
    1024.0,
    -1075.0,
    1e300,
    -1e300,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::from_bits(0x7ff4_0000_0000_0000),
    1e-300,
    -1e-300,
    9007199254740992.0,
    4503599627370496.0,
    1e19,
];

/// All functions of the crate, with the seed of their probe sequence.
pub const NAMES: &[&str] = &[
    "sinf", "cosf", "atanf", "atan2f", "powf", "hypotf", "log", "atan2", "pow", "hypot",
];

pub fn seed_of(name: &str) -> u64 {
    // FNV-1a of the name.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in name.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Calls `sink(args, index)` for every input of `name`'s probe sequence: first the special values and
/// their pairings, then `n` generated probes. `args` holds the operand bit patterns (one or two).
pub fn for_each_input(name: &str, n: u64, sink: &mut dyn FnMut(&[u64])) {
    let mut rng = Rng(seed_of(name));
    match name {
        "sinf" | "cosf" | "atanf" => {
            for x in special_f32() {
                sink(&[u64::from(x.to_bits())]);
            }
            for i in 0..n {
                let x = gen_f32_unary(&mut rng, i);
                sink(&[u64::from(x.to_bits())]);
            }
        }
        "powf" | "atan2f" | "hypotf" => {
            let sp = special_f32();
            for &x in &sp {
                for &y in F32_PARTNERS {
                    sink(&[u64::from(x.to_bits()), u64::from(y.to_bits())]);
                }
            }
            for (i, &y) in sp.iter().enumerate() {
                for &x in F32_PARTNERS {
                    sink(&[u64::from(x.to_bits()), u64::from(y.to_bits())]);
                }
                if i % 7 == 0 {
                    for &x in sp.iter().step_by(13) {
                        sink(&[u64::from(x.to_bits()), u64::from(y.to_bits())]);
                    }
                }
            }
            for i in 0..n {
                let (a, b) = if name == "powf" {
                    gen_powf(&mut rng, i)
                } else {
                    gen_atan2f(&mut rng, i)
                };
                sink(&[u64::from(a.to_bits()), u64::from(b.to_bits())]);
            }
        }
        "log" => {
            for x in special_f64() {
                sink(&[x.to_bits()]);
            }
            for i in 0..n {
                sink(&[gen_log(&mut rng, i).to_bits()]);
            }
        }
        "atan2" | "pow" | "hypot" => {
            let sp = special_f64();
            for &x in &sp {
                for &y in F64_PARTNERS {
                    sink(&[x.to_bits(), y.to_bits()]);
                }
            }
            for (i, &y) in sp.iter().enumerate() {
                for &x in F64_PARTNERS {
                    sink(&[x.to_bits(), y.to_bits()]);
                }
                if i % 9 == 0 {
                    for &x in sp.iter().step_by(17) {
                        sink(&[x.to_bits(), y.to_bits()]);
                    }
                }
            }
            if name == "atan2" {
                // Every aim vector DDNet's physics can feed it in a small window, exhaustively.
                for ty in -200..=200 {
                    for tx in -200..=200 {
                        sink(&[f64::from(ty).to_bits(), f64::from(tx).to_bits()]);
                    }
                }
            }
            for i in 0..n {
                let (a, b) = if name == "atan2" {
                    gen_atan2(&mut rng, i)
                } else {
                    gen_pow(&mut rng, i)
                };
                sink(&[a.to_bits(), b.to_bits()]);
            }
        }
        other => panic!("unknown function {other}"),
    }
}

/// Our implementation of `name` applied to the operand bits `args`; the result as bits.
pub fn ours(name: &str, args: &[u64]) -> u64 {
    let f = |i: usize| f32::from_bits(args[i] as u32);
    let d = |i: usize| f64::from_bits(args[i]);
    match name {
        "sinf" => u64::from(ddai_libm::sinf(f(0)).to_bits()),
        "cosf" => u64::from(ddai_libm::cosf(f(0)).to_bits()),
        "atanf" => u64::from(ddai_libm::atanf(f(0)).to_bits()),
        "atan2f" => u64::from(ddai_libm::atan2f(f(0), f(1)).to_bits()),
        "hypotf" => u64::from(ddai_libm::hypotf(f(0), f(1)).to_bits()),
        "powf" => u64::from(ddai_libm::powf(f(0), f(1)).to_bits()),
        "log" => ddai_libm::log(d(0)).to_bits(),
        "atan2" => ddai_libm::atan2(d(0), d(1)).to_bits(),
        "pow" => ddai_libm::pow(d(0), d(1)).to_bits(),
        "hypot" => ddai_libm::hypot(d(0), d(1)).to_bits(),
        other => panic!("unknown function {other}"),
    }
}

/// FNV-1a over the little-endian bytes of `v`, continuing from `h`.
pub fn fnv(mut h: u64, v: u64) -> u64 {
    for b in v.to_le_bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Probes per hashed block of the golden file.
pub const GOLDEN_BLOCK: usize = 4096;
/// Generated probes per function in the golden file (the special-value part comes on top).
pub const GOLDEN_PROBES: u64 = 262_144;
