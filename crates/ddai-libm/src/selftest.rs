//! A start-up self-test (task 5.5a review, F2): the arithmetic the ports rest on gives glibc's bits on this
//! machine, checked in a few milliseconds. The bot runs it before it plays and refuses to start if it fails.
//!
//! * the `fma` the ports use is compared with the crate's own integer `fma` on the corner-case operands
//!   (this is also what [`crate::fma_mode`] decides on);
//! * each of the ten functions is evaluated on a fixed pseudo-random sequence of inputs in the ranges the bot
//!   uses (angles, aim vectors, distances, velocity-ramp bases and exponents) and the results are hashed;
//!   the hashes were recorded from glibc 2.39 (test `the_expected_hashes_are_glibcs` re-derives them on Linux).

use crate::{atan2, atan2f, atanf, cosf, hypot, hypotf, log, pow, powf, sinf};

/// Probes per function.
const N: usize = 20_000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }

    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        self.f64_in(f64::from(lo), f64::from(hi)) as f32
    }
}

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn put(&mut self, bits: u64) {
        for byte in bits.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

type Body = Box<dyn FnMut(&mut Rng, &mut Fnv)>;

/// Hash of `N` evaluations of `body` on the generator seeded with `seed`.
fn run(seed: u64, mut body: Body) -> u64 {
    let mut rng = Rng(seed);
    let mut h = Fnv::new();
    for _ in 0..N {
        body(&mut rng, &mut h);
    }
    h.0
}

/// One function of the test: its name, the hash of its results on the fixed inputs, with `f` supplying the
/// function so that a test can feed `std` (glibc) instead.
pub(crate) struct Fns {
    pub sinf: fn(f32) -> f32,
    pub cosf: fn(f32) -> f32,
    pub atanf: fn(f32) -> f32,
    pub atan2f: fn(f32, f32) -> f32,
    pub powf: fn(f32, f32) -> f32,
    pub hypotf: fn(f32, f32) -> f32,
    pub log: fn(f64) -> f64,
    pub atan2: fn(f64, f64) -> f64,
    pub pow: fn(f64, f64) -> f64,
    pub hypot: fn(f64, f64) -> f64,
}

pub(crate) const OURS: Fns = Fns {
    sinf,
    cosf,
    atanf,
    atan2f,
    powf,
    hypotf,
    log,
    atan2,
    pow,
    hypot,
};

/// `(name, hash)` for every function.
pub(crate) fn hashes(f: &Fns) -> [(&'static str, u64); 10] {
    let b32 = |h: &mut Fnv, v: f32| h.put(u64::from(v.to_bits()));
    let b64 = |h: &mut Fnv, v: f64| h.put(v.to_bits());
    let (sinf_f, cosf_f, atanf_f, atan2f_f, powf_f, hypotf_f) = (f.sinf, f.cosf, f.atanf, f.atan2f, f.powf, f.hypotf);
    let (log_f, atan2_f, pow_f, hypot_f) = (f.log, f.atan2, f.pow, f.hypot);
    [
        (
            "sinf",
            run(
                1,
                Box::new(move |r, h| {
                    let x = if r.next() & 3 == 0 {
                        r.f32_in(-1.0e6, 1.0e6)
                    } else {
                        r.f32_in(-4000.0, 4000.0)
                    };
                    b32(h, sinf_f(x));
                }),
            ),
        ),
        (
            "cosf",
            run(
                2,
                Box::new(move |r, h| {
                    let x = if r.next() & 3 == 0 {
                        r.f32_in(-1.0e6, 1.0e6)
                    } else {
                        r.f32_in(-4000.0, 4000.0)
                    };
                    b32(h, cosf_f(x));
                }),
            ),
        ),
        (
            "atanf",
            run(3, Box::new(move |r, h| b32(h, atanf_f(r.f32_in(-100.0, 100.0))))),
        ),
        (
            "atan2f",
            run(
                4,
                Box::new(move |r, h| {
                    let (y, x) = (r.f32_in(-1000.0, 1000.0), r.f32_in(-1000.0, 1000.0));
                    b32(h, atan2f_f(y, x));
                }),
            ),
        ),
        (
            "powf",
            run(
                5,
                Box::new(move |r, h| {
                    let (x, y) = (r.f32_in(0.0, 3.0), r.f32_in(-8.0, 8.0));
                    b32(h, powf_f(x, y));
                }),
            ),
        ),
        (
            "hypotf",
            run(
                6,
                Box::new(move |r, h| {
                    let (x, y) = (r.f32_in(-1000.0, 1000.0), r.f32_in(-1000.0, 1000.0));
                    b32(h, hypotf_f(x, y));
                }),
            ),
        ),
        (
            "log",
            run(
                7,
                Box::new(move |r, h| {
                    let x = r.f64_in(-30.0, 30.0).exp2();
                    b64(h, log_f(x));
                }),
            ),
        ),
        (
            "atan2",
            run(
                8,
                Box::new(move |r, h| {
                    // Integer aim vectors, as `CCharacterCore::Tick` passes them.
                    let (y, x) = (r.f64_in(-8192.0, 8192.0).trunc(), r.f64_in(-8192.0, 8192.0).trunc());
                    b64(h, atan2_f(y, x));
                }),
            ),
        ),
        (
            "pow",
            run(
                9,
                Box::new(move |r, h| {
                    let (x, y) = (r.f64_in(0.0, 5.0), r.f64_in(-20.0, 20.0));
                    b64(h, pow_f(x, y));
                }),
            ),
        ),
        (
            "hypot",
            run(
                10,
                Box::new(move |r, h| {
                    let (x, y) = (r.f64_in(-1.0e4, 1.0e4), r.f64_in(-1.0e4, 1.0e4));
                    b64(h, hypot_f(x, y));
                }),
            ),
        ),
    ]
}

/// FNV hashes of the results on the fixed inputs, recorded from glibc 2.39 (x86-64, FMA variants).
pub(crate) const EXPECTED: [(&str, u64); 10] = [
    ("sinf", 0x106c99b85c5f75f3),
    ("cosf", 0x0a31c43dae83766f),
    ("atanf", 0xfc6c0ab1d1a5b053),
    ("atan2f", 0x8533d99b1d7f0fd8),
    ("powf", 0x5ff69a1667e6c6d4),
    ("hypotf", 0x61c834ee2a4dc88d),
    ("log", 0x5e0fa9a68188c14f),
    ("atan2", 0x16b1daae94752de3),
    ("pow", 0x2047daeaff8ecff2),
    ("hypot", 0x1041c3763ca281b5),
];

/// Check that this machine computes glibc's bits: the `fma` first, then every function. `Ok` carries the
/// description of the `fma` in use (for the log); `Err` says what differs.
///
/// # Errors
///
/// A description of the first failing check.
pub fn self_test() -> Result<&'static str, String> {
    let fma_in_use = crate::fma_mode_description();
    for (a, b, c) in crate::softfma::check_operands() {
        let (want, got) = (crate::soft_fma(a, b, c), crate::fma(a, b, c));
        if !(want.to_bits() == got.to_bits() || (want.is_nan() && got.is_nan())) {
            return Err(format!(
                "ddai-libm self-test: fma({a:e}, {b:e}, {c:e}) = {got:e} ({:#x}), expected {want:e} ({:#x}); fma in use: {fma_in_use}",
                got.to_bits(),
                want.to_bits()
            ));
        }
    }
    for ((name, got), (_, want)) in hashes(&OURS).into_iter().zip(EXPECTED) {
        if got != want {
            return Err(format!(
                "ddai-libm self-test: {name} gives hash {got:#018x}, glibc gave {want:#018x}; fma in use: {fma_in_use}"
            ));
        }
    }
    Ok(fma_in_use)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_self_test_passes() {
        let how = self_test().unwrap_or_else(|e| panic!("{e}"));
        println!("self-test ok, fma: {how}");
    }

    #[test]
    fn it_notices_a_function_that_is_off_in_the_last_bit() {
        let mut f = OURS;
        f.powf = |x, y| {
            let r = powf(x, y);
            if x.to_bits() % 1000 == 0 {
                f32::from_bits(r.to_bits() ^ 1)
            } else {
                r
            }
        };
        let off: Vec<_> = hashes(&f)
            .into_iter()
            .zip(EXPECTED)
            .filter(|((_, a), (_, b))| a != b)
            .map(|((n, _), _)| n)
            .collect();
        assert_eq!(off, ["powf"]);
    }

    #[test]
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn the_expected_hashes_are_glibcs() {
        let std_fns = Fns {
            sinf: f32::sin,
            cosf: f32::cos,
            atanf: f32::atan,
            atan2f: f32::atan2,
            powf: f32::powf,
            hypotf: f32::hypot,
            log: f64::ln,
            atan2: f64::atan2,
            pow: f64::powf,
            hypot: f64::hypot,
        };
        let wrong: Vec<String> = hashes(&std_fns)
            .into_iter()
            .zip(EXPECTED)
            .filter(|((_, got), (_, want))| got != want)
            .map(|((name, got), _)| format!("(\"{name}\", {got:#018x}),"))
            .collect();
        assert!(
            wrong.is_empty(),
            "glibc's hashes differ from EXPECTED:\n{}",
            wrong.join("\n")
        );
    }
}
