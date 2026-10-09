//! Speed of the ported functions against the C library's (through `std`, glibc on Linux).
//!
//! `cargo run --release -p ddai-libm --example speed` (use `nice -n 15`, one thread). For every function
//! it prints ns/call in two settings: `throughput` (independent calls over an array of inputs, what a
//! search loop sees) and `latency` (every call needs the previous result, what a tick of physics sees).
//! The inputs are the kind the bot feeds the functions, not random bit patterns.

use std::hint::black_box;
use std::time::Instant;

const N: usize = 1 << 16;
const REPS: usize = 200;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn time<T>(mut f: impl FnMut() -> T) -> f64 {
    // best of 5 runs
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn row(name: &str, tp_ours: f64, tp_std: f64, lat_ours: f64, lat_std: f64) {
    println!(
        "{name:8} throughput {:6.2} vs {:6.2} ns ({:+5.0}%)   latency {:6.2} vs {:6.2} ns ({:+5.0}%)",
        tp_ours,
        tp_std,
        (tp_ours / tp_std - 1.0) * 100.0,
        lat_ours,
        lat_std,
        (lat_ours / lat_std - 1.0) * 100.0
    );
}

fn unary_f32(name: &str, xs: &[f32], ours: fn(f32) -> f32, std_: fn(f32) -> f32) {
    let calls = (N * REPS) as f64;
    let tp = |f: fn(f32) -> f32| {
        time(|| {
            let mut acc = 0f32;
            for _ in 0..REPS {
                for &x in xs {
                    acc += f(black_box(x));
                }
            }
            acc
        }) / calls
            * 1e9
    };
    let lat = |f: fn(f32) -> f32| {
        time(|| {
            let mut x = 0.3f32;
            for _ in 0..N * REPS / 8 {
                x = f(black_box(x)) * 0.5 + 0.3;
            }
            x
        }) / (calls / 8.0)
            * 1e9
    };
    row(name, tp(ours), tp(std_), lat(ours), lat(std_));
}

fn binary_f32(name: &str, xs: &[(f32, f32)], ours: fn(f32, f32) -> f32, std_: fn(f32, f32) -> f32) {
    let calls = (N * REPS) as f64;
    let tp = |f: fn(f32, f32) -> f32| {
        time(|| {
            let mut acc = 0f32;
            for _ in 0..REPS {
                for &(x, y) in xs {
                    acc += f(black_box(x), black_box(y));
                }
            }
            acc
        }) / calls
            * 1e9
    };
    let lat = |f: fn(f32, f32) -> f32| {
        time(|| {
            let mut x = 0.3f32;
            for _ in 0..N * REPS / 8 {
                x = f(black_box(x) + 1.5, black_box(0.7)) * 0.5 + 0.3;
            }
            x
        }) / (calls / 8.0)
            * 1e9
    };
    row(name, tp(ours), tp(std_), lat(ours), lat(std_));
}

fn unary_f64(name: &str, xs: &[f64], ours: fn(f64) -> f64, std_: fn(f64) -> f64) {
    let calls = (N * REPS) as f64;
    let tp = |f: fn(f64) -> f64| {
        time(|| {
            let mut acc = 0f64;
            for _ in 0..REPS {
                for &x in xs {
                    acc += f(black_box(x));
                }
            }
            acc
        }) / calls
            * 1e9
    };
    let lat = |f: fn(f64) -> f64| {
        time(|| {
            let mut x = 1.3f64;
            for _ in 0..N * REPS / 8 {
                x = f(black_box(x)) * 0.5 + 1.3;
            }
            x
        }) / (calls / 8.0)
            * 1e9
    };
    row(name, tp(ours), tp(std_), lat(ours), lat(std_));
}

fn binary_f64(name: &str, xs: &[(f64, f64)], ours: fn(f64, f64) -> f64, std_: fn(f64, f64) -> f64) {
    let calls = (N * REPS) as f64;
    let tp = |f: fn(f64, f64) -> f64| {
        time(|| {
            let mut acc = 0f64;
            for _ in 0..REPS {
                for &(x, y) in xs {
                    acc += f(black_box(x), black_box(y));
                }
            }
            acc
        }) / calls
            * 1e9
    };
    let lat = |f: fn(f64, f64) -> f64| {
        time(|| {
            let mut x = 1.3f64;
            for _ in 0..N * REPS / 8 {
                x = f(black_box(x) + 0.5, black_box(0.7)) * 0.5 + 1.3;
            }
            x
        }) / (calls / 8.0)
            * 1e9
    };
    row(name, tp(ours), tp(std_), lat(ours), lat(std_));
}

fn main() {
    let mut rng = Rng(7);
    // sin/cos/atan: angles and tangent-like values in the range the physics and the brain use.
    let angles: Vec<f32> = (0..N).map(|_| ((rng.unit() - 0.5) * 12.0) as f32).collect();
    let ratios: Vec<f32> = (0..N).map(|_| ((rng.unit() - 0.5) * 8.0) as f32).collect();
    let aim: Vec<(f32, f32)> = (0..N)
        .map(|_| {
            (
                ((rng.unit() - 0.5) * 2000.0) as f32,
                ((rng.unit() - 0.5) * 2000.0) as f32,
            )
        })
        .collect();
    let ramp: Vec<(f32, f32)> = (0..N)
        .map(|_| ((1.01 + rng.unit() * 2.0) as f32, ((rng.unit() - 0.2) * 6.0) as f32))
        .collect();
    let curv: Vec<f64> = (0..N).map(|_| f64::from((1.01 + rng.unit() * 2.0) as f32)).collect();
    let aim64: Vec<(f64, f64)> = (0..N)
        .map(|_| {
            (
                (rng.unit() * 2000.0 - 1000.0).round(),
                (rng.unit() * 2000.0 - 1000.0).round(),
            )
        })
        .collect();
    let pow64: Vec<(f64, f64)> = (0..N).map(|_| (rng.unit() * 20.0, rng.unit() * 5.0)).collect();

    println!("ddai-libm vs std (glibc): ns per call, best of 5");
    unary_f32("sinf", &angles, ddai_libm::sinf, f32::sin);
    unary_f32("cosf", &angles, ddai_libm::cosf, f32::cos);
    unary_f32("atanf", &ratios, ddai_libm::atanf, f32::atan);
    binary_f32("atan2f", &aim, ddai_libm::atan2f, f32::atan2);
    binary_f32("powf", &ramp, ddai_libm::powf, f32::powf);
    unary_f64("log", &curv, ddai_libm::log, f64::ln);
    binary_f64("atan2", &aim64, ddai_libm::atan2, f64::atan2);
    binary_f64("pow", &pow64, ddai_libm::pow, f64::powf);
}
