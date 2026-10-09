//! Portable, bit-exact ports of the glibc 2.39 math functions DDNet-AI's physics depends on: the same
//! numbers on Linux and Windows (task 5.5a, D-047/D-127).
//!
//! # Why this crate exists
//!
//! `ddai-physics` is bit-exact with DDNet 20.1's C++ server because the server, on Linux, calls glibc's
//! `sinf`, `cosf`, `atanf`, `atan2f`, `powf` (and, in `double`, `atan2` and `log`), and on `linux-gnu`
//! Rust's `std` calls the very same functions (D-004). On Windows, `std` goes to the UCRT, whose last bits
//! differ: physics, search and prediction would silently diverge from the server. The `libm` crate is no
//! way out: its `powf` differs from glibc's in 9.7% of probes (D-004). So this crate re-implements the
//! glibc algorithms in plain Rust and every platform uses it.
//!
//! | function | glibc symbol | algorithm | licence of the source |
//! |---|---|---|---|
//! | [`sinf`], [`cosf`] | `sinf`, `cosf` | ARM optimized-routines `sincosf` (polynomial in double, 2-step reduction) | MIT |
//! | [`powf`] | `powf` | ARM optimized-routines `powf` (`log2` table + `exp2` table, in double) | MIT |
//! | [`log`] | `log` | ARM optimized-routines `log` (table, 128 intervals) | MIT |
//! | [`pow`] | `pow` | ARM optimized-routines `pow` (extended-precision log + exp tables) | MIT |
//! | [`atanf`], [`atan2f`] | `atanf`, `atan2f` | Sun fdlibm | Sun permissive notice |
//! | [`atan2`] | `atan2` | IBM Accurate Mathematical Library | LGPL-2.1+, used under GPL-3 (see below) |
//! | [`hypotf`], [`hypot`] | `hypotf`, `hypot` | glibc (Borges' correction for `hypot`) | LGPL-2.1+, used under GPL-3 |
//!
//! Every file carries the notice of its source; the repository `NOTICE` lists them all and gives the
//! reasoning for the LGPL parts: section 3 of the GNU LGPL 2.1 lets a licensee apply the ordinary GNU GPL
//! (here: version 3) instead, and DDNet-AI is GPL-3.0-only.
//!
//! # Which glibc, and why there are `fma` calls
//!
//! The reference is glibc 2.39 (Ubuntu 24.04) **on x86-64 with FMA3 and AVX2**, the variant glibc's `ifunc`
//! machinery selects on every x86 CPU of the last decade (the Oracle A/B traces were recorded on such a
//! CPU, and so are the DDNet servers). In that variant (`__sinf_fma`, `__powf_fma`, `__log_fma`,
//! `__ieee754_atan2_fma`, `__pow_fma`) GCC contracted `a + b * c` into fused multiply-adds, and the
//! ARM-derived `log`/`pow` code takes its `__FP_FAST_FMA` paths. Results differ from the plain-SSE2
//! variant in the last bit for a small fraction of inputs (549 of 10^7 random probes for `log`), so the
//! ports reproduce the **fused** operations, written as explicit `f64::mul_add` calls at exactly the places
//! the machine code of the shipped `libm.so.6` has them (found by reading its disassembly and confirmed by
//! probing). `f64::mul_add` is correctly rounded on every platform, so Windows gets the same bits.
//!
//! Consequences:
//!
//! * No `unsafe`, no `target_feature` tricks, no dependency (not even `libm`).
//! * Without the `fma` target feature (the default build, see D-001: no `target-cpu` in committed
//!   configuration) every `mul_add` is a call to the C library's `fma`, so these functions are slower than
//!   glibc's own on a baseline x86-64 build (measured in the task report); building with
//!   `-C target-feature=+fma` makes them as fast as glibc's. The results never depend on it.
//! * [`f32::mul_add`]-style fusion is *not* applied anywhere else in the repository; the physics
//!   arithmetic stays un-fused exactly like DDNet's C++ (see `ddai-physics`'s `real.rs`).
//!
//! # Proof
//!
//! `tests/glibc_probe.rs` compares every function with glibc through `std` (so it runs on Linux only):
//! all 2^32 `f32` inputs for `sinf`/`cosf`/`atanf`; 10^9 random probes (special-value lists, class-balanced
//! generators in `tests/common/mod.rs`) for the others; every integer aim vector in a +-4096 window for
//! `atan2(int, int)`. `tests/golden.rs` replays a fixed probe sequence on **every** platform and compares
//! FNV hashes recorded from glibc, which is what the Windows CI job runs.
//!
//! Special values follow glibc: signed zeros, subnormals, infinities, NaN (payload and sign of the result
//! included for the x86-64 SSE rules) and the overflow/underflow results; `errno` and the floating-point
//! exception flags are not reproduced.

mod atan2;
mod atan2f;
mod atanf;
mod hypot;
mod log;
mod pow;
mod powf;
mod sincosf;
mod tables;
mod util;

pub use atan2::atan2;
pub use atan2f::atan2f;
pub use atanf::atanf;
pub use hypot::{hypot, hypotf};
pub use log::log;
pub use pow::pow;
pub use powf::powf;
pub use sincosf::{cosf, sinf};
