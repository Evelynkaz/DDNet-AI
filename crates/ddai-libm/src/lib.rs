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
//! | function | glibc symbol | algorithm | licence of the source file |
//! |---|---|---|---|
//! | [`sinf`], [`cosf`] | `sinf`, `cosf` | ARM optimized-routines `sincosf` as imported by glibc (polynomial in double, 2-step reduction) | FSF copyright, LGPL-2.1+, used under GPL-3 |
//! | [`powf`] | `powf` | ARM optimized-routines `powf` as imported by glibc (`log2` table + `exp2` table, in double) | same |
//! | [`log`] | `log` | ARM optimized-routines `log` as imported by glibc (table, 128 intervals) | same |
//! | [`pow`] | `pow` | ARM optimized-routines `pow` as imported by glibc (extended-precision log + exp tables) | same |
//! | [`atanf`], [`atan2f`] | `atanf`, `atan2f` | Sun fdlibm | Sun permissive notice |
//! | [`atan2`] | `atan2` | IBM Accurate Mathematical Library | FSF/IBM copyright, LGPL-2.1+, used under GPL-3 |
//! | [`hypotf`], [`hypot`] | `hypotf`, `hypot` | glibc (Borges' correction for `hypot`) | FSF copyright, LGPL-2.1+, used under GPL-3 |
//!
//! Every file carries the notice of its source; the repository `NOTICE` lists them all and gives the
//! reasoning for the LGPL parts: section 3 of the GNU LGPL 2.1 lets a licensee apply the ordinary GNU GPL
//! (here: version 3) instead, which the headers of those files do (the permission paragraphs are the GPL's,
//! the FSF/IBM copyright lines are kept, the changes are dated), and DDNet-AI is GPL-3.0. The ports were made
//! from glibc's copies of the ARM code, so that is the licence they follow; the Arm origin is in the headers.
//!
//! # Which glibc, and why there are `fma` calls
//!
//! The reference is glibc 2.39 (Ubuntu 24.04) **on x86-64 with FMA3 and AVX2**, the variant glibc's `ifunc`
//! machinery selects on every x86 CPU of the last decade (the Oracle A/B traces were recorded on such a
//! CPU, and so are the DDNet servers). In that variant (`__sinf_fma`, `__powf_fma`, `__log_fma`,
//! `__ieee754_atan2_fma`, `__pow_fma`) GCC contracted `a + b * c` into fused multiply-adds, and the
//! ARM-derived `log`/`pow` code takes its `__FP_FAST_FMA` paths. Results differ from the plain-SSE2
//! variant in the last bit for a small fraction of inputs (549 of 10^7 random probes for `log`), so the
//! ports reproduce the **fused** operations, written as explicit calls of a correctly rounded `fma` at exactly the places
//! the machine code of the shipped `libm.so.6` has them (found by reading its disassembly and confirmed by
//! probing). The `fma` is correctly rounded on every platform (see below), so Windows gets the same bits.
//!
//! Consequences:
//!
//! * No `unsafe`, no `target_feature` tricks, no dependency (not even `libm`).
//! * Which `fma` runs is decided once per process by [`fma_mode`]: the instruction when the build has the
//!   `fma` target feature (`-C target-feature=+fma`; then the functions are as fast as glibc's); else, on
//!   `linux-gnu` with an FMA CPU, glibc's own `fma` after it has agreed with [`soft_fma`] on a self-check of a few
//!   tens of thousands of corner-case operands; else (Windows, musl, macOS, a CPU without FMA) the crate's own
//!   integer [`soft_fma`], a correctly rounded fused multiply-add in safe Rust written for this crate. The reason:
//!   glibc's `fma` is the reference, mingw-w64's is known to be wrong in corner cases, and the MSVC UCRT's software
//!   path (CPUs without FMA3) is unchecked and cannot be checked on CI (its `fma` has not been shown to be wrong),
//!   so no other C library's `fma` is trusted: a deliberate safety margin, not a measured need. The results never depend on which of the three runs; [`self_test`] checks that, and the bot
//!   refuses to play if it fails.
//!   Without `+fma` on `linux-gnu` each fused operation is a library call, so the functions are 2 to 5 times slower
//!   than glibc's on the `fma`-heavy ones (no measurable effect on `World::step`); with [`soft_fma`] (about 54 ns per
//!   fused operation: always on Windows without `+fma`) 14 to 39 times slower (0.2 to 1 microsecond per call, about
//!   +7% on `World::step`; measured in D-127). The choice is made once per call of `sinf`, `powf`, ... (a generic
//!   parameter, not a branch per fused operation).
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
mod selftest;
mod sincosf;
mod softfma;
mod tables;
mod util;

pub use atan2::atan2;
pub use atan2f::atan2f;
pub use atanf::atanf;
pub use hypot::{hypot, hypotf};
pub use log::log;
pub use pow::pow;
pub use powf::powf;
pub use selftest::self_test;
pub use sincosf::{cosf, sinf};
pub use softfma::{FmaMode, fma, mode as fma_mode, mode_description as fma_mode_description, soft_fma};
