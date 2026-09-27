//! Line-for-line Rust ports of the fdlibm routines V8 compiles into Node 24's `Math.sin`,
//! `Math.cos`, `Math.tanh`, `Math.atan`, `Math.atan2`, `Math.exp` and `Math.log` — see
//! `deps/v8/src/base/ieee754.cc` at the Node 24.21.0 / V8 13.6 tag, and the crate README for the
//! full citations, the V8/glibc build-flag investigation that explains *why* it is fdlibm and not
//! glibc, and the empirical proof against the real Node binary.
//!
//! Split into one file per function (plus `bits.rs` for the shared word-extraction helpers and
//! `expm1.rs` for the one private helper `tanh` needs), instead of one 3000-line file, so each
//! function's port stays next to its own license header and V8 line citation. Every constant
//! keeps its original name (`S1`, `PIO2_1T`, `AT`, ...) in `SCREAMING_SNAKE_CASE`, and every
//! helper keeps its original short, C-style name (`kernel_sin`, not `sine_kernel`), specifically
//! so a line-by-line diff against the V8 source stays possible — see the crate README.

#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_range_loop)]
// fdlibm deliberately splits some constants (pi/2, 1/ln2, 2/pi, ...) into multi-part
// hi/lo representations whose individual parts are *not* full-precision (that's the point: the
// split lets later arithmetic recover precision a single rounded constant would lose). Clippy's
// `approx_constant` only knows the single-part `std::f64::consts::*` value and flags these parts
// as "did you mean PI/FRAC_PI_2/LOG2_E" — substituting the suggested constant would silently
// change the bit pattern and break bit-exactness with V8, which is the entire point of this
// crate, so this lint is off for the whole module tree.
#![allow(clippy::approx_constant)]
// `x - x` is fdlibm's idiom for "NaN, and (on real hardware) raise the FP invalid-operation
// exception" for Inf/NaN inputs — not a bug, ported as-is from `deps/v8/src/base/ieee754.cc`.
#![allow(clippy::eq_op)]

mod atan;
mod bits;
mod exp;
mod expm1;
mod log;
mod tanh;
mod trig;

pub use atan::{atan, atan2};
pub use exp::exp;
pub use log::log;
pub use tanh::tanh;
pub use trig::{cos, sin};
