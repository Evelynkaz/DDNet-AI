//! V8-bit-exact ports of the JavaScript numeric primitives the legacy TS bot (`Wranked1/DDNet-AI`)
//! relies on, plus a port of its deterministic `Rng` (`src/nn/rng.ts`).
//!
//! # Why this crate exists
//!
//! `ddai-physics` (the `TsCompat` profile, task 1.9) and `ddai-planner` (task 3.2) both need to
//! reproduce, bit for bit, what the old TS planner computed while running on V8 (pinned to Node
//! 24.21.0, decision D-020). `docs/PLAN.md` §1.1 originally put this module (`jsmath`) inside
//! `ddai-planner`, but `ddai-physics` cannot depend on `ddai-planner` (the dependency graph in
//! §1.1 is `physics ← world ← brain ← {planner, fly}`, the other way around) and yet the
//! `TsCompat` physics profile needs the exact same V8 semantics (`Math.round`'s tie-breaking,
//! `%` on doubles, etc.) that the planner's scoring needs. This crate is the refinement: a leaf
//! crate with **no dependencies of its own**, so both `ddai-physics` and `ddai-planner` can depend
//! on it without creating a cycle.
//!
//! # What is ported, and from where
//!
//! Two different sources back the two families of functions here (see each module's doc comment
//! and the crate README for the full citations and the proof methodology):
//!
//! - [`semantics`]: functions whose *value* matches Rust `std`/hardware IEEE-754 ops exactly, but
//!   whose *edge-case semantics* differ from Rust (tie-breaking, NaN propagation, signed zero,
//!   `ToInt32`/`ToUint32`, JS's specific multi-argument `hypot`/`max`/`min` reduction). Ported
//!   from V8's Torque builtins (`src/builtins/math.tq`) and CodeStubAssembler macros
//!   (`src/codegen/code-stub-assembler.cc`), not from libm.
//! - [`fdlibm`] (private) + [`pow`]: the transcendental functions (`sin`, `cos`, `tanh`, `atan`,
//!   `atan2`, `exp`, `log`), ported line-for-line from V8's own copy of fdlibm
//!   (`src/base/ieee754.cc`, BSD/Sun-licensed, see `NOTICE`), because Node's build of V8 does
//!   **not** set `v8_use_libm_trig_functions` (verified against `deps/v8/gni/v8.gni`,
//!   `deps/v8/BUILD.gn` and Node's own `tools/v8_gypfiles/v8.gyp`, which never references
//!   `third_party/glibc` — see the crate README "Какая реализация у каждой функции"). `pow` is the
//!   one exception: V8's `Math.pow` (`src/numbers/ieee754.cc`, guarded by the
//!   `v8_flags.use_std_math_pow` runtime flag, default `true`) calls `std::pow` directly (with two
//!   special cases), which on this glibc/Linux target is bit-identical to Rust's `f64::powf` —
//!   both ultimately call the same libm `pow` symbol — so [`pow::pow`] does not need an fdlibm
//!   port, only the two special cases and the NaN/Infinity-exponent pre-checks V8 adds.
//!
//! [`rng`] ports `Rng` from `src/nn/rng.ts` (splitmix32 seeding, xoshiro128**, `nextFloat`,
//! Box-Muller `nextGaussian` with the carried spare) plus the planner's separate opponent-seed LCG
//! step.
//!
//! # No dependencies, no `unsafe`
//!
//! This crate has zero runtime dependencies (not even `libm`) and contains no `unsafe` code (the
//! fdlibm bit-twiddling that the original C++ does through pointer aliasing is done here through
//! [`f64::to_bits`]/[`f64::from_bits`], which are safe). See the crate README for the full
//! rationale, the V8/fdlibm/glibc license notices this carries, and the proof methodology
//! (`tools/jsmath-oracle`) that backs the "bit-exact" claim.

mod fdlibm;
mod pow;
mod rng;
mod semantics;

pub use pow::pow;
pub use rng::{Rng, opp_seed_next};
pub use semantics::{
    PI, abs, ceil, floor, hypot, hypot2, imul, js_max as max, js_min as min, max_n, min_n, rem, round, shl, shr, sign,
    sqrt, to_int32, to_uint32, trunc, ushr,
};

// The transcendental fdlibm ports are exposed at the crate root under their own names (not
// re-exported bare, to keep `sin`/`cos`/etc. discoverable via `jsmath::sin` while their
// implementation stays private to `fdlibm`, matching how V8 itself splits "semantics" builtins
// (Torque) from "transcendental" ones (ieee754.cc)).
pub use fdlibm::{atan, atan2, cos, exp, log, sin, tanh};
