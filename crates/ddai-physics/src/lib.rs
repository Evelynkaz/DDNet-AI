//! `ddai-physics` will hold the Rust port of DDNet's server-side physics: character movement,
//! collision against the map's tile layers, tuning parameters (including tune zones), weapons,
//! and world stepping — parameterized over the floating-point scalar (`f32`/`f64`) so it can be
//! checked bit-for-bit against the C++ DDNet reference and against the legacy TypeScript bot.
//!
//! For now this crate holds only the plain-data map representation (see [`map`]) that the rest
//! of the parity infrastructure (`ddai-trace`, the C++ oracle) is built on. The physics port
//! itself lands in a later task (see `docs/PLAN.md` §1.3).

pub mod map;
