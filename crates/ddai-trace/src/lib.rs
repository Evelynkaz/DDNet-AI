//! Parity infrastructure for porting DDNet physics to Rust bit-exactly.
//!
//! This crate holds the file formats and generators shared between Rust and the C++ "Oracle A"
//! harness (`tools/ddnet-oracle`), so the two sides can exchange maps, scenarios and traces
//! without either side depending on the other's code:
//!
//! - [`rawmap`]: a plain little-endian encoding of [`ddai_physics::map::MapData`].
//! - [`synthetic`]: deterministic hand-built test maps (`synthetic::build`).
//! - [`scenario`]: a map reference, world/tuning setup and explicit per-tick inputs.
//! - [`generator`]: `random-v1`, a deterministic "interesting inputs" scenario generator.
//! - [`trace`]: the applied input and resulting `CCharacterCore` state, per tick per character,
//!   as written by the oracle and read back for comparison in task 1.3.
//! - [`hash`]: FNV-1a 64 (the canonical per-tick state hash) and a thin sha256 wrapper.
//!
//! See `docs/formats.md` for the exact byte layouts.

pub mod generator;
pub mod hash;
pub mod io;
pub mod rawmap;
pub mod scenario;
pub mod synthetic;
pub mod trace;

mod prng;

pub use prng::SplitMix64;
