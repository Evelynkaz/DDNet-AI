#![warn(missing_docs)]
//! The Rust port of DDNet 20.1's server-side character-core physics: collision against the
//! map's tile layers ([`collision`]), tuning parameters ([`tuning`]), the character/world core
//! ([`core`]) and a stepping helper that reproduces Oracle A's per-tick loop exactly
//! ([`core_world`]) — all parameterized over the floating-point scalar ([`real::Real`],
//! `f32`/`f64`) so the `f32` instantiation can be checked bit-for-bit against the C++ DDNet
//! reference (see `docs/formats.md` and `docs/DECISIONS.md` D-002/D-003) while `f64` compiles
//! and runs for later TS-compat parity work.
//!
//! [`map`] holds the plain-data map representation (tile layers, no logic) everything else here
//! is built from; [`prng`] and [`vmath`] are small supporting ports ([`prng::Prng`]'s PCG-XSH-RR
//! algorithm, [`vmath::Vec2`] and its free functions).
//!
//! [`world`] is the server-level layer on top (task 1.6): the DDRace character logic (tiles, freeze,
//! teams, switches), every weapon — hammer, gun, grenade, shotgun and laser (`CLaser`), ninja — the
//! map entities (pickups, draggers and their beams, turrets and their plasma, lights, doors) and the
//! server's per-tick order, bit-exact against the real server (Oracle B, `docs/formats.md` §11-§12,
//! §30). Out of scope there, with no observable effect on any compared field: `/rescue`, `/pause`,
//! team locking/flocking and practice mode.

/// Port of 20.1 `CCollision` (`src/game/collision.{h,cpp}`).
pub mod collision;
/// Port of 20.1 `CCharacterCore`/`CWorldCore`/`CTeamsCore` (`src/game/gamecore.{h,cpp}`,
/// `src/game/teamscore.{h,cpp}`).
pub mod core;
// `core_world` documents itself via its own `//!` (module-inner) doc comment; an outer `///`
// here as well would create a second, differently-scoped copy of that doc that rustdoc merges
// in a way that breaks its intra-doc links (verified) — same reasoning for `map`/`real` below.
pub mod core_world;
pub mod map;
/// Port of 20.1 `CPrng` (`src/game/prng.{h,cpp}`).
pub mod prng;
pub mod real;
// `switch` documents itself via its own `//!` (module-inner) doc comment — same reasoning as
// `core_world`/`map`/`real` above (task 1.6: switch-layer behavior, `CDoor` collision setup and
// timed-switch expiry, built on top of task 1.3's [`core::Switcher`] placeholder).
pub mod switch;
/// Port of 20.1 `CTuningParams`/`CTuneParam` (`src/game/tuning.h`, `src/game/gamecore.h`).
pub mod tuning;
/// Port of the subset of 20.1 `base/vmath.h`/`base/math.h` this crate needs.
pub mod vmath;
// `world` documents itself via its own `//!` (module-inner) doc comment — same reasoning as
// `core_world`/`map`/`real`/`switch` above (task 1.6, Stage A: `World<R>`, the server-level
// DDRace simulation).
pub mod world;
