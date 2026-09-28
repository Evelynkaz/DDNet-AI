//! Literal, `f64` + `ddai_jsmath` Rust port of the legacy TS bot's `src/core/*` physics
//! (`SimWorld`, `CharacterCore`, `Collision`, `Projectile`/`Laser`, tuning, vmath, types) —
//! `Wranked1/DDNet-AI`, GPL-3.0, ported for DDNet-AI.
//!
//! # Why this crate exists (decision D-035)
//!
//! This is **not** the production physics (see `ddai-physics`, which is bit-exact with the C++
//! DDNet server instead — a different, incompatible goal). It exists purely so that a future
//! Rust port of the legacy planner (`src/plan/*.ts`) can be proven, bit for bit, to make the
//! *exact same decisions* the TS planner made while running on the TS physics it was written and
//! tuned against — including that physics's own quirks and divergences from real DDNet. Folding
//! TS's V8-specific numeric behavior and TS-only bugs into `ddai-physics::World` as feature flags
//! would complicate and risk the code every other task depends on for real gameplay; a separate,
//! disposable, test-only crate does not. See `docs/DECISIONS.md` D-035, D-003, D-018, D-020.
//!
//! # What "literal port" means here
//!
//! Every function in [`vmath`], [`tuning`], [`collision`], [`character_core`], [`projectile`] and
//! [`world`] is a direct translation of its `src/core/*.ts` counterpart — same operation order,
//! same TS-specific numeric quirks (reproduced, never "fixed" — see each module's doc comment and
//! the crate README's "Известные причуды TS"), same observable iteration order over tees. Every
//! JS-numeric operation goes through [`ddai_jsmath`] (never a raw `f64::powf`/`f64::round`/etc.),
//! per the crate's own README warning about LLVM constant-folding `pow` for literal operands.
//!
//! The one place this crate deliberately does **not** mirror TS's code shape is the object graph:
//! TS's `CharacterCore`/`Projectile`/`Laser` hold a live reference back to their `SimWorld`
//! (`this.world`), which safe Rust (no `unsafe`, no `Rc<RefCell<_>>` — `docs/CLAUDE.md`) cannot
//! express as a struct field. The methods that need it ([`character_core::CharacterCore::tick`]
//! and friends) are ported instead as [`world::SimWorld`] methods operating on a tee index — see
//! `character_core`'s and `world`'s module doc comments for the exact mapping. Every number these
//! methods compute, and the order they compute it in, is unchanged.
//!
//! # Zero `unsafe`, no allocation per tick in steady state
//!
//! This crate contains no `unsafe` code. [`world::SimWorld`] pre-reserves and reuses its scratch
//! buffers (the "all alive cores" id list, the tile-index list `handleTiles` walks, the
//! collision-sweep snapshot list `move_` builds) across `step()` calls, so once tee count and map
//! size stabilize, `step()` allocates nothing new — see the crate README's "Аллокации" section
//! for the one caveat (fresh episodes/species with growing tee counts still grow those buffers,
//! same as any `Vec` would).

pub mod character_core;
pub mod collision;
pub mod map_load;
pub mod projectile;
pub mod trace_ts;
pub mod tuning;
pub mod types;
pub mod vmath;
pub mod world;

pub use character_core::CharacterCore;
pub use collision::Collision;
pub use map_load::{LoadedTsMap, load_map_bytes};
pub use projectile::{EntityWorld, Laser, Projectile};
pub use types::{PlayerInput, ProjectileState, TeeState, WorldEvent};
pub use vmath::Vec2;
pub use world::{GrenadeSpec, SimState, SimWorld, SimWorldOptions};
