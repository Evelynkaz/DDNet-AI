//! `ddai-world` (task 2.4): turns the client's snapshot stream (task 2.2b/2.3) into a predicted
//! `ddai_physics::World<f32>` at the tick the bot's own inputs will land on ("PredTick"), and
//! builds the `ddai_brain::Observation` the fly/planner decide from.
//!
//! A dedicated crate rather than a module of `ddai-client` or `ddai-brain`: it sits strictly
//! *between* the two (network snapshot in, `Observation` out) and depends on both plus
//! `ddai-physics`, while neither of those needs to depend on it — folding it into either would
//! give one of them a dependency it does not otherwise need (`ddai-client` has no reason to know
//! about `ddai-brain::Observation`; `ddai-brain` has no reason to know about the network protocol
//! at all). It mirrors the existing `ddai-fly`/`ddai-brain` split: `ddai-brain` is the
//! plain-data contract, `ddai-fly` and (now) `ddai-world` are two independent producers/consumers
//! of it.
//!
//! - [`reckoning`]: reconstructs a character's *exact* current-tick physics core from the lossy,
//!   dead-reckoned data a `CNetObj_Character` actually carries (`character.cpp:859-868`/
//!   `gameclient.cpp:1727-1742`).
//! - [`live_world`]: [`LiveWorld`] itself — snapshot ingestion, prediction, `Observation` building.
//! - [`projectiles`]: snapshot projectile items -> physics `Projectile`s, the way the DDNet client's
//!   prediction builds them (task 2.4b — replaces the map-spawned cannons of `World::from_map`).
//! - [`accuracy`]: the "measurement mode" acceptance criterion 1 asks for — logs predicted-vs-
//!   actual differences and summarizes them (own tee: bit-exact fraction; others: an error-px
//!   distribution at chosen horizons).

pub mod accuracy;
pub mod live_world;
pub mod projectiles;
pub mod reckoning;

pub use live_world::{
    LiveWorld, character_observation, correct_late_inputs, player_input_from_net, player_input_to_net,
    retarget_late_inputs,
};
