//! `ddai-botctl` (task 5.6): the part of the web control that the bot (`ddai-bot`) and the web unit (`ddai-web`)
//! must agree on, kept in one place so they cannot drift apart:
//!
//! - [`proto`]: the control-channel protocol v1 — the typed requests the web may send the bot and the reply it
//!   gets back (`docs/formats.md` §26). **There is no chat in it**: no variant carries text to the game server.
//! - [`names`]: the name folding of D-021 (exact matching after normalisation);
//! - [`relations`]: the friend / war / ignore / clan-war / clan-friend lists and their file, which the bot reads
//!   and the web editor writes.
//!
//! The crate has no dependency on the bot, the client or the web server (and none on the network).

pub mod names;
pub mod proto;
pub mod relations;
