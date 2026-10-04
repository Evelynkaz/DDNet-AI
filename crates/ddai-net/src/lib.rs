//! `ddai-net`: transport-independent low-level Teeworlds 0.6 + DDNet network protocol.
//!
//! This crate implements the layer below game messages and snapshots (those come in task
//! 2.2b): Huffman compression, integer/string packing, packet and chunk (de)serialisation, the
//! control handshake including DDNet's `TKEN` security-token negotiation, a sans-IO reliable
//! delivery connection state machine, and DDNet UUID-based extended message ids.
//!
//! Byte layouts are documented in `docs/formats.md` (§"Протокол 0.6+DDNet: низкий уровень") and
//! in doc comments on the relevant types below, both citing the DDNet 20.1 C++ source
//! (`src/engine/shared/{network,huffman,packer,compression,uuid_manager}.{h,cpp}`,
//! pinned rev `c9d208138f85755521f16a0096b6fe036c5c8698`) that this module was ported from.
//!
//! Decision D-029: this is our own safe implementation. DDNet's C++ and libtw2 (MIT/Apache) are
//! references and test oracles (see `tests/`), never vendored into this crate.
//!
//! Nothing in this crate panics on attacker-controlled input — every parsing function returns a
//! `Result`/`Option` and every loop is bounded by the size of its input or output buffer. See
//! `tests/robustness.rs` for the fuzz-style proof.

pub mod control;
pub mod huffman;
pub mod packer;
pub mod packet;
pub mod uuid;

pub mod conn;

pub mod assembly;
pub mod delta;
pub mod generated;
pub mod intstr;
pub mod message;
pub mod owner_chat;
pub mod server_command;
pub mod serverinfo;
pub mod snapshot;
pub mod sysmsg;
pub mod tuning;
pub mod view;

pub use huffman::Huffman;
