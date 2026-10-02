//! The live map view (task 5.2a): a `FrameSource` abstraction producing per-tick world state, a
//! `MapScene` classifier built from `ddai_map`/`ddai_trace` map data, a compact binary wire frame
//! format, and one real `FrameSource` implementation — replaying Oracle B server traces (task
//! 1.5). See `docs/formats.md`'s new section for the wire formats and `crates/ddai-web/README.md`
//! for the architecture.

pub mod bot_source;
pub mod fly;
pub mod frame;
pub mod hub;
pub mod map_resolve;
pub mod replay;
pub mod scene;
pub mod source;
