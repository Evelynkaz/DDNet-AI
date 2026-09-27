// GENERATED — do not edit by hand.
//
// Produced by `tools/ddnet-protocol-gen/generate.py` from DDNet's own protocol description
// (`datasrc/network.py` + `datasrc/datatypes.py`), commit c9d208138f85755521f16a0096b6fe036c5c8698 ("20.1").
// Regenerate with (from the repository root):
//
//   python3 tools/ddnet-protocol-gen/generate.py ~/aiddnet/build/ddnet-20.1/src
//
// Re-running against the same pinned commit's tree reproduces these files byte-for-byte (the
// script formats its own output with `rustfmt`). See `tools/ddnet-protocol-gen/README.md`.
//! Rust code generated from DDNet 20.1's own protocol description
//! (`datasrc/network.py` + `datasrc/datatypes.py`) by `tools/ddnet-protocol-gen/generate.py`.
//! See that script and `tools/ddnet-protocol-gen/README.md` for how to regenerate.
//!
//! This module (and everything under it) contains no hand-written protocol logic — only
//! mechanical field lists, ids, and the encode/decode/validate functions `datasrc/compile.py`'s
//! own templates would produce, ported to Rust. Hand-written logic that *uses* this (message
//! dispatch, snapshot delta unpacking, the typed view API, ...) lives in the parent `ddai_net`
//! crate modules, never here.

pub mod enums;
pub mod messages;
pub mod objects;
