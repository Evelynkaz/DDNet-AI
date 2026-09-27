//! Library half of `ddai-connectome`: fetch/verify the MaleCNS connectome minimal file set and
//! read its Arrow Feather tables into compact internal tables. See `README.md` for usage and
//! `docs/research/fly-data.md` / `docs/FLY.md` §2 for the data itself.
//!
//! Split into a library so integration tests (`tests/`) can call the read/verify logic directly
//! without spawning the CLI binary, and so `fetch`'s network code stays out of anything that must
//! run offline.

pub mod goog_hash;
pub mod hashing;
pub mod manifest;
pub mod verify;

pub mod fetch;
pub mod inspect;
pub mod stats;
pub mod subgraph;
pub mod tables;
