//! `ddai-flyg`: the `.flyg` v1 binary format — the fly's connectome subgraph (topology, synapse
//! signs, receptive fields, output groups) that the fly model (phase 7) loads.
//!
//! This crate defines the format, (de)serializes it, and validates it. It does **not** know how
//! to *select* a subgraph from the raw MaleCNS connectome — that lives in `ddai-connectome`'s
//! `subgraph` module, which depends on this crate (not the other way around). This crate has no
//! dependency on `ddai-connectome` or `arrow`, by design: it is meant to be the one thing the fly
//! model links against, without pulling in the heavy Arrow/Feather reading machinery that only
//! the offline subgraph-building step needs. See `README.md` for the file-level picture and
//! `docs/formats.md` (main repo, Russian) for the on-disk format writeup.

pub mod error;
pub mod format;
pub mod io;
pub mod validate;

pub use error::{FlygError, ValidationError};
pub use format::*;
pub use io::{load, save};
pub use validate::validate;
