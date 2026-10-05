//! `ddai-controls` (task 8.2, D-014): the controls the fly is measured against -- an MLP and a
//! GRU that see exactly the fly's inputs (the ray-grid features and proprioception of
//! `ddai_fly::encoder`, flattened) and produce exactly its outputs (direction / jump / hook /
//! fire / aim logits), trained with the same loss ([`ddai_fly::bc`]), the same optimiser and the
//! same data by `ddai-train`.
//!
//! Everything is hand-written (no ML framework), like the fly's own backward pass:
//! [`mlp::Mlp`], [`gru::Gru`] behind the [`net::SeqNet`] trait, [`brain::ControlBrain`] as a
//! `ddai_brain::Brain`, and [`bundle`] for the checkpoint file the arena factory loads.

pub mod brain;
pub mod bundle;
pub mod features;
pub mod gru;
pub mod mlp;
pub mod net;
pub mod proposer;

pub use brain::{ControlBrain, ControlTemplate};
pub use bundle::{ControlBundle, load_control_bundle, save_control_bundle};
pub use net::{NetKind, SeqNet};
