//! `ddai-bot` (task 4.1): the live DDNet block bot around the brain.
//!
//! - [`bot`]: the sans-IO state machine ([`bot::Bot`]) — the per-snapshot pipeline, modes, brain
//!   wiring, post-filters;
//! - [`runner`]: the real-time shell around `ddai_client::Client` (snapshot collapsing, the wall
//!   clock, protocol actions, shutdown, the D-037 stop conditions);
//! - [`bridge`]: the read-only live-state feed to the web unit over a Unix socket;
//! - the parts: [`relations`] / [`names`] / [`players`] (lists, exact matching, log hashes),
//!   [`activity`] (AFK clock, block attribution), [`target`] (`pickTarget`), [`reach`], [`unstick`],
//!   [`wander`], [`input`] (fire counter), [`sent`] (in-flight inputs), [`planning`] (shield / seal /
//!   rope helpers), [`latency`], [`brains`], [`hooks`] (no-op hooks for task 4.2).
//!
//! The bot never writes chat (D-007): the client API has no way to, and the unstick kill is the
//! `Cl_Kill` protocol message. It never evades a kick or ban (D-016/D-037): the runner stops.

pub mod activity;
pub mod bot;
pub mod brains;
pub mod bridge;
pub mod consts;
pub mod hooks;
pub mod input;
pub mod latency;
pub mod mapgrid;
pub mod names;
pub mod planning;
pub mod players;
pub mod reach;
pub mod relations;
pub mod runner;
pub mod seal_worker;
pub mod sent;
pub mod target;
pub mod tees;
pub mod unstick;
pub mod wander;

pub use bot::{Bot, BotConfig, BotEvent, BotStats, Mode, Output, Status};
pub use brains::{BrainError, BrainKind, BrainOptions, make_brain};
pub use hooks::Hooks;
pub use relations::Relations;
