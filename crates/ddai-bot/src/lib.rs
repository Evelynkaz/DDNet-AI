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
//!   rope helpers), [`latency`], [`brains`], [`hooks`] (the navigation / wayblock / trek hook traits) and [`nav_hooks`] (their real bodies over
//!   `ddai-nav`, task 4.2: goto, follow, seek, home, the Copy Love Box wayblock, the freeze memory).
//!
//! The bot never writes chat (D-007): the client API has no call that takes text, and the unstick kill is the
//! `Cl_Kill` protocol message. The one exception, allowed by the owner (D-078, task 4.6, [`killfallback`]): the typed server
//! command `/kill`, sent only when a `Cl_Kill` had no effect (DDNet's `sv_kill_protection`). It never evades a kick or ban (D-016/D-037): the runner stops.

pub mod activity;
pub mod bot;
pub mod brains;
pub mod bridge;
pub mod clipper;
pub mod command;
pub mod console;
pub mod consts;
pub mod control;
pub mod hooks;
pub mod identity;
pub mod input;
pub mod killfallback;
pub mod latency;
pub mod mapgrid;
pub mod nav_hooks;
pub mod planning;
pub mod players;
pub mod reach;
pub mod runner;
pub mod seal_worker;
pub mod sent;
pub mod settings;
pub mod target;
pub mod tees;
pub mod unstick;
pub mod wander;
pub mod wb_guard;

pub use bot::{Bot, BotConfig, BotEvent, BotStats, Mode, Output, Status};
pub use brains::{BrainError, BrainKind, BrainOptions, make_brain};
pub use command::{BotCommand, CommandReply};
pub use ddai_botctl::names;
pub use ddai_botctl::relations;
pub use hooks::Hooks;
pub use relations::Relations;
