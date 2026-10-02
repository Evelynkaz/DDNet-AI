//! The web side of the bot control (task 5.6, D-070): the owner's commands and the friend / war / ignore editor.
//!
//! - [`client`]: talks to the bot's control socket (`ddai_bot::control`, `docs/formats.md` §26) — one connection per
//!   request, bounded in size and time, **connect only** (this process never creates or listens on that socket).
//! - [`relations`]: reads and writes the lists file (`<data-dir>/bot/relations.json`) with the same code the bot uses
//!   (`ddai_botctl::relations`), so the exact-match normalisation of D-021 is one implementation.
//!
//! The HTTP routes are in `crate::http::bot`. Everything here is for the logged-in owner only; names never go to a
//! log and never to anyone but the owner's browser.

pub mod client;
pub mod relations;
