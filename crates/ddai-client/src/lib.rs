//! `ddai-client`: the DDNet 20.x client session (task 2.3).
//!
//! Two halves, matching the task's design ask ("sans-IO `Session` state machine ... + a thin
//! ... UDP driver in a new `ddai-client` crate"):
//!
//! - [`session`]: [`session::Session`], a sans-IO state machine over `ddai-net`'s low-level
//!   connection ([`ddai_net::conn::Connection`]) and message/snapshot layer — the join sequence,
//!   snapshot dispatch, `Sv_TuneParams`, and the guarded single outgoing path
//!   ([`allowlist`]) that keeps `Cl_Say` off the wire (decision D-007) except the one typed server
//!   command `/kill` (D-078, task 4.6; `ddai_net::server_command`). No sockets, no wall
//!   clock; every method takes an explicit `now: Duration` and returns bytes to send rather than
//!   sending them.
//! - [`driver`]: the real-time layer — [`driver::Client`], a background thread that owns a real
//!   `UdpSocket`, drives a [`session::Session`], handles map-cache file I/O, and implements the
//!   live-play policy (connection-rate limiting, reconnect backoff, redirect-follow-once, never
//!   auto-reconnecting after a kick/ban) — this is the crate's actual public entry point for
//!   playing against a real server (`Client::connect`).
//!
//! Everything else is support code: [`timing`] (input-timing/prediction-margin, ported from
//! `client.cpp`'s `NETMSG_INPUTTIMING` feedback loop), [`smooth_time`] (the `CSmoothTime` port
//! that backs it), [`allowlist`] (the outgoing-message guard), and [`map_cache`] (the on-disk map
//! cache's path/IO helpers).
//!
//! # Scope: 0.6+DDNet only, live play restricted to 127.0.0.1
//!
//! This crate never speaks the 0.7/"sixup" protocol translation DDNet 20.1's client also supports
//! (out of scope for this whole project — see `docs/PLAN.md`), and — per `CLAUDE.md`'s live-play
//! policy — its own [`driver::Client`] only ever opens a UDP socket to addresses the caller gives
//! it; nothing in this crate reaches out to any fixed hostname or IP on its own. HTTPS map
//! download (`NETMSG_MAP_DETAILS.url`) is deliberately **not** implemented for that same reason:
//! the url a server offers always points at a public map mirror (e.g. `maps.ddnet.org`), and this
//! project's live-play rule is 127.0.0.1 only for this task — see [`session::SessionEvent::MapChanging`]'s
//! docs for how that decision is still surfaced to a caller that might one day run this crate
//! against a server it *is* allowed to fetch from.

pub mod allowlist;
pub mod driver;
pub mod favourites;
pub mod live_servers;
pub mod map_cache;
pub mod proxy;
pub mod relay_probe;
pub mod relay_rule;
pub mod safe_file;
pub mod server_list;
pub mod session;
pub mod single_instance;
pub mod smooth_time;
pub mod socks5;
#[cfg(any(test, feature = "test-util"))]
pub mod socks5_testserver;
pub mod timing;
pub mod transport;

pub use ddai_net::generated::enums;
pub use ddai_net::generated::objects::PlayerInput;
pub use ddai_net::view;
pub use driver::{
    Client, ClientEvent, GaveUpCategory, InputState, InputTag, LiveWorldSnapshot, MAX_HOLD_TICKS, MAX_LATE_PRESS_TICKS,
    next_fire_counter,
};
pub use session::{ClientConfig, MapLoadedEvent, MapSource, ServerCapabilities, Session, SessionEvent};
pub use timing::{MarginStats, MarginSummary};
