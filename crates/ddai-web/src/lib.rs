//! `ddai-web`: the bot's own web server (task 5.1 — the security skeleton).
//!
//! An axum + WebSocket HTTP server, meant to be embedded in the `ddnet-ai` binary, listening only
//! on loopback (Caddy terminates HTTPS in front of it later — task 5.3). This crate is the
//! skeleton: password login (argon2id), server-side cookie sessions, brute-force protection,
//! CSRF/Origin checks, security headers, request limits, and an authenticated WebSocket carrying
//! small typed JSON control messages. The actual game view (map, players, commands) is a later
//! task; here the page is a minimal login + status shell that proves the whole auth/WS path
//! works.
//!
//! Entry points for `ddnet-ai`:
//! - [`bind`] + [`run`]: start serving (`ddnet-ai web`).
//! - [`secrets::generate_and_store_password`]: `ddnet-ai web-passwd`.

pub mod auth;
pub mod config;
pub mod control;
pub mod headers;
pub mod http;
pub mod launch;
pub mod live;
pub mod origin;
pub mod rand_util;
pub mod secrets;
pub mod server;
pub mod session_guard;
pub mod state;
pub mod toml_kv;
pub mod training;
pub mod ws;

pub use config::WebConfig;
pub use server::{BindError, Bound, bind, build_router, run};
