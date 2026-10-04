//! HTTP handlers: the static page/assets, the login/logout/me JSON API and (task 5.6) the owner-only bot control API. The WebSocket upgrade
//! lives in [`crate::ws`] instead, since it's not really "HTTP" once upgraded.

pub mod assets;
pub mod bot;
pub mod launch;
pub mod login;
pub mod map;
pub mod train;
