//! HTTP handlers: the static page/assets and the login/logout/me JSON API. The WebSocket upgrade
//! lives in [`crate::ws`] instead, since it's not really "HTTP" once upgraded.

pub mod assets;
pub mod login;
pub mod map;
