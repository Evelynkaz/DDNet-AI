//! Authentication and authorization building blocks: password hashing lives in
//! [`crate::secrets`], everything else (sessions, the signed cookie, login rate limiting, CSRF
//! token checks) lives here.

pub mod cookie;
pub mod csrf;
pub mod device;
pub mod rate_limit;
pub mod session;
