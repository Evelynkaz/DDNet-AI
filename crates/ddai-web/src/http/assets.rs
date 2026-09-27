//! The single static page (login form → status screen, acceptance criterion 7) and its CSS/JS,
//! embedded into the binary with `include_str!` so the running server never reads from disk (and
//! there is nothing resembling a directory to list). No routes besides these three exist for
//! static content, so any other path falls through to the router's default 404.

use axum::http::header;
use axum::response::IntoResponse;

const INDEX_HTML: &str = include_str!("../../assets/index.html");
const APP_CSS: &str = include_str!("../../assets/app.css");
const APP_JS: &str = include_str!("../../assets/app.js");

pub async fn page() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], INDEX_HTML)
}

pub async fn css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS)
}

pub async fn js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], APP_JS)
}
