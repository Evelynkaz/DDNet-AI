//! The single static page (login form → status screen, acceptance criterion 7) and its CSS/JS,
//! embedded into the binary with `include_str!` so the running server never reads from disk (and
//! there is nothing resembling a directory to list). No routes besides these four exist for
//! static content, so any other path falls through to the router's default 404.

use axum::http::header;
use axum::response::IntoResponse;

const INDEX_HTML: &str = include_str!("../../assets/index.html");
const APP_CSS: &str = include_str!("../../assets/app.css");
const APP_JS: &str = include_str!("../../assets/app.js");
const FLY_JS: &str = include_str!("../../assets/fly.js");
const TRAIN_JS: &str = include_str!("../../assets/train.js");
const LAUNCH_JS: &str = include_str!("../../assets/launch.js");
const LAUNCH_CSS: &str = include_str!("../../assets/launch.css");
const SAY_JS: &str = include_str!("../../assets/say.js");
const SAY_CSS: &str = include_str!("../../assets/say.css");

pub async fn page() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], INDEX_HTML)
}

pub async fn css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS)
}

pub async fn js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], APP_JS)
}

/// Task 7.4: the fly panel's script (the «Муха» tab), loaded before `app.js`.
pub async fn fly_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], FLY_JS)
}

/// Task 5.8: the training panel's script (the «Обучение» tab), loaded before `app.js`.
pub async fn train_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], TRAIN_JS)
}

/// Task 5.9: the «Запуск» card's script (the «Бот» tab), loaded before `app.js`.
pub async fn launch_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], LAUNCH_JS)
}

/// Task 5.9: the «Запуск» card's styles (scoped under `.launch-card`).
pub async fn launch_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], LAUNCH_CSS)
}

/// Task 4.9: the chat input's script (`SayCard`, mounted by `app.js`; to be mounted under the chat panel of the «Игра» tab later).
pub async fn say_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], SAY_JS)
}

/// Task 4.9: the chat input's styles (scoped under `.say-card`).
pub async fn say_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], SAY_CSS)
}
