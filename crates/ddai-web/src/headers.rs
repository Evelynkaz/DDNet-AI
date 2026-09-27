//! Security headers applied to every response (acceptance criterion 6): a strict CSP with no
//! inline scripts (all JS/CSS is served as separate static files), `X-Content-Type-Options`,
//! `X-Frame-Options`/`frame-ancestors`, `Referrer-Policy`, and (scoped to `/api/*` and `/ws`)
//! `Cache-Control: no-store`.

use axum::extract::Request;
use axum::http::{HeaderValue, header};
use axum::middleware::Next;
use axum::response::Response;
use tower_http::set_header::SetResponseHeaderLayer;

/// `connect-src 'self' wss:` matches acceptance criterion 6's wording exactly: `'self'` already
/// covers a same-origin `ws:`/`wss:` upgrade (browsers treat `ws`/`wss` as `http`/`https` for CSP
/// source matching), `wss:` is kept in addition per the spec text.
const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; \
                    connect-src 'self' wss:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

/// Applies the headers that must be on *every* response, page or API: CSP, nosniff,
/// X-Frame-Options, Referrer-Policy. `Cache-Control: no-store` is intentionally not here — it's
/// scoped to `/api/*` and `/ws` only, via [`no_store_layer`], since static assets and the page
/// itself are fine to let a browser cache normally.
pub fn common_layers<S>(router: axum::Router<S>) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(SetResponseHeaderLayer::overriding(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(CSP),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            header::X_FRAME_OPTIONS,
            HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            header::REFERRER_POLICY,
            HeaderValue::from_static("no-referrer"),
        ))
}

/// Middleware (not a static layer, since it must run regardless of which handler matched) that
/// sets `Cache-Control: no-store` on every response from the API/WS sub-router.
pub async fn no_store(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

#[cfg(test)]
mod tests {
    use super::CSP;

    #[test]
    fn csp_has_no_inline_script_allowance() {
        assert!(!CSP.contains("unsafe-inline"));
        assert!(!CSP.contains("unsafe-eval"));
    }

    #[test]
    fn csp_restricts_frame_ancestors_and_default_src() {
        assert!(CSP.contains("frame-ancestors 'none'"));
        assert!(CSP.contains("default-src 'self'"));
        assert!(CSP.contains("connect-src 'self' wss:"));
    }
}
