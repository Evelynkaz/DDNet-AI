//! `GET /assets/<path>`: the DDNet client graphics the game view draws with (task 5.10), served at runtime from the
//! DDNet data directory (`--ddnet-data`, the `data/` of a DDNet 20.1 install or build) — **never copied into the
//! repository**: DDNet's art is CC BY-SA 3.0 (the page credits it; see `docs/DECISIONS.md` and the README).
//!
//! Only a closed set of paths is served, and only after login (like every other route but the page itself):
//! - a fixed list of sheets and one font (`game.png`, `emoticons.png`, `extras.png`, `hud.png`, `particles.png`,
//!   `arrow.png`, `editor/entities_clear/ddnet.png`, `editor/speed_arrow_array.png`, `fonts/DejaVuSans.ttf`);
//! - `mapres/<name>.png` (external tilesets and images a map names) and `skins/<name>.png` (stock skins), where
//!   `<name>` must be a plain file stem ([`is_safe_name`]). Both names come from untrusted data (a map's image names,
//!   a player's skin name), so nothing with a slash, a dot-dot, a control character or a leading dot is ever joined to
//!   the directory; the joined path is canonicalised and must still lie under the (canonicalised) data directory, so
//!   even a symlink planted inside it cannot lead out.
//!
//! No internet is involved: an unknown skin is a 404 and the page draws the `default` skin instead.

use std::path::{Path, PathBuf};

use axum::extract::{Path as UrlPath, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;

use crate::session_guard::current_session;
use crate::state::SharedState;

/// Largest file served (the biggest stock sheet is under 2 MiB).
const MAX_ASSET_BYTES: u64 = 16 * 1024 * 1024;

/// The fixed files, relative to the data directory.
const FIXED: &[&str] = &[
    "game.png",
    "emoticons.png",
    "extras.png",
    "hud.png",
    "particles.png",
    "arrow.png",
    "editor/entities_clear/ddnet.png",
    "editor/speed_arrow_array.png",
    "fonts/DejaVuSans.ttf",
];

/// Whether `name` is a plain file stem that may be looked up under `mapres/` or `skins/`: 1 to 64 characters of
/// ASCII letters, digits, `_`, `-`, `+`, `.` and inner spaces, starting with a letter, digit or `_`, containing no `..`.
/// (Stock names are like `grass_main`, `bg_cloud1`, `coala_bluekitty`, `hammie-chew`, `santa_default`.)
pub fn is_safe_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    if !(bytes[0].is_ascii_alphanumeric() || bytes[0] == b'_') {
        return false;
    }
    if name.contains("..") || name.ends_with('.') || name.ends_with(' ') {
        return false;
    }
    bytes
        .iter()
        .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'+' | b'.' | b' '))
}

/// The data-directory-relative path a request names, or `None` when it is not one of the served shapes.
pub fn resolve_request(path: &str) -> Option<String> {
    if FIXED.contains(&path) {
        return Some(path.to_string());
    }
    for dir in ["mapres", "skins"] {
        if let Some(rest) = path.strip_prefix(dir).and_then(|r| r.strip_prefix('/'))
            && let Some(stem) = rest.strip_suffix(".png")
            && is_safe_name(stem)
        {
            return Some(path.to_string());
        }
    }
    None
}

fn content_type(path: &str) -> &'static str {
    if path.ends_with(".ttf") {
        "font/ttf"
    } else {
        "image/png"
    }
}

/// Reads `rel` under `root`, refusing anything that resolves outside it (symlinks included) or is not a regular file.
fn read_under(root: &Path, rel: &str) -> Option<Vec<u8>> {
    let root = std::fs::canonicalize(root).ok()?;
    let full: PathBuf = std::fs::canonicalize(root.join(rel)).ok()?;
    if !full.starts_with(&root) {
        return None;
    }
    let meta = std::fs::metadata(&full).ok()?;
    if !meta.is_file() || meta.len() > MAX_ASSET_BYTES {
        return None;
    }
    std::fs::read(&full).ok()
}

pub async fn get_asset(State(state): State<SharedState>, UrlPath(path): UrlPath<String>, jar: CookieJar) -> Response {
    if current_session(&state, &jar).is_none() {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Some(root) = state.config.ddnet_data_dir.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(rel) = resolve_request(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let ctype = content_type(&rel);
    let body = tokio::task::spawn_blocking(move || read_under(&root, &rel)).await;
    match body {
        Ok(Some(bytes)) => (
            [
                (header::CONTENT_TYPE, ctype.to_string()),
                // Stock art does not change under a running server; `private` because the page is behind a login.
                (header::CACHE_CONTROL, "private, max-age=86400".to_string()),
            ],
            bytes,
        )
            .into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_names_are_safe() {
        for name in [
            "grass_main",
            "bg_cloud1",
            "coala_bluekitty",
            "hammie-chew",
            "santa_default",
            "PaladiN",
            "x_ninja",
            "generic_unhookable",
            "a b",
            "v1.2",
            "_x",
        ] {
            assert!(is_safe_name(name), "{name}");
        }
    }

    #[test]
    fn anything_that_could_leave_the_directory_or_confuse_a_path_is_refused() {
        for name in [
            "",
            "..",
            ".",
            ".hidden",
            "../skins/greyfox",
            "a/b",
            "a\\b",
            "a..b",
            "/etc/passwd",
            "name.",
            "name ",
            " name",
            "-lead",
            "a\0b",
            "a\nb",
            "a%2fb",
            "ünï",
            "a:b",
            &"x".repeat(65),
        ] {
            assert!(!is_safe_name(name), "{name:?}");
        }
    }

    #[test]
    fn only_the_served_shapes_resolve() {
        assert_eq!(resolve_request("game.png").as_deref(), Some("game.png"));
        assert_eq!(
            resolve_request("mapres/grass_main.png").as_deref(),
            Some("mapres/grass_main.png")
        );
        assert_eq!(
            resolve_request("skins/default.png").as_deref(),
            Some("skins/default.png")
        );
        assert_eq!(
            resolve_request("editor/entities_clear/ddnet.png").as_deref(),
            Some("editor/entities_clear/ddnet.png")
        );
        for bad in [
            "",
            "mapres/",
            "mapres/.png",
            "mapres/grass_main",
            "mapres/grass_main.PNG",
            "mapres/../game.png",
            "mapres/a/b.png",
            "skins/../../etc/passwd.png",
            "skins/..%2f.png",
            "audio/x.wv",
            "settings_ddnet.cfg",
            "editor/entities_clear/../../../etc/passwd",
            "maps/Copy Love Box.map",
            "/game.png",
            "fonts/../game.png",
            "mapres/grass_main.png/",
        ] {
            assert_eq!(resolve_request(bad), None, "{bad:?}");
        }
    }

    #[cfg(unix)] // making a symlink on Windows needs a privilege
    #[test]
    fn a_symlink_that_leads_out_of_the_directory_is_not_followed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("data");
        let outside = tmp.path().join("secret.png");
        std::fs::create_dir_all(root.join("skins")).unwrap();
        std::fs::write(&outside, b"secret").unwrap();
        std::fs::write(root.join("skins/default.png"), b"png").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("skins/evil.png")).unwrap();
        assert_eq!(read_under(&root, "skins/default.png").as_deref(), Some(&b"png"[..]));
        assert_eq!(read_under(&root, "skins/evil.png"), None);
        assert_eq!(read_under(&root, "skins/missing.png"), None);
        // A directory with a safe-looking name is not a file.
        std::fs::create_dir_all(root.join("skins/dir.png")).unwrap();
        assert_eq!(read_under(&root, "skins/dir.png"), None);
    }

    #[test]
    fn a_file_over_the_cap_is_not_served() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("mapres")).unwrap();
        let f = std::fs::File::create(tmp.path().join("mapres/huge.png")).unwrap();
        f.set_len(MAX_ASSET_BYTES + 1).unwrap();
        assert_eq!(read_under(tmp.path(), "mapres/huge.png"), None);
    }
}
