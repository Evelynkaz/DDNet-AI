//! Where things live.
//!
//! The root of everything the bot keeps on disk (the *data directory*: maps, settings, secrets, runs, logs):
//!
//! 1. `DDNET_AI_DATA_DIR`, if set and not empty (every platform);
//! 2. Linux and other Unix: `$HOME/aiddnet/data` (what the VPS has always used);
//! 3. Windows: `%USERPROFILE%\ddnet-ai\data`.
//!
//! Commands that take `--data-dir` use it instead; [`data_root`] is the default of those that do not.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The environment variable that overrides the default data directory.
pub const DATA_DIR_ENV: &str = "DDNET_AI_DATA_DIR";

fn non_empty(v: Option<OsString>) -> Option<OsString> {
    v.filter(|v| !v.is_empty())
}

/// The user's home directory: `$HOME` on Unix; `%USERPROFILE%` on Windows (then `$HOME`, which Git Bash / MSYS sessions set).
#[must_use]
pub fn home_dir() -> Option<PathBuf> {
    home_from(std::env::var_os("HOME"), std::env::var_os("USERPROFILE"))
}

fn home_from(home: Option<OsString>, userprofile: Option<OsString>) -> Option<PathBuf> {
    let (first, second) = if cfg!(windows) {
        (userprofile, home)
    } else {
        (home, None)
    };
    non_empty(first).or_else(|| non_empty(second)).map(PathBuf::from)
}

/// The default data directory (see the module docs), or `None` when neither the override nor a home directory is known.
#[must_use]
pub fn data_root() -> Option<PathBuf> {
    data_root_from(std::env::var_os(DATA_DIR_ENV), home_dir())
}

fn data_root_from(over: Option<OsString>, home: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(dir) = non_empty(over) {
        return Some(PathBuf::from(dir));
    }
    let home = home?;
    Some(if cfg!(windows) {
        home.join("ddnet-ai").join("data")
    } else {
        home.join("aiddnet").join("data")
    })
}

/// [`data_root`], or the relative directory `data` when there is none (the fallback the per-crate helpers always had).
#[must_use]
pub fn data_root_or_relative() -> PathBuf {
    data_root().unwrap_or_else(|| PathBuf::from("data"))
}

/// Expands a leading `~` (`~`, `~/rest`, and on Windows also `~\rest`) to the home directory. Anything else, and `~user`, is returned
/// as is. `None` when the path starts with `~` but the home directory is unknown.
#[must_use]
pub fn expand_tilde(path: &str) -> Option<PathBuf> {
    expand_tilde_with(path, home_dir().as_deref())
}

fn expand_tilde_with(path: &str, home: Option<&Path>) -> Option<PathBuf> {
    if path == "~" {
        return home.map(Path::to_path_buf);
    }
    let rest = path
        .strip_prefix("~/")
        .or_else(|| if cfg!(windows) { path.strip_prefix("~\\") } else { None });
    match rest {
        Some(rest) => home.map(|h| h.join(rest)),
        None => Some(PathBuf::from(path)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(s: &str) -> Option<OsString> {
        Some(OsString::from(s))
    }

    #[test]
    fn the_override_wins_and_empty_means_unset() {
        assert_eq!(
            data_root_from(os("/srv/bot"), Some(PathBuf::from("/home/u"))),
            Some(PathBuf::from("/srv/bot"))
        );
        let default = data_root_from(os(""), Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(default, data_root_from(None, Some(PathBuf::from("/home/u"))).unwrap());
        assert_eq!(data_root_from(None, None), None);
    }

    #[test]
    fn the_platform_default_is_what_the_deployment_has_always_used() {
        let d = data_root_from(None, Some(PathBuf::from("/home/u"))).unwrap();
        if cfg!(windows) {
            assert_eq!(d, PathBuf::from("/home/u").join("ddnet-ai").join("data"));
        } else {
            assert_eq!(d, PathBuf::from("/home/u/aiddnet/data"));
        }
    }

    #[test]
    fn home_comes_from_the_platforms_variable() {
        let h = home_from(os("/h"), os("C:\\Users\\u"));
        if cfg!(windows) {
            assert_eq!(h, Some(PathBuf::from("C:\\Users\\u")));
            assert_eq!(
                home_from(os("/h"), os("")),
                Some(PathBuf::from("/h")),
                "falls back to HOME"
            );
        } else {
            assert_eq!(h, Some(PathBuf::from("/h")));
        }
        assert_eq!(home_from(os(""), None), None);
    }

    #[test]
    fn tilde_expansion() {
        let home = PathBuf::from("/home/u");
        assert_eq!(expand_tilde_with("~/a/b", Some(&home)), Some(home.join("a/b")));
        assert_eq!(expand_tilde_with("~", Some(&home)), Some(home.clone()));
        assert_eq!(expand_tilde_with("/abs", Some(&home)), Some(PathBuf::from("/abs")));
        assert_eq!(
            expand_tilde_with("~user/x", Some(&home)),
            Some(PathBuf::from("~user/x"))
        );
        assert_eq!(expand_tilde_with("~/a", None), None);
        assert_eq!(expand_tilde_with("rel/a", None), Some(PathBuf::from("rel/a")));
        if cfg!(windows) {
            assert_eq!(expand_tilde_with("~\\a", Some(&home)), Some(home.join("a")));
        }
    }
}
