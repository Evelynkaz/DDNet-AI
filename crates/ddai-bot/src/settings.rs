//! `~/aiddnet/data/bot/settings.toml` (task 4.3): the few things the console commands change and the next
//! run should remember — the brain, the wayblock mode, where the lists live, `!low` and `!strong`. Every
//! field is optional: a missing file or key means "use the command line's / the built-in default".
//! Nothing here is a secret and nothing a nickname, but it is **never in git** (CLAUDE.md).
//!
//! The command line wins over the file. A file that does not parse is renamed to `<name>.bad-<unix
//! seconds>` and the run goes on with the defaults (the TS `start.mjs:242-246` did the same with
//! `settings.json`), so one bad edit never keeps the bot from starting nor is silently overwritten.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The settings file's contents.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// `hybrid | planner | scripted | idle | fly`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brain: Option<String>,
    /// `off | left | right | auto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wb: Option<String>,
    /// The friend / war lists file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relations: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strong: Option<bool>,
}

/// `~/aiddnet/data/bot/settings.toml`.
pub fn default_path() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("aiddnet/data/bot/settings.toml"),
        _ => PathBuf::from("data/bot/settings.toml"),
    }
}

/// What loading found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loaded {
    /// The file's settings (or the defaults when there is no file).
    Ok(Settings),
    /// The file did not parse and was moved to this path; the defaults apply.
    Corrupt { moved_to: Option<PathBuf>, why: String },
}

impl Loaded {
    pub fn settings(self) -> Settings {
        match self {
            Loaded::Ok(s) => s,
            Loaded::Corrupt { .. } => Settings::default(),
        }
    }
}

/// Reads `path`. A missing file is the defaults; a corrupt one is renamed aside.
pub fn load(path: &Path) -> Loaded {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Loaded::Ok(Settings::default()),
        Err(e) => {
            return Loaded::Corrupt {
                moved_to: None,
                why: format!("cannot read {}: {e}", path.display()),
            };
        }
    };
    match toml::from_str::<Settings>(&text) {
        Ok(s) => Loaded::Ok(s),
        Err(e) => {
            let ts = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let mut name = path.as_os_str().to_owned();
            name.push(format!(".bad-{ts}"));
            let bad = PathBuf::from(name);
            let moved_to = std::fs::rename(path, &bad).ok().map(|()| bad);
            Loaded::Corrupt {
                moved_to,
                why: e.to_string(),
            }
        }
    }
}

/// Writes `settings` atomically (`.tmp` then rename), creating the directory.
pub fn save(path: &Path, settings: &Settings) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let text = toml::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

/// Loads the file (corrupt -> defaults), applies `change`, writes it back.
pub fn update(path: &Path, change: impl FnOnce(&mut Settings)) -> std::io::Result<()> {
    let mut s = load(path).settings();
    change(&mut s);
    save(path, &s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_the_defaults_and_a_saved_one_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bot/settings.toml");
        assert_eq!(load(&path), Loaded::Ok(Settings::default()));
        let s = Settings {
            brain: Some("planner".into()),
            wb: Some("left".into()),
            relations: Some(PathBuf::from("/x/relations.json")),
            low: Some(true),
            strong: Some(false),
        };
        save(&path, &s).unwrap();
        assert_eq!(load(&path), Loaded::Ok(s));
        assert!(!dir.path().join("bot/settings.toml.tmp").exists(), "atomic");
    }

    #[test]
    fn a_corrupt_file_is_renamed_aside_and_the_defaults_apply() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, "brain = [oops").unwrap();
        let Loaded::Corrupt { moved_to, why } = load(&path) else {
            panic!("a corrupt file must be reported");
        };
        assert!(!why.is_empty());
        let bad = moved_to.expect("renamed");
        assert!(
            bad.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("settings.toml.bad-")
        );
        assert!(bad.exists() && !path.exists(), "the bad file is kept, not deleted");
        assert_eq!(
            load(&path),
            Loaded::Ok(Settings::default()),
            "and the next run starts clean"
        );
    }

    #[test]
    fn update_changes_one_key_and_keeps_the_others() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        update(&path, |s| s.brain = Some("hybrid".into())).unwrap();
        update(&path, |s| s.low = Some(true)).unwrap();
        let s = load(&path).settings();
        assert_eq!((s.brain.as_deref(), s.low), (Some("hybrid"), Some(true)));
    }
}
