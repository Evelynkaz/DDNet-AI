//! `~/aiddnet/data/bot/settings.toml` (task 4.3): the few things the console commands change and the next
//! run should remember — the brain, the wayblock mode, where the lists live, `!low` and `!strong` — and (task 5.6)
//! the bot's clan and skin. Every
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
    /// The clan sent in `Cl_StartInfo` (default `Neuroset`, D-068). The nick is not here: see `identity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clan: Option<String>,
    /// A fixed skin; without it the bot picks a random stock skin at every start (D-068).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skin: Option<String>,
    /// `false` switches the owner's website chat off (task 4.9, D-094: the emergency switch; `--no-owner-chat` does the same for one run).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_chat: Option<bool>,
    /// The first words of the owner's website lines that start a duel (task 4.12, D-108): sending one is evidence that a two-player DDRace team
    /// is a duel. Case-insensitive; default `["/duel", "/1vs1"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duel_commands: Option<Vec<String>>,
    /// Task 3.17 (D-111): the learned window model (`--window-model <file>`; the flag wins). Without it the bot decides as before; the marker
    /// `bot/window-model.off` switches the model off while it exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_model: Option<PathBuf>,
    /// Task 3.16 (D-115): the hybrid's search budget in whole ms, 1 to 8 (`--hybrid-budget-ms`; the decision cap moves with it). Absent: 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid_budget_ms: Option<RawKnob>,
    /// Task 3.16 (D-115): a **fixed** prediction margin in ms, 0 to 30 (`--prediction-margin-ms`): the adaptive controller is off for the run.
    /// Absent: adaptive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction_margin_ms: Option<RawKnob>,
}

/// A number-valued key of the settings file that must never make the file unreadable (task 3.16, F1): a hand edit such as `-1`, `3.5` or `"5"` is
/// a perfectly parsable TOML value of the wrong kind, and a typed field would turn it into a serde error, which moves the whole file aside
/// (`.bad-<ts>`) and loses every other setting. Here any TOML value reads; [`crate::timing_knobs::resolve`] judges it (whole number in range, or ignored
/// with a warning).
#[derive(Debug, Clone)]
pub enum RawKnob {
    /// A TOML integer (any sign and size that fits an `i64`).
    Int(i64),
    /// Anything else (a float, a string, a table, ...), kept as the TOML value itself so that a save writes back exactly what was read (review F9:
    /// kept as text it gained a layer of quoting on every save).
    Other(toml::Value),
}

/// `toml::Value` has no `Eq` (floats); values compare by the text TOML shows for them.
impl PartialEq for RawKnob {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RawKnob::Int(a), RawKnob::Int(b)) => a == b,
            (RawKnob::Other(a), RawKnob::Other(b)) => a.to_string() == b.to_string(),
            _ => false,
        }
    }
}

impl Eq for RawKnob {}

impl<'de> Deserialize<'de> for RawKnob {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match toml::Value::deserialize(d)? {
            toml::Value::Integer(i) => RawKnob::Int(i),
            other => RawKnob::Other(other),
        })
    }
}

impl Serialize for RawKnob {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            RawKnob::Int(i) => s.serialize_i64(*i),
            RawKnob::Other(value) => value.serialize(s),
        }
    }
}

/// `~/aiddnet/data/bot/settings.toml`.
pub fn default_path() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("aiddnet/data/bot/settings.toml"),
        _ => PathBuf::from("data/bot/settings.toml"),
    }
}

/// The keys of [`Settings`], for [`unknown_keys`]. A test keeps this list equal to the struct's fields.
pub const KNOWN_KEYS: &[&str] = &[
    "brain",
    "wb",
    "relations",
    "low",
    "strong",
    "clan",
    "skin",
    "owner_chat",
    "duel_commands",
    "window_model",
    "hybrid_budget_ms",
    "prediction_margin_ms",
];

/// The top-level keys of the file at `path` that [`Settings`] does not know (a typo such as `owner-chat`), sorted. An unreadable or
/// unparsable file has none to report (the loader deals with it).
pub fn unknown_keys(path: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(table) = text.parse::<toml::Table>() else {
        return Vec::new();
    };
    let mut keys: Vec<String> = table
        .keys()
        .filter(|k| !KNOWN_KEYS.contains(&k.as_str()))
        .cloned()
        .collect();
    keys.sort();
    keys
}

/// The marker file that switches the owner chat off and survives `launch apply` and settings rewrites (`<data-dir>/bot/owner-chat.off`;
/// any entry of that name counts, also a symlink or an empty file).
pub const OWNER_CHAT_OFF_MARKER: &str = "owner-chat.off";

/// Why the owner chat is off for this run (task 4.9, D-094). The owner chat **fails closed**: anything that makes the switch's state
/// uncertain turns it off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatOff {
    /// `--no-owner-chat`.
    Flag,
    /// `owner_chat = false` in the settings file.
    Setting,
    /// The marker file `bot/owner-chat.off` exists.
    Marker,
    /// The settings file could not be read or parsed: the switch in it cannot be trusted.
    SettingsUnreadable,
    /// The settings file has keys the bot does not know (a typo of `owner_chat`, say).
    UnknownKeys(Vec<String>),
}

impl ChatOff {
    pub fn describe(&self) -> String {
        match self {
            ChatOff::Flag => "--no-owner-chat".to_string(),
            ChatOff::Setting => "owner_chat = false in the settings file".to_string(),
            ChatOff::Marker => format!("the marker file bot/{OWNER_CHAT_OFF_MARKER} exists"),
            ChatOff::SettingsUnreadable => "the settings file could not be read (fail closed)".to_string(),
            ChatOff::UnknownKeys(k) => {
                format!("the settings file has unknown keys {k:?}: a typo of owner_chat? (fail closed)")
            }
        }
    }
}

/// Whether the owner chat is off, and why. Pure: the caller reads the flag, the settings and the marker.
pub fn owner_chat_off(
    flag: bool,
    settings: &Settings,
    settings_unreadable: bool,
    unknown: &[String],
    marker_present: bool,
) -> Option<ChatOff> {
    if flag {
        Some(ChatOff::Flag)
    } else if settings.owner_chat == Some(false) {
        Some(ChatOff::Setting)
    } else if marker_present {
        Some(ChatOff::Marker)
    } else if settings_unreadable {
        Some(ChatOff::SettingsUnreadable)
    } else if !unknown.is_empty() {
        Some(ChatOff::UnknownKeys(unknown.to_vec()))
    } else {
        None
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
    fn the_owner_chat_fails_closed() {
        let on = Settings::default();
        let none: &[String] = &[];
        assert_eq!(
            owner_chat_off(false, &on, false, none, false),
            None,
            "nothing switches it off: it is on"
        );
        assert_eq!(
            owner_chat_off(
                false,
                &Settings {
                    owner_chat: Some(true),
                    ..Settings::default()
                },
                false,
                none,
                false
            ),
            None
        );
        assert_eq!(owner_chat_off(true, &on, false, none, false), Some(ChatOff::Flag));
        let off = Settings {
            owner_chat: Some(false),
            ..Settings::default()
        };
        assert_eq!(owner_chat_off(false, &off, false, none, false), Some(ChatOff::Setting));
        assert_eq!(owner_chat_off(false, &on, false, none, true), Some(ChatOff::Marker));
        assert_eq!(
            owner_chat_off(false, &on, true, none, false),
            Some(ChatOff::SettingsUnreadable)
        );
        let typo = vec!["owner-chat".to_string()];
        assert_eq!(
            owner_chat_off(false, &on, false, &typo, false),
            Some(ChatOff::UnknownKeys(typo.clone()))
        );
        assert!(ChatOff::UnknownKeys(typo).describe().contains("owner-chat"));
    }

    /// F7: a typo of the key and a string where a bool belongs both end with the chat off, not on.
    #[test]
    fn a_typo_or_a_corrupt_file_switches_the_chat_off_and_unknown_keys_are_listed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, "owner-chat = false\nclan = \"X\"\nbrian = 1\n").unwrap();
        let unknown = unknown_keys(&path);
        assert_eq!(unknown, vec!["brian".to_string(), "owner-chat".to_string()]);
        let Loaded::Ok(s) = load(&path) else {
            panic!("a file with unknown keys still loads")
        };
        assert_eq!(s.clan.as_deref(), Some("X"));
        assert!(matches!(
            owner_chat_off(false, &s, false, &unknown, false),
            Some(ChatOff::UnknownKeys(_))
        ));
        // a string where a bool belongs: the loader calls the file corrupt, and the verdict is "off"
        std::fs::write(&path, "owner_chat = \"false\"\n").unwrap();
        let loaded = load(&path);
        assert!(matches!(loaded, Loaded::Corrupt { .. }));
        assert_eq!(
            owner_chat_off(false, &loaded.settings(), true, &[], false),
            Some(ChatOff::SettingsUnreadable)
        );
        // a good file has no unknown keys, and a missing file has none either
        std::fs::write(&path, "owner_chat = true\nclan = \"X\"\n").unwrap();
        assert!(unknown_keys(&path).is_empty());
        assert!(unknown_keys(&dir.path().join("missing.toml")).is_empty());
    }

    #[test]
    fn the_known_keys_are_the_structs_fields() {
        let full = Settings {
            brain: Some("a".into()),
            wb: Some("a".into()),
            relations: Some(PathBuf::from("a")),
            low: Some(true),
            strong: Some(true),
            clan: Some("a".into()),
            skin: Some("a".into()),
            owner_chat: Some(true),
            duel_commands: Some(vec!["/duel".into()]),
            window_model: Some(PathBuf::from("a")),
            hybrid_budget_ms: Some(RawKnob::Int(4)),
            prediction_margin_ms: Some(RawKnob::Int(10)),
        };
        let table: toml::Table = toml::to_string(&full).unwrap().parse().unwrap();
        let mut keys: Vec<&str> = table.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut known = KNOWN_KEYS.to_vec();
        known.sort_unstable();
        assert_eq!(keys, known, "update KNOWN_KEYS with the struct");
    }

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
            clan: Some("Neuroset".into()),
            skin: Some("pinky".into()),
            owner_chat: Some(false),
            duel_commands: Some(vec!["/duel".into(), "/1vs1".into()]),
            window_model: Some(PathBuf::from("/x/m1.oppnet")),
            hybrid_budget_ms: Some(RawKnob::Int(3)),
            prediction_margin_ms: Some(RawKnob::Int(0)),
        };
        save(&path, &s).unwrap();
        assert_eq!(load(&path), Loaded::Ok(s));
        assert!(!dir.path().join("bot/settings.toml.tmp").exists(), "atomic");
    }

    /// Task 3.16, review F1: a value of the wrong kind in one of the two lag keys must not cost the owner the whole file.
    #[test]
    fn a_bad_looking_lag_key_never_moves_the_file_aside_or_costs_the_other_settings() {
        let dir = tempfile::tempdir().unwrap();
        for (i, (value, other)) in [
            ("-1", false),
            ("99", false),
            ("3.5", true),
            ("\"5\"", true),
            ("[1, 2]", true),
            ("true", true),
            ("{ a = 1 }", true),
            ("4", false),
        ]
        .into_iter()
        .enumerate()
        {
            for key in ["hybrid_budget_ms", "prediction_margin_ms"] {
                let path = dir.path().join(format!("s{i}-{key}.toml"));
                std::fs::write(&path, format!("brain = \"planner\"\nclan = \"X\"\n{key} = {value}\n")).unwrap();
                let Loaded::Ok(s) = load(&path) else {
                    panic!("{key} = {value} made the file corrupt");
                };
                assert_eq!(s.brain.as_deref(), Some("planner"), "{key} = {value}");
                assert_eq!(s.clan.as_deref(), Some("X"));
                assert!(path.exists(), "the file stays where it is");
                assert!(unknown_keys(&path).is_empty());
                let got = if key == "hybrid_budget_ms" {
                    &s.hybrid_budget_ms
                } else {
                    &s.prediction_margin_ms
                };
                assert_eq!(
                    matches!(got, Some(RawKnob::Other(_))),
                    other,
                    "{key} = {value}: {got:?}"
                );
            }
        }
        // And it round-trips through any number of saves exactly (the console commands save the whole struct; review F9: a value that was kept as text
        // gained a layer of quoting on every save).
        for (i, text) in ["4.5", "\"5\"", "[1, 2]", "true"].into_iter().enumerate() {
            let path = dir.path().join(format!("rt{i}.toml"));
            std::fs::write(&path, format!("prediction_margin_ms = {text}\nhybrid_budget_ms = 3\n")).unwrap();
            let Loaded::Ok(first) = load(&path) else {
                panic!("corrupt")
            };
            for n in 0..3 {
                update(&path, |s| s.low = Some(n % 2 == 0)).unwrap();
            }
            let Loaded::Ok(s) = load(&path) else {
                panic!("corrupt after a save")
            };
            assert_eq!(s.hybrid_budget_ms, Some(RawKnob::Int(3)));
            assert!(matches!(s.prediction_margin_ms, Some(RawKnob::Other(_))), "{text}");
            assert_eq!(
                s.prediction_margin_ms, first.prediction_margin_ms,
                "{text}: changed by three saves"
            );
            // Scalars come back as the same text (an array is laid out over several lines by the serializer; its value is compared above).
            if !text.starts_with('[') {
                let file = std::fs::read_to_string(&path).unwrap();
                let line = file.lines().find(|l| l.starts_with("prediction_margin_ms")).unwrap();
                assert_eq!(
                    line.split_once('=').unwrap().1.trim().replace(' ', ""),
                    text.replace(' ', ""),
                    "{text}: the file line"
                );
            }
        }
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
