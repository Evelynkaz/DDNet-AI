//! The friend / war / ignore / clan-war / clan-friend lists (`docs/research/orig-bot.md` §7.3),
//! stored in `~/aiddnet/data/bot/relations.json` — **never in git** (they hold real nicknames). The
//! file keeps the old bot's shape, `{"v":..,"war":[..],"friend":[..],"clanWar":[..],"clanFriend":[..],
//! "ignore":[..]}`, with display names; matching is **exact on the folded form**
//! ([`crate::names::fold_name`]) — the TS substring match is a bug (D-021, §13.5).
//!
//! Semantics in the fight (§7.3): friend / ignore / clanFriend are never targets and are "spared" by
//! the hook veto and the planner; war is `+900` in the target score and ignores AFK; a friend is not
//! "rescued" (dropped, D-021/the task spec). The list commands (`!friend` ...) are task 4.3; this
//! module keeps only what the fight needs plus `load`/`save` and the mutual-exclusion rules so 4.3
//! has nothing to redo.

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::names::fold_name;

/// The five lists, by display name in the file and by folded key in memory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Relations {
    friend: BTreeSet<String>,
    war: BTreeSet<String>,
    ignore: BTreeSet<String>,
    clan_war: BTreeSet<String>,
    clan_friend: BTreeSet<String>,
    /// Bumped on every change; [`crate::players::PlayerTable`] recomputes its per-player flags when
    /// it differs from the one it last saw.
    version: u64,
}

/// Which list a name goes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListKind {
    Friend,
    War,
    Ignore,
    ClanWar,
    ClanFriend,
}

/// What a player is on the lists, computed once per name change (see `PlayerTable`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RelationFlags {
    pub friend: bool,
    pub war: bool,
    pub ignore: bool,
    pub clan_war: bool,
    pub clan_friend: bool,
}

impl RelationFlags {
    /// Never a target: friend, ignore or clan-friend (`pickTarget`, `bot.ts:3065-3067`).
    pub fn never_target(self) -> bool {
        self.friend || self.ignore || self.clan_friend
    }

    /// At war: listed by name or by clan (`bot.ts:3068`).
    pub fn at_war(self) -> bool {
        self.war || self.clan_war
    }

    /// `isFriendId` (`bot.ts:2822-2826`): friend by name or by clan.
    pub fn friendly(self) -> bool {
        self.friend || self.clan_friend
    }

    /// Helpers that make the frozen bot wait instead of killing itself (`helperNear`,
    /// `bot.ts:4418-4432`): friend, ignore or clan-friend.
    pub fn helper(self) -> bool {
        self.never_target()
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct RelationsFile {
    #[serde(default)]
    v: u32,
    #[serde(default)]
    war: Vec<String>,
    #[serde(default)]
    friend: Vec<String>,
    #[serde(default, rename = "clanWar")]
    clan_war: Vec<String>,
    #[serde(default, rename = "clanFriend")]
    clan_friend: Vec<String>,
    #[serde(default)]
    ignore: Vec<String>,
}

/// Why loading the relations file failed.
#[derive(Debug, thiserror::Error)]
pub enum RelationsError {
    #[error("reading {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path} is not a valid relations file: {source}")]
    Parse { path: PathBuf, source: serde_json::Error },
}

/// `~/aiddnet/data/bot/relations.json`.
pub fn default_path() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home)
            .join("aiddnet")
            .join("data")
            .join("bot")
            .join("relations.json"),
        _ => PathBuf::from("data").join("bot").join("relations.json"),
    }
}

impl Relations {
    pub fn new() -> Self {
        Relations::default()
    }

    /// The change counter (see the field's doc comment).
    pub fn version(&self) -> u64 {
        self.version
    }

    fn set(&self, kind: ListKind) -> &BTreeSet<String> {
        match kind {
            ListKind::Friend => &self.friend,
            ListKind::War => &self.war,
            ListKind::Ignore => &self.ignore,
            ListKind::ClanWar => &self.clan_war,
            ListKind::ClanFriend => &self.clan_friend,
        }
    }

    fn set_mut(&mut self, kind: ListKind) -> &mut BTreeSet<String> {
        match kind {
            ListKind::Friend => &mut self.friend,
            ListKind::War => &mut self.war,
            ListKind::Ignore => &mut self.ignore,
            ListKind::ClanWar => &mut self.clan_war,
            ListKind::ClanFriend => &mut self.clan_friend,
        }
    }

    /// Number of entries on `kind`.
    pub fn len(&self, kind: ListKind) -> usize {
        self.set(kind).len()
    }

    /// No entry on any list.
    pub fn is_empty(&self) -> bool {
        [
            ListKind::Friend,
            ListKind::War,
            ListKind::Ignore,
            ListKind::ClanWar,
            ListKind::ClanFriend,
        ]
        .iter()
        .all(|&k| self.set(k).is_empty())
    }

    /// Whether `folded` (an already folded name/clan key) is on `kind`. **Exact** match.
    pub fn contains_key(&self, kind: ListKind, folded: &str) -> bool {
        !folded.is_empty() && self.set(kind).contains(folded)
    }

    /// Whether `raw` (any spelling) is on `kind` after folding.
    pub fn contains(&self, kind: ListKind, raw: &str) -> bool {
        self.contains_key(kind, &fold_name(raw))
    }

    /// The flags of a player with these folded name/clan keys.
    pub fn flags_for(&self, name_key: &str, clan_key: &str) -> RelationFlags {
        RelationFlags {
            friend: self.contains_key(ListKind::Friend, name_key),
            war: self.contains_key(ListKind::War, name_key),
            ignore: self.contains_key(ListKind::Ignore, name_key),
            clan_war: self.contains_key(ListKind::ClanWar, clan_key),
            clan_friend: self.contains_key(ListKind::ClanFriend, clan_key),
        }
    }

    /// Adds `raw` to `kind` with the old bot's exclusions (`!war` removes the name from friend and
    /// ignore; `!friend` and `!ignore` remove it from war; clan war and clan friend exclude each
    /// other — `orig-bot.md` §7.3). Returns false when the name folds to nothing.
    pub fn add(&mut self, kind: ListKind, raw: &str) -> bool {
        let key = fold_name(raw);
        if key.is_empty() {
            return false;
        }
        let excluded: &[ListKind] = match kind {
            ListKind::War => &[ListKind::Friend, ListKind::Ignore],
            ListKind::Friend | ListKind::Ignore => &[ListKind::War],
            ListKind::ClanWar => &[ListKind::ClanFriend],
            ListKind::ClanFriend => &[ListKind::ClanWar],
        };
        for &other in excluded {
            self.set_mut(other).remove(&key);
        }
        self.set_mut(kind).insert(key);
        self.version += 1;
        true
    }

    /// Removes `raw` from `kind`; true if it was there.
    pub fn remove(&mut self, kind: ListKind, raw: &str) -> bool {
        let removed = self.set_mut(kind).remove(&fold_name(raw));
        if removed {
            self.version += 1;
        }
        removed
    }

    /// Empties `kind`.
    pub fn clear(&mut self, kind: ListKind) {
        self.set_mut(kind).clear();
        self.version += 1;
    }

    /// Parses the file's JSON. Entries are folded; empty ones dropped.
    pub fn from_json(text: &str) -> Result<Relations, serde_json::Error> {
        let file: RelationsFile = serde_json::from_str(text)?;
        let mut r = Relations::new();
        let fold_all = |names: &[String]| -> BTreeSet<String> {
            names.iter().map(|n| fold_name(n)).filter(|k| !k.is_empty()).collect()
        };
        r.friend = fold_all(&file.friend);
        r.war = fold_all(&file.war);
        r.ignore = fold_all(&file.ignore);
        r.clan_war = fold_all(&file.clan_war);
        r.clan_friend = fold_all(&file.clan_friend);
        Ok(r)
    }

    /// Loads `path`; a missing file is an empty set of lists (first run), any other problem is an
    /// error the caller must see (a corrupt file silently read as "no friends" would make the bot
    /// attack a friend).
    pub fn load(path: &Path) -> Result<Relations, RelationsError> {
        match std::fs::read_to_string(path) {
            Ok(text) => Relations::from_json(&text).map_err(|source| RelationsError::Parse {
                path: path.to_path_buf(),
                source,
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Relations::new()),
            Err(source) => Err(RelationsError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    /// Writes the lists (folded keys — the original display spelling is not kept) atomically
    /// (`.tmp` + rename), creating the directory.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = RelationsFile {
            v: 2,
            war: self.war.iter().cloned().collect(),
            friend: self.friend.iter().cloned().collect(),
            clan_war: self.clan_war.iter().cloned().collect(),
            clan_friend: self.clan_friend.iter().cloned().collect(),
            ignore: self.ignore.iter().cloned().collect(),
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&file).map_err(io::Error::other)?)?;
        std::fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_is_exact_after_folding_not_a_substring() {
        let mut r = Relations::new();
        assert!(r.add(ListKind::Friend, "Bob"));
        // The TS bug: `nameKey.includes(key)` made "bob" match "bobby" and "xbobx". Not any more.
        assert!(r.contains(ListKind::Friend, "bob"));
        assert!(r.contains(ListKind::Friend, "  BOB "));
        assert!(!r.contains(ListKind::Friend, "bobby"));
        assert!(!r.contains(ListKind::Friend, "xbobx"));
        assert!(!r.contains(ListKind::Friend, "bo"));
        assert!(!r.flags_for("bobby", "").friend);
    }

    #[test]
    fn the_duplicate_prefix_is_ignored_on_both_sides() {
        let mut r = Relations::new();
        r.add(ListKind::War, "(1)Rival");
        assert!(r.flags_for("rival", "").war, "a listed '(1)x' matches plain 'x'");
        let mut r2 = Relations::new();
        r2.add(ListKind::War, "Rival");
        assert!(
            r2.contains(ListKind::War, "(2)rival"),
            "and the server's renamed copy matches the list"
        );
        assert!(r2.flags_for(&fold_name("(2)Rival"), "").war);
    }

    #[test]
    fn clan_lists_are_exact_too_and_independent_of_names() {
        let mut r = Relations::new();
        r.add(ListKind::ClanFriend, "TeamX");
        let f = r.flags_for("someone", "teamx");
        assert!(f.clan_friend && f.never_target() && f.friendly() && f.helper());
        assert!(!r.flags_for("someone", "teamxy").clan_friend);
        assert!(!r.flags_for("teamx", "other").clan_friend, "a name is not a clan");
    }

    #[test]
    fn empty_names_match_nothing_and_cannot_be_added() {
        let mut r = Relations::new();
        assert!(!r.add(ListKind::Friend, "   "));
        assert!(!r.add(ListKind::Friend, "(3)"));
        assert!(!r.flags_for("", "").friend);
        assert!(r.is_empty());
    }

    #[test]
    fn list_exclusions_follow_the_old_bot() {
        let mut r = Relations::new();
        r.add(ListKind::Friend, "a");
        r.add(ListKind::Ignore, "a");
        assert!(r.contains(ListKind::Friend, "a") && r.contains(ListKind::Ignore, "a"));
        r.add(ListKind::War, "a");
        assert!(r.contains(ListKind::War, "a"));
        assert!(
            !r.contains(ListKind::Friend, "a") && !r.contains(ListKind::Ignore, "a"),
            "war removes both"
        );
        r.add(ListKind::Friend, "a");
        assert!(!r.contains(ListKind::War, "a"), "friend removes war");
        r.add(ListKind::ClanWar, "c");
        r.add(ListKind::ClanFriend, "c");
        assert!(!r.contains(ListKind::ClanWar, "c") && r.contains(ListKind::ClanFriend, "c"));
    }

    #[test]
    fn version_changes_only_on_real_changes() {
        let mut r = Relations::new();
        let v0 = r.version();
        r.add(ListKind::Friend, "x");
        assert!(r.version() > v0);
        let v1 = r.version();
        assert!(!r.remove(ListKind::Friend, "nobody"));
        assert_eq!(r.version(), v1);
        assert!(r.remove(ListKind::Friend, "X"));
        assert!(r.version() > v1);
    }

    #[test]
    fn the_old_bots_file_shape_loads() {
        let r = Relations::from_json(
            r#"{"v":1,"war":["Enemy One"],"friend":["Pal"],"clanWar":["Bad Clan"],"clanFriend":["Good"],"ignore":["Noise","(1)Noise2"]}"#,
        )
        .unwrap();
        assert!(r.contains(ListKind::War, "enemy   one"));
        assert!(r.contains(ListKind::Friend, "pal"));
        assert!(r.contains(ListKind::ClanWar, "bad clan"));
        assert!(r.contains(ListKind::ClanFriend, "good"));
        assert!(r.contains(ListKind::Ignore, "noise2"));
        assert!(Relations::from_json("{}").unwrap().is_empty());
    }

    #[test]
    fn save_and_load_round_trip_and_a_missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bot").join("relations.json");
        assert!(Relations::load(&path).unwrap().is_empty(), "first run");
        let mut r = Relations::new();
        r.add(ListKind::Friend, "Pal");
        r.add(ListKind::ClanWar, "Clan X");
        r.save(&path).unwrap();
        let back = Relations::load(&path).unwrap();
        assert!(back.contains(ListKind::Friend, "pal") && back.contains(ListKind::ClanWar, "clan x"));
        std::fs::write(&path, "{ not json").unwrap();
        assert!(
            matches!(Relations::load(&path), Err(RelationsError::Parse { .. })),
            "corrupt is an error"
        );
    }
}
