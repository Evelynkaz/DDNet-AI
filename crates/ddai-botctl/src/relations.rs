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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ddai_os::private::OwnerOnly;
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
    /// Bumped on every change; `ddai-bot`'s player table recomputes its per-player flags when
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

impl ListKind {
    pub const ALL: [ListKind; 5] = [
        ListKind::Friend,
        ListKind::War,
        ListKind::Ignore,
        ListKind::ClanWar,
        ListKind::ClanFriend,
    ];

    /// The console / wire name (`friend`, `war`, `ignore`, `clanwar`, `clanfriend`).
    pub fn name(self) -> &'static str {
        match self {
            ListKind::Friend => "friend",
            ListKind::War => "war",
            ListKind::Ignore => "ignore",
            ListKind::ClanWar => "clanwar",
            ListKind::ClanFriend => "clanfriend",
        }
    }

    /// Exact lower-case wire name only.
    pub fn parse(s: &str) -> Option<ListKind> {
        ListKind::ALL.into_iter().find(|k| k.name() == s)
    }

    /// A clan list (entries are clan tags, not nicknames).
    pub fn is_clan(self) -> bool {
        matches!(self, ListKind::ClanWar | ListKind::ClanFriend)
    }
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
    /// Deliberately **no** `serde_json::Error` in here, not even as a source: its text quotes the offending value
    /// (`invalid type: string "SomeNick", expected a sequence`), and the lists hold nicknames. Only where the problem
    /// is (line, column) and what kind it is.
    #[error("{path} is not a valid relations file ({category} error at line {line}, column {column})")]
    Parse {
        path: PathBuf,
        category: &'static str,
        line: usize,
        column: usize,
    },
}

/// `<data dir>/bot/relations.json` (`~/aiddnet/data/bot/relations.json` on Linux, see `ddai_os::dirs`).
pub fn default_path() -> PathBuf {
    ddai_os::dirs::data_root_or_relative()
        .join("bot")
        .join("relations.json")
}

impl Relations {
    pub fn new() -> Self {
        Relations::default()
    }

    /// The change counter (see the field's doc comment).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Replaces every list with `other`'s (the web editor changed the file; the running bot reloads it) and
    /// bumps the change counter past both, so whoever caches per-player flags recomputes them.
    pub fn replace_with(&mut self, other: Relations) {
        let version = self.version.max(other.version) + 1;
        *self = other;
        self.version = version;
    }

    /// A short fingerprint (FNV-1a 64, 16 hex digits) of all five lists, for comparing what the web wrote with what the
    /// bot loaded without sending a name anywhere. Not a security hash: it only has to tell two lists apart.
    pub fn digest(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for kind in ListKind::ALL {
            eat(kind.name().as_bytes());
            eat(&[0x1e]);
            for entry in self.set(kind) {
                eat(entry.as_bytes());
                eat(&[0x1f]);
            }
            eat(&[0x1d]);
        }
        format!("{h:016x}")
    }

    /// Whether `other` holds exactly the same entries on every list (the counter is not compared).
    pub fn same_lists(&self, other: &Relations) -> bool {
        self.friend == other.friend
            && self.war == other.war
            && self.ignore == other.ignore
            && self.clan_war == other.clan_war
            && self.clan_friend == other.clan_friend
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

    /// The lists a name leaves when it is put on `kind` (`!war` removes it from friend and ignore; ...).
    pub fn exclusions(kind: ListKind) -> &'static [ListKind] {
        match kind {
            ListKind::War => &[ListKind::Friend, ListKind::Ignore],
            ListKind::Friend | ListKind::Ignore => &[ListKind::War],
            ListKind::ClanWar => &[ListKind::ClanFriend],
            ListKind::ClanFriend => &[ListKind::ClanWar],
        }
    }

    /// The entries of `kind` (folded keys), in order.
    pub fn names(&self, kind: ListKind) -> Vec<&str> {
        self.set(kind).iter().map(String::as_str).collect()
    }

    /// Adds `raw` to `kind` with the old bot's exclusions (`!war` removes the name from friend and
    /// ignore; `!friend` and `!ignore` remove it from war; clan war and clan friend exclude each
    /// other — `orig-bot.md` §7.3). Returns false when the name folds to nothing.
    pub fn add(&mut self, kind: ListKind, raw: &str) -> bool {
        let key = fold_name(raw);
        if key.is_empty() {
            return false;
        }
        for &other in Relations::exclusions(kind) {
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
            Ok(text) => Relations::from_json(&text).map_err(|e| RelationsError::Parse {
                path: path.to_path_buf(),
                category: match e.classify() {
                    serde_json::error::Category::Io => "io",
                    serde_json::error::Category::Syntax => "syntax",
                    serde_json::error::Category::Data => "data",
                    serde_json::error::Category::Eof => "eof",
                },
                line: e.line(),
                column: e.column(),
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Relations::new()),
            Err(source) => Err(RelationsError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    /// Writes the lists (folded keys — the original display spelling is not kept) atomically
    /// (own `.tmp` + fsync + rename, mode 0600), creating the directory. Two processes that read-modify-write the file
    /// must hold [`RelationsLock`] around the whole sequence: this only keeps one write from tearing.
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
        // A tmp name of its own (pid + counter) so two writers never share a file, fsync'd before the rename so a
        // crash leaves the old file or the new one, never a torn one (task 5.6 review F6).
        static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(format!(
            ".tmp.{}.{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let tmp = PathBuf::from(tmp_name);
        // The file holds real nicknames: owner-only (0600), like the sockets and the secrets.
        let written = (|| -> io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .owner_only()
                .open(&tmp)?;
            // Windows: the ACL goes on before the nicknames are written (Unix: the mode was given at creation).
            ddai_os::private::restrict_file(&tmp)?;
            f.write_all(&serde_json::to_vec_pretty(&file).map_err(io::Error::other)?)?;
            f.sync_all()
        })();
        if let Err(e) = written.and_then(|()| std::fs::rename(&tmp, path)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }
}

/// An advisory lock (`flock`) on `<file>.lock` that serialises read-modify-write of the lists file **across processes**
/// (the bot's console edits and the web editor; task 5.6 review F6). Held for the lifetime of the value; released on
/// drop (or when the process dies). Waiting is bounded: the bot must never stall its decision loop on a stuck holder.
pub struct RelationsLock {
    _file: std::fs::File,
}

/// How long [`RelationsLock::acquire`] waits for another writer.
pub const LOCK_WAIT: Duration = Duration::from_secs(2);

impl RelationsLock {
    /// Takes the lock for the lists file at `path`, creating the directory and the `.lock` file (mode 0600) as needed.
    pub fn acquire(path: &Path) -> io::Result<RelationsLock> {
        RelationsLock::acquire_within(path, LOCK_WAIT)
    }

    pub fn acquire_within(path: &Path, wait: Duration) -> io::Result<RelationsLock> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut lock_name = path.as_os_str().to_owned();
        lock_name.push(".lock");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .owner_only()
            .open(PathBuf::from(lock_name))?;
        let started = Instant::now();
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(RelationsLock { _file: file }),
                Err(std::fs::TryLockError::WouldBlock) => {
                    if started.elapsed() >= wait {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "the lists file is locked by another writer",
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(std::fs::TryLockError::Error(e)) => return Err(e),
            }
        }
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

    #[test]
    fn replace_with_takes_the_other_lists_and_bumps_the_counter_past_both() {
        let mut running = Relations::new();
        running.add(ListKind::Friend, "old");
        running.add(ListKind::Friend, "old2");
        let mut file = Relations::new();
        file.add(ListKind::War, "new");
        let (v_running, v_file) = (running.version(), file.version());
        assert!(!running.same_lists(&file));
        running.replace_with(file.clone());
        assert!(running.same_lists(&file));
        assert!(running.contains(ListKind::War, "new") && !running.contains(ListKind::Friend, "old"));
        assert!(running.version() > v_running.max(v_file));
    }

    #[test]
    fn list_kinds_have_one_wire_name_each_and_parse_back_exactly() {
        for k in ListKind::ALL {
            assert_eq!(ListKind::parse(k.name()), Some(k));
        }
        assert_eq!(ListKind::parse("Friend"), None, "exact, lower-case only");
        assert_eq!(ListKind::parse("clanFriend"), None);
        assert_eq!(ListKind::parse(""), None);
        assert!(ListKind::ClanWar.is_clan() && ListKind::ClanFriend.is_clan() && !ListKind::Friend.is_clan());
    }

    #[test]
    fn the_saved_file_is_private() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relations.json");
        let mut r = Relations::new();
        r.add(ListKind::Friend, "x");
        r.save(&path).unwrap();
        assert!(
            ddai_os::private::is_restricted(&path).unwrap(),
            "owner only (Unix 0600, Windows ACL)"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        r.save(&path).unwrap(); // and again over the existing file
        assert!(ddai_os::private::is_restricted(&path).unwrap());
    }

    #[test]
    fn a_parse_error_never_quotes_a_nickname() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relations.json");
        for text in [
            r#"{"friend":"SecretPal"}"#,
            r#"{"friend":["SecretPal",3]}"#,
            r#"{"friend":["SecretPal"] "war":[]}"#,
            r#"{"friend":["SecretPal"]"#,
            r#"{"v":"SecretPal"}"#,
        ] {
            std::fs::write(&path, text).unwrap();
            let e = Relations::load(&path).unwrap_err();
            let shown = format!("{e} / {e:?}");
            assert!(!shown.contains("SecretPal"), "{shown}");
            assert!(shown.contains("line 1"), "where the problem is is still said: {shown}");
            assert!(
                std::error::Error::source(&e).is_none(),
                "no serde error behind it either"
            );
        }
    }

    #[test]
    fn the_digest_tells_lists_apart_and_ignores_the_change_counter() {
        let mut a = Relations::new();
        a.add(ListKind::Friend, "x");
        let mut b = Relations::new();
        b.add(ListKind::Friend, "x");
        b.add(ListKind::War, "y");
        b.remove(ListKind::War, "y");
        assert_ne!(a.version(), b.version());
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.digest().len(), 16);
        let mut c = Relations::new();
        c.add(ListKind::War, "x");
        assert_ne!(a.digest(), c.digest(), "the same name on another list differs");
        assert_ne!(Relations::new().digest(), a.digest());
        let mut d = Relations::new();
        d.add(ListKind::Friend, "xy");
        let mut e = Relations::new();
        e.add(ListKind::Friend, "x");
        e.add(ListKind::Friend, "y");
        assert_ne!(d.digest(), e.digest(), "entry boundaries count");
    }

    /// The review's reproduction (F6): two writers, different lists, hundreds of rounds. Without a lock and a tmp file
    /// of their own they tore the file and failed renames.
    #[test]
    fn two_writers_with_the_lock_never_corrupt_the_file_or_lose_an_edit() {
        let dir = tempfile::tempdir().unwrap();
        let path = std::sync::Arc::new(dir.path().join("bot").join("relations.json"));
        let rounds = 300;
        let writers: Vec<_> = [ListKind::Friend, ListKind::War]
            .into_iter()
            .map(|kind| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for i in 0..rounds {
                        // A long wait: this test is about "no torn file, no lost edit", not about lock latency (a slow CI runner
                        // under 2 x 300 contended rounds can pass the production 2 s bound).
                        let _lock = RelationsLock::acquire_within(&path, Duration::from_secs(120)).expect("lock");
                        let mut r = Relations::load(&path).expect("always parses");
                        assert!(r.add(kind, &format!("{}{i}", kind.name())));
                        r.save(&path).expect("save");
                    }
                })
            })
            .collect();
        let reader = {
            let path = path.clone();
            std::thread::spawn(move || {
                // A reader takes no lock (the bot's reload does not): the rename keeps every read whole.
                for _ in 0..2000 {
                    if path.exists() {
                        Relations::load(&path).expect("a reader never sees a torn file");
                    }
                }
            })
        };
        for w in writers {
            w.join().unwrap();
        }
        reader.join().unwrap();
        let r = Relations::load(&path).unwrap();
        assert_eq!(r.len(ListKind::Friend), rounds, "no lost update");
        assert_eq!(r.len(ListKind::War), rounds);
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn saves_without_the_lock_still_never_tear_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = std::sync::Arc::new(dir.path().join("relations.json"));
        let writers: Vec<_> = (0..2)
            .map(|w| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for i in 0..300 {
                        let mut r = Relations::new();
                        r.add(ListKind::Friend, &format!("w{w}-{i}"));
                        r.save(&path).expect("own tmp name: no collision");
                        Relations::load(&path).expect("whole file");
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
    }

    #[test]
    fn the_lock_excludes_a_second_holder_and_the_wait_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relations.json");
        let first = RelationsLock::acquire(&path).unwrap();
        let started = Instant::now();
        let err = RelationsLock::acquire_within(&path, Duration::from_millis(100))
            .err()
            .unwrap();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the wait is bounded, not forever"
        );
        drop(first);
        assert!(
            RelationsLock::acquire_within(&path, Duration::from_millis(100)).is_ok(),
            "released on drop"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join("relations.json.lock"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }
}
