//! The friend / war / ignore / clan-war / clan-friend lists editor's storage (task 5.6): reads and writes the lists
//! file (`<data-dir>/bot/relations.json`, mode `0600`, never in git) with the very code the bot loads it with
//! (`ddai_botctl::relations`). So **matching is exact after normalisation** (D-021, `ddai_botctl::names::fold_name`):
//! what the owner types is folded (trimmed, whitespace collapsed, lower-cased, a leading `(<digits>)` duplicate-name
//! prefix removed) and the folded form is what is stored, shown and matched. There is no substring matching anywhere.
//!
//! **Names are only for the owner.** They are returned to the logged-in owner's browser and written to that one file.
//! Nothing here logs one: errors name the file, never an entry.
//!
//! **One writer at a time, and never over a file that does not parse.** Every edit takes the store's lock, reads the
//! file fresh, changes it and writes it atomically (`.tmp` + rename). A file that cannot be read or parsed is an error
//! the owner must see (`StoreError::Corrupt`): it is never replaced by a list built from nothing (the bot refuses to start
//! on one for the same reason: a corrupt file silently read as "no friends" would make it attack a friend).

use std::path::PathBuf;

use ddai_botctl::names::fold_name;
use ddai_botctl::relations::{ListKind, Relations, RelationsLock};
use serde::Serialize;
use tokio::sync::Mutex;

/// Most entries on one list (the file stays small; a list this long is not an editor's job).
pub const MAX_ENTRIES_PER_LIST: usize = 1000;
/// Longest name or clan the editor accepts, in bytes (DDNet's own limits are 15 and 11).
pub const MAX_NAME_BYTES: usize = 64;

/// Why an edit or a read failed. Never holds an entry's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StoreError {
    #[error("the name is empty after normalisation")]
    Empty,
    #[error("the name is too long")]
    TooLong,
    #[error("the name has control characters")]
    Control,
    #[error("that list is full")]
    Full,
    /// The lists file exists but cannot be read or does not parse: left untouched.
    #[error("the lists file cannot be read")]
    Corrupt,
    #[error("the lists file cannot be written")]
    Write,
}

/// All five lists, as normalised names (the exact keys the bot matches on).
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct RelationsView {
    pub friend: Vec<String>,
    pub war: Vec<String>,
    pub ignore: Vec<String>,
    pub clanwar: Vec<String>,
    pub clanfriend: Vec<String>,
}

impl RelationsView {
    fn of(r: &Relations) -> RelationsView {
        let names = |k: ListKind| r.names(k).into_iter().map(str::to_string).collect::<Vec<_>>();
        RelationsView {
            friend: names(ListKind::Friend),
            war: names(ListKind::War),
            ignore: names(ListKind::Ignore),
            clanwar: names(ListKind::ClanWar),
            clanfriend: names(ListKind::ClanFriend),
        }
    }

    pub fn list(&self, kind: ListKind) -> &[String] {
        match kind {
            ListKind::Friend => &self.friend,
            ListKind::War => &self.war,
            ListKind::Ignore => &self.ignore,
            ListKind::ClanWar => &self.clanwar,
            ListKind::ClanFriend => &self.clanfriend,
        }
    }
}

/// What an add did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddOutcome {
    /// The normalised form that was stored (and that the bot matches on).
    pub folded: String,
    /// False when it was already on that list.
    pub changed: bool,
    /// The lists the name was taken off (a friend is never at war, ...: `Relations::exclusions`).
    pub moved_from: Vec<ListKind>,
    pub view: RelationsView,
    /// `Relations::digest` of the file as read back after the write: what the bot must report after its reload.
    pub digest: String,
}

/// What a remove did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoveOutcome {
    pub folded: String,
    /// False when it was not on that list.
    pub changed: bool,
    pub view: RelationsView,
    pub digest: String,
}

/// The lists file with its locks: the in-process one orders this process's edits, and a `flock` on `<file>.lock`
/// ([`RelationsLock`]) orders them against the bot's console edits. The file lock is taken **inside** the blocking task, so
/// it is held for the whole read-modify-write even when the HTTP request that started it is cancelled.
pub struct RelationsStore {
    path: PathBuf,
    lock: Mutex<()>,
}

/// Checks a typed name or clan before it is folded.
pub fn validate_name(raw: &str) -> Result<String, StoreError> {
    if raw.len() > MAX_NAME_BYTES {
        return Err(StoreError::TooLong);
    }
    if raw.chars().any(char::is_control) {
        return Err(StoreError::Control);
    }
    let folded = fold_name(raw);
    if folded.is_empty() {
        return Err(StoreError::Empty);
    }
    Ok(folded)
}

fn load(path: &std::path::Path) -> Result<Relations, StoreError> {
    Relations::load(path).map_err(|_| StoreError::Corrupt)
}

impl RelationsStore {
    pub fn new(path: PathBuf) -> RelationsStore {
        RelationsStore {
            path,
            lock: Mutex::new(()),
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// The lists as they are in the file (a missing file is five empty lists).
    pub async fn view(&self) -> Result<RelationsView, StoreError> {
        let _guard = self.lock.lock().await;
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || load(&path).map(|r| RelationsView::of(&r)))
            .await
            .map_err(|_| StoreError::Corrupt)?
    }

    /// Puts `raw` on `kind` (taking it off the lists it excludes) and saves.
    pub async fn add(&self, kind: ListKind, raw: &str) -> Result<AddOutcome, StoreError> {
        let folded = validate_name(raw)?;
        let _guard = self.lock.lock().await;
        let path = self.path.clone();
        let raw = raw.to_string();
        tokio::task::spawn_blocking(move || {
            let _lock = RelationsLock::acquire(&path).map_err(|_| StoreError::Write)?;
            let mut r = load(&path)?;
            let already = r.contains_key(kind, &folded);
            if !already && r.len(kind) >= MAX_ENTRIES_PER_LIST {
                return Err(StoreError::Full);
            }
            let moved_from: Vec<ListKind> = Relations::exclusions(kind)
                .iter()
                .copied()
                .filter(|&o| r.contains_key(o, &folded))
                .collect();
            if !already || !moved_from.is_empty() {
                r.add(kind, &raw);
                r.save(&path).map_err(|_| StoreError::Write)?;
            }
            // Re-read what was written: the view is exactly what the bot will load.
            let back = load(&path)?;
            Ok(AddOutcome {
                folded,
                changed: !already || !moved_from.is_empty(),
                moved_from,
                view: RelationsView::of(&back),
                digest: back.digest(),
            })
        })
        .await
        .map_err(|_| StoreError::Write)?
    }

    /// Takes `raw` off `kind` and saves.
    pub async fn remove(&self, kind: ListKind, raw: &str) -> Result<RemoveOutcome, StoreError> {
        let folded = validate_name(raw)?;
        let _guard = self.lock.lock().await;
        let path = self.path.clone();
        let raw = raw.to_string();
        tokio::task::spawn_blocking(move || {
            let _lock = RelationsLock::acquire(&path).map_err(|_| StoreError::Write)?;
            let mut r = load(&path)?;
            let changed = r.remove(kind, &raw);
            if changed {
                r.save(&path).map_err(|_| StoreError::Write)?;
            }
            let back = load(&path)?;
            Ok(RemoveOutcome {
                folded,
                changed,
                view: RelationsView::of(&back),
                digest: back.digest(),
            })
        })
        .await
        .map_err(|_| StoreError::Write)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn store() -> (tempfile::TempDir, RelationsStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = RelationsStore::new(dir.path().join("bot").join("relations.json"));
        (dir, s)
    }

    #[tokio::test]
    async fn add_and_remove_round_trip_through_the_file_in_normalised_form() {
        let (_d, s) = store();
        assert_eq!(
            s.view().await.unwrap(),
            RelationsView::default(),
            "a missing file is empty lists"
        );
        let added = s.add(ListKind::Friend, "  Some   NICK ").await.unwrap();
        assert_eq!(added.folded, "some nick");
        assert!(added.changed && added.moved_from.is_empty());
        assert_eq!(added.view.friend, vec!["some nick".to_string()]);
        // The file is what the bot loads: the same normalisation, exact match.
        let loaded = Relations::load(s.path()).unwrap();
        assert!(loaded.contains(ListKind::Friend, "SOME nick"));
        assert!(!loaded.contains(ListKind::Friend, "some"), "no substring matching");
        assert!(!loaded.contains(ListKind::Friend, "some nicks"));
        assert_eq!(
            std::fs::metadata(s.path()).unwrap().permissions().mode() & 0o777,
            0o600,
            "the file holds nicknames: owner only"
        );
        // Adding the same (differently spelled) name again changes nothing.
        let again = s.add(ListKind::Friend, "some nick").await.unwrap();
        assert!(!again.changed);
        assert_eq!(again.view.friend.len(), 1);
        // Removing by any spelling works; removing what is not there is a no-op.
        let gone = s.remove(ListKind::Friend, "(3)SOME NICK").await.unwrap();
        assert!(gone.changed && gone.view.friend.is_empty());
        let nothing = s.remove(ListKind::Friend, "some nick").await.unwrap();
        assert!(!nothing.changed);
        assert_eq!(s.view().await.unwrap(), RelationsView::default());
    }

    #[tokio::test]
    async fn the_lists_exclude_each_other_like_the_console_and_say_so() {
        let (_d, s) = store();
        s.add(ListKind::Friend, "a").await.unwrap();
        let war = s.add(ListKind::War, "A").await.unwrap();
        assert_eq!(
            war.moved_from,
            vec![ListKind::Friend],
            "war takes a name off the friend list"
        );
        assert!(war.view.friend.is_empty() && war.view.war == vec!["a".to_string()]);
        let ignore = s.add(ListKind::Ignore, "a").await.unwrap();
        assert_eq!(ignore.moved_from, vec![ListKind::War]);
        s.add(ListKind::ClanWar, "Foes").await.unwrap();
        let cf = s.add(ListKind::ClanFriend, "foes").await.unwrap();
        assert_eq!(cf.moved_from, vec![ListKind::ClanWar]);
        assert_eq!(cf.view.clanwar, Vec::<String>::new());
        assert_eq!(cf.view.clanfriend, vec!["foes".to_string()]);
        // A name and a clan are different lists: the same text can be on both.
        s.add(ListKind::Friend, "foes").await.unwrap();
        let v = s.view().await.unwrap();
        assert!(v.friend.contains(&"foes".to_string()) && v.clanfriend.contains(&"foes".to_string()));
    }

    #[tokio::test]
    async fn bad_names_are_refused_without_touching_the_file() {
        let (_d, s) = store();
        for (raw, err) in [
            ("", StoreError::Empty),
            ("   ", StoreError::Empty),
            ("(12)", StoreError::Empty),
            ("a\nb", StoreError::Control),
            ("a\u{7}", StoreError::Control),
        ] {
            assert_eq!(s.add(ListKind::Friend, raw).await.unwrap_err(), err, "{raw:?}");
            assert_eq!(s.remove(ListKind::Friend, raw).await.unwrap_err(), err, "{raw:?}");
        }
        let long = "x".repeat(MAX_NAME_BYTES + 1);
        assert_eq!(s.add(ListKind::Friend, &long).await.unwrap_err(), StoreError::TooLong);
        assert!(!s.path().exists(), "nothing was written");
        s.add(ListKind::Friend, &"x".repeat(MAX_NAME_BYTES)).await.unwrap();
    }

    #[tokio::test]
    async fn a_corrupt_file_is_an_error_and_is_never_overwritten() {
        let (_d, s) = store();
        std::fs::create_dir_all(s.path().parent().unwrap()).unwrap();
        std::fs::write(s.path(), "{ not json").unwrap();
        assert_eq!(s.view().await.unwrap_err(), StoreError::Corrupt);
        assert_eq!(s.add(ListKind::Friend, "x").await.unwrap_err(), StoreError::Corrupt);
        assert_eq!(s.remove(ListKind::Friend, "x").await.unwrap_err(), StoreError::Corrupt);
        assert_eq!(std::fs::read_to_string(s.path()).unwrap(), "{ not json");
    }

    #[tokio::test]
    async fn a_list_has_a_size_cap() {
        let (_d, s) = store();
        let mut r = Relations::new();
        for i in 0..MAX_ENTRIES_PER_LIST {
            r.add(ListKind::Friend, &format!("n{i}"));
        }
        std::fs::create_dir_all(s.path().parent().unwrap()).unwrap();
        r.save(s.path()).unwrap();
        assert_eq!(s.add(ListKind::Friend, "one more").await.unwrap_err(), StoreError::Full);
        assert!(
            !s.add(ListKind::Friend, "n5").await.unwrap().changed,
            "an existing entry is not 'full'"
        );
        assert!(
            s.add(ListKind::War, "one more").await.is_ok(),
            "other lists are separate"
        );
    }

    #[tokio::test]
    async fn concurrent_edits_all_land() {
        let (_d, s) = store();
        let s = std::sync::Arc::new(s);
        let tasks: Vec<_> = (0..20)
            .map(|i| {
                let s = s.clone();
                tokio::spawn(async move { s.add(ListKind::Friend, &format!("p{i}")).await.unwrap() })
            })
            .collect();
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(s.view().await.unwrap().friend.len(), 20, "no lost update");
    }

    #[test]
    fn the_view_has_one_entry_per_list_kind_in_the_documented_shape() {
        let v = RelationsView::default();
        let json = serde_json::to_value(&v).unwrap();
        for k in ListKind::ALL {
            assert!(json.get(k.name()).is_some_and(|x| x.is_array()), "{}", k.name());
            assert!(v.list(k).is_empty());
        }
    }
}
