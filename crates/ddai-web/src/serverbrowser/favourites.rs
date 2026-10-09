//! The favourites file's writer (the web is the only one: `<launch-dir>/favourites.json`). Every change is read-modify-write under one
//! lock, validated as a whole with [`ddai_client::favourites`] (the same strict rules the root helper and the bot apply when they
//! read it), and written atomically. A file that cannot be trusted is never overwritten: the owner must look at it first.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ddai_client::favourites::{self, FILE_NAME, FavouriteError, Favourites, LoadError, Rules};

use crate::launch::write_atomic;

/// Why a change was not made. Every variant has a fixed code for the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// The change itself is not allowed (see [`FavouriteError`]).
    Refused(FavouriteError),
    /// The favourite asked for is not there.
    NotFound,
    /// The favourites file exists but cannot be trusted: nothing is written over it.
    FileUnusable(&'static str),
    /// The launch directory is missing or the file could not be written.
    WriteFailed,
    /// The allow-list names a server by a host name: favourites are refused while one exists.
    AllowListNotLiteral,
}

impl StoreError {
    pub fn code(self) -> &'static str {
        match self {
            StoreError::Refused(e) => e.code(),
            StoreError::NotFound => "not_found",
            StoreError::FileUnusable(code) => code,
            StoreError::WriteFailed => "favourites_write_failed",
            StoreError::AllowListNotLiteral => "allowlist_not_literal",
        }
    }
}

pub struct FavouritesStore {
    path: PathBuf,
    rules: Rules,
    lock: Mutex<()>,
}

impl FavouritesStore {
    pub fn new(launch_dir: &Path, rules: Rules) -> FavouritesStore {
        FavouritesStore {
            path: launch_dir.join(FILE_NAME),
            rules,
            lock: Mutex::new(()),
        }
    }

    pub fn rules(&self) -> Rules {
        self.rules
    }

    /// The current list (a missing file is an empty list).
    pub fn load(&self) -> Result<Favourites, &'static str> {
        favourites::load(&self.path, self.rules).map_err(|e| match e {
            LoadError::Unreadable => "favourites_unreadable",
            LoadError::Invalid(_) => "favourites_invalid",
        })
    }

    /// Applies `change` to the list under the lock and writes the result. `change` returns what the caller wants back.
    pub fn change<T>(&self, change: impl FnOnce(&mut Favourites) -> Result<T, StoreError>) -> Result<T, StoreError> {
        let _guard = self.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut list = self.load().map_err(StoreError::FileUnusable)?;
        let out = change(&mut list)?;
        let bytes = list.to_bytes(self.rules).map_err(StoreError::Refused)?;
        let dir = self.path.parent().ok_or(StoreError::WriteFailed)?;
        if !dir.is_dir() {
            return Err(StoreError::WriteFailed);
        }
        write_atomic(dir, FILE_NAME, &bytes, 0o644).map_err(|_| StoreError::WriteFailed)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_client::favourites::Favourite;

    fn fav(addr: &str) -> Favourite {
        Favourite {
            address: addr.to_string(),
            name: "S".into(),
            nick: "Muha".into(),
            connection: "direct".into(),
            consent_at: 5,
            notes: String::new(),
            added_at: 5,
            reopened_at: 0,
        }
    }

    #[test]
    fn changes_are_validated_written_and_a_broken_file_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let store = FavouritesStore::new(dir.path(), Rules::default());
        assert!(store.load().unwrap().favourites.is_empty());
        store
            .change(|l| {
                l.favourites.push(fav("93.184.216.35:8308"));
                Ok(())
            })
            .unwrap();
        assert_eq!(store.load().unwrap().favourites.len(), 1);
        #[cfg(unix)]
        {
            let mode = std::os::unix::fs::PermissionsExt::mode(
                &std::fs::metadata(dir.path().join(FILE_NAME)).unwrap().permissions(),
            );
            assert_eq!(mode & 0o777, 0o644);
        }
        // An invalid result is refused and the file stays as it was.
        let before = std::fs::read(dir.path().join(FILE_NAME)).unwrap();
        let err = store
            .change(|l| {
                l.favourites.push(fav("10.0.0.1:8303"));
                Ok(())
            })
            .unwrap_err();
        assert_eq!(err, StoreError::Refused(FavouriteError::BadAddress));
        let err = store
            .change(|l| {
                l.favourites.push(fav("93.184.216.35:8308"));
                Ok(())
            })
            .unwrap_err();
        assert_eq!(err, StoreError::Refused(FavouriteError::Duplicate));
        assert_eq!(std::fs::read(dir.path().join(FILE_NAME)).unwrap(), before);
        // A file that cannot be trusted is left alone, whatever the change.
        std::fs::write(dir.path().join(FILE_NAME), b"{not json").unwrap();
        let err = store.change(|_| Ok(())).unwrap_err();
        assert_eq!(err, StoreError::FileUnusable("favourites_invalid"));
        assert_eq!(std::fs::read(dir.path().join(FILE_NAME)).unwrap(), b"{not json");
        // No launch directory: a write failure, nothing created.
        let store = FavouritesStore::new(&dir.path().join("nope"), Rules::default());
        assert_eq!(store.change(|_| Ok(())).unwrap_err(), StoreError::WriteFailed);
    }
}
