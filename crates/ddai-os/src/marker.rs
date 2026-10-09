//! "Marker files" that switch a feature off (`selfkill.off`, `preinput.off`, `duel-detect.off`, `window-model.off`,
//! `owner-chat.off`): the bot looks for them and, when it cannot tell, must behave as if the marker were there.
//!
//! The fail-safe rule is the same on every platform: the marker counts as **absent** only when the file system says plainly that
//! it is not there. Any other answer (permission denied, an I/O error, a path that runs through something that is not a directory)
//! leaves the state unknown, and unknown means "present" (the feature stays off).
//!
//! The platforms disagree about the last case: below a regular file, Linux says `ENOTDIR` (not `NotFound`), Windows says
//! `ERROR_PATH_NOT_FOUND`, which Rust reports as `NotFound`. A plain "only `NotFound` means absent" check is therefore fail-safe on
//! Linux and fail-open on Windows. [`is_present`] closes that: on `NotFound` it walks up the ancestors, and if the nearest one that
//! exists is not a directory (or cannot be looked at), the marker's state is unknown and it counts as present.

use std::io::ErrorKind;
use std::path::Path;

/// Whether the marker at `path` counts as present (see the module docs): it exists in any form (a file, a directory, a dangling
/// symlink), or its state cannot be determined.
#[must_use]
pub fn is_present(path: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Ok(_) => true,
        Err(e) if e.kind() != ErrorKind::NotFound => true,
        Err(_) => an_ancestor_is_not_a_directory(path),
    }
}

/// On `NotFound`: is the nearest existing ancestor something other than a directory, or unreadable? (A symlink to a directory is a
/// directory here: `metadata` follows it. A missing ancestor is skipped; the walk goes on to its parent.)
fn an_ancestor_is_not_a_directory(path: &Path) -> bool {
    for ancestor in path.ancestors().skip(1) {
        if ancestor.as_os_str().is_empty() {
            return false;
        }
        match std::fs::metadata(ancestor) {
            Ok(m) => return !m.is_dir(),
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(_) => return true,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_marker_in_an_existing_directory_is_absent() {
        let d = tempfile::tempdir().unwrap();
        assert!(!is_present(&d.path().join("x.off")));
    }

    #[test]
    fn a_missing_marker_in_a_missing_directory_is_absent() {
        let d = tempfile::tempdir().unwrap();
        assert!(!is_present(&d.path().join("nowhere").join("deeper").join("x.off")));
    }

    #[test]
    fn a_marker_in_any_form_is_present() {
        let d = tempfile::tempdir().unwrap();
        let m = d.path().join("x.off");
        std::fs::write(&m, "").unwrap();
        assert!(is_present(&m), "an empty file counts");
        std::fs::remove_file(&m).unwrap();
        std::fs::create_dir(&m).unwrap();
        assert!(is_present(&m), "a directory counts");
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_counts() {
        // (Making a symlink on Windows needs a privilege.)
        let d = tempfile::tempdir().unwrap();
        let m = d.path().join("x.off");
        std::os::unix::fs::symlink("/nonexistent", &m).unwrap();
        assert!(is_present(&m));
    }

    #[test]
    fn a_path_below_a_regular_file_is_unknown_and_counts_as_present_on_every_platform() {
        let d = tempfile::tempdir().unwrap();
        let file = d.path().join("bot");
        std::fs::write(&file, b"").unwrap();
        assert!(is_present(&file.join("x.off")), "ENOTDIR on Linux, NotFound on Windows");
        assert!(
            is_present(&file.join("deeper").join("x.off")),
            "a missing directory below the file too"
        );
    }

    /// The part Windows needs: it is reached there because the system says `NotFound`; on Linux the system says `ENOTDIR` and the walk
    /// is not reached, so it is tested directly.
    #[test]
    fn the_ancestor_walk_finds_a_file_where_a_directory_should_be() {
        let d = tempfile::tempdir().unwrap();
        let file = d.path().join("bot");
        std::fs::write(&file, b"").unwrap();
        assert!(an_ancestor_is_not_a_directory(&file.join("x.off")));
        assert!(an_ancestor_is_not_a_directory(&file.join("a").join("b").join("x.off")));
        assert!(!an_ancestor_is_not_a_directory(&d.path().join("x.off")));
        assert!(!an_ancestor_is_not_a_directory(
            &d.path().join("a").join("b").join("x.off")
        ));
        assert!(!an_ancestor_is_not_a_directory(Path::new("x.off")));
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_directory_counts_as_present() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("locked");
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o000)).unwrap();
        // (root can look anyway: then the marker is plainly absent, which is also right.)
        let readable = std::fs::symlink_metadata(dir.join("x.off")).is_err_and(|e| e.kind() == ErrorKind::NotFound);
        assert_eq!(is_present(&dir.join("x.off")), !readable);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_to_a_directory_is_a_directory_for_the_walk() {
        let d = tempfile::tempdir().unwrap();
        let real = d.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = d.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(!is_present(&link.join("x.off")));
    }

    #[test]
    fn a_relative_path_without_parents_is_absent() {
        assert!(!is_present(Path::new("definitely-not-here-x.off")));
    }
}
