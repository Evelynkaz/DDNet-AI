//! Owner-only files and directories.
//!
//! * **Unix**: permission bits, `0600` for files and `0700` for directories.
//! * **Windows**: an access-control list that grants full control to the current user alone and nothing to anybody else. The ACL is
//!   set with the system's own `icacls.exe` (`/inheritance:r /grant:r <user>:F`), which needs no `unsafe` and no Win32 binding. If it
//!   cannot be run or fails, the call still succeeds but reports [`Protection::ParentInherited`] with the reason: the caller must log a
//!   warning, and the file keeps the permissions its parent directory gave it, which for everything under the user's profile
//!   (`%USERPROFILE%`) is "this user, the administrators and the system", never "everyone". A file created by [`OwnerOnly::owner_only`]
//!   on Windows has no extra protection from that call alone: call [`restrict_file`] on it **before** writing secret content.

use std::fs::OpenOptions;
use std::io;
use std::path::Path;

/// How well a path ended up protected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Protection {
    /// Unix: the mode bits were set. Windows: the ACL was replaced by one granting the current user alone.
    Exact,
    /// Windows only: the ACL could not be set (`reason`); the path keeps what its parent gave it. The caller must warn.
    ParentInherited { reason: String },
}

/// `OpenOptions` that create the file private to its owner.
pub trait OwnerOnly {
    /// Unix: `.mode(0o600)`, so the file is private from the moment it exists (not after a `chmod`). Windows: nothing yet, see the
    /// module docs; follow up with [`restrict_file`].
    fn owner_only(&mut self) -> &mut Self;
}

impl OwnerOnly for OpenOptions {
    #[cfg(unix)]
    fn owner_only(&mut self) -> &mut Self {
        use std::os::unix::fs::OpenOptionsExt;
        self.mode(0o600)
    }

    #[cfg(not(unix))]
    fn owner_only(&mut self) -> &mut Self {
        self
    }
}

/// Makes the file at `path` readable and writable by its owner alone (idempotent).
pub fn restrict_file(path: &Path) -> io::Result<Protection> {
    imp::restrict(path, false)
}

/// Makes the directory at `path` (and, on Windows, what it will contain) accessible to its owner alone (idempotent).
pub fn restrict_dir(path: &Path) -> io::Result<Protection> {
    imp::restrict(path, true)
}

/// Creates `path` and its missing parents; the directories it creates are owner-only (on Unix every created component, on Windows the
/// leaf; the parents inherit from the user's profile).
pub fn create_dir_all_restricted(path: &Path) -> io::Result<Protection> {
    imp::create_dir_all_restricted(path)
}

/// Whether nobody but the owner can access `path`: Unix, no group/other permission bits; Windows, an ACL with exactly one entry,
/// the current user's.
pub fn is_restricted(path: &Path) -> io::Result<bool> {
    imp::is_restricted(path)
}

/// Writes `contents` to a new file at `path`, readable by its owner alone from the first byte, durably. Fails if `path` exists.
pub fn write_new_private(path: &Path, contents: &[u8]) -> io::Result<Protection> {
    use std::io::Write;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .owner_only()
        .open(path)?;
    let protection = restrict_file(path)?;
    file.write_all(contents)?;
    file.sync_all()?;
    Ok(protection)
}

#[cfg(unix)]
mod imp {
    use super::Protection;
    use std::fs;
    use std::io;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::path::Path;

    pub fn restrict(path: &Path, dir: bool) -> io::Result<Protection> {
        fs::set_permissions(path, fs::Permissions::from_mode(if dir { 0o700 } else { 0o600 }))?;
        Ok(Protection::Exact)
    }

    pub fn create_dir_all_restricted(path: &Path) -> io::Result<Protection> {
        // Created with mode 0700 (cut by the umask only towards stricter); a directory that already exists keeps its mode.
        fs::DirBuilder::new().recursive(true).mode(0o700).create(path)?;
        Ok(Protection::Exact)
    }

    pub fn is_restricted(path: &Path) -> io::Result<bool> {
        Ok(fs::metadata(path)?.permissions().mode() & 0o077 == 0)
    }
}

#[cfg(windows)]
mod imp {
    use super::{Protection, icacls};
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::process::Command;

    pub fn restrict(path: &Path, dir: bool) -> io::Result<Protection> {
        let result = icacls::current_principal().and_then(|who| {
            let out = Command::new(icacls::exe())
                .args(icacls::restrict_args(path, &who, dir))
                .output()?;
            if out.status.success() {
                Ok(())
            } else {
                Err(io::Error::other(format!(
                    "icacls exited with {}: {}",
                    out.status,
                    String::from_utf8_lossy(&out.stdout).trim()
                )))
            }
        });
        Ok(match result {
            Ok(()) => Protection::Exact,
            Err(e) => Protection::ParentInherited { reason: e.to_string() },
        })
    }

    pub fn create_dir_all_restricted(path: &Path) -> io::Result<Protection> {
        fs::create_dir_all(path)?;
        restrict(path, true)
    }

    pub fn is_restricted(path: &Path) -> io::Result<bool> {
        let who = icacls::current_principal()?;
        let out = Command::new(icacls::exe()).arg(path).output()?;
        if !out.status.success() {
            return Err(io::Error::other(format!("icacls exited with {}", out.status)));
        }
        let entries = icacls::entries(&String::from_utf8_lossy(&out.stdout), path);
        Ok(icacls::only_user(&entries, &who))
    }
}

/// The command lines and the output of `icacls.exe`, as pure functions (tested on every platform; only run on Windows).
#[cfg_attr(not(windows), allow(dead_code))]
mod icacls {
    use std::ffi::OsString;
    use std::io;
    use std::path::{Path, PathBuf};

    /// `%SystemRoot%\System32\icacls.exe` (not looked up on `PATH`), or plain `icacls`.
    pub fn exe() -> PathBuf {
        match std::env::var_os("SystemRoot") {
            Some(root) if !root.is_empty() => PathBuf::from(root).join("System32").join("icacls.exe"),
            _ => PathBuf::from("icacls"),
        }
    }

    /// `DOMAIN\user` (or just `user` without a domain), from the environment of the logon session.
    pub fn current_principal() -> io::Result<String> {
        let user = std::env::var("USERNAME").ok().filter(|u| !u.is_empty());
        let domain = std::env::var("USERDOMAIN").ok().filter(|d| !d.is_empty());
        match (domain, user) {
            (Some(d), Some(u)) => Ok(format!("{d}\\{u}")),
            (None, Some(u)) => Ok(u),
            _ => Err(io::Error::other("USERNAME is not set")),
        }
    }

    /// The arguments that replace the ACL of `path` by "`who` has full control" (for a directory: inherited by what it will contain).
    pub fn restrict_args(path: &Path, who: &str, dir: bool) -> Vec<OsString> {
        let grant = if dir {
            format!("{who}:(OI)(CI)F")
        } else {
            format!("{who}:F")
        };
        vec![
            path.as_os_str().to_owned(),
            "/inheritance:r".into(),
            "/grant:r".into(),
            grant.into(),
        ]
    }

    /// The principals of the entries `icacls <path>` printed. The first line carries the path before its first entry; further lines
    /// are indented. A line is an entry when it holds `:(`.
    pub fn entries(output: &str, path: &Path) -> Vec<String> {
        let path = path.display().to_string();
        output
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                let line = line.strip_prefix(path.as_str()).unwrap_or(line).trim();
                let (principal, _) = line.split_once(":(")?;
                Some(principal.trim().to_string())
            })
            .filter(|p| !p.is_empty())
            .collect()
    }

    /// Exactly one entry and it is `who` (case-insensitive; `who` may be `user` alone while icacls prints `HOST\user`). `icacls` prints
    /// in the console's OEM code page, so a user name with non-ASCII letters (a Cyrillic account name) cannot be compared by text:
    /// there the single remaining entry (the one `restrict_args` leaves) is taken to be the user's own.
    pub fn only_user(entries: &[String], who: &str) -> bool {
        let user = who.rsplit('\\').next().unwrap_or(who);
        if entries.len() != 1 {
            return false;
        }
        if !user.is_ascii() {
            return true;
        }
        entries[0].to_lowercase().ends_with(&user.to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn icacls_command_line() {
        let a = icacls::restrict_args(Path::new("C:\\d\\secret.toml"), "PC\\me", false);
        let a: Vec<String> = a.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert_eq!(a, ["C:\\d\\secret.toml", "/inheritance:r", "/grant:r", "PC\\me:F"]);
        let d = icacls::restrict_args(Path::new("C:\\d"), "PC\\me", true);
        assert_eq!(d.last().unwrap().to_string_lossy(), "PC\\me:(OI)(CI)F");
    }

    #[test]
    fn icacls_output_is_read_into_principals() {
        let path = PathBuf::from("C:\\Users\\me\\ddnet-ai\\data\\secrets\\auth.toml");
        let one = format!(
            "{} PC\\me:(F)\n\nSuccessfully processed 1 files; Failed processing 0 files\n",
            path.display()
        );
        let e = icacls::entries(&one, &path);
        assert_eq!(e, ["PC\\me"]);
        assert!(icacls::only_user(&e, "PC\\me"));
        assert!(icacls::only_user(&e, "me"));
        let many = format!(
            "{} NT AUTHORITY\\SYSTEM:(I)(F)\n       BUILTIN\\Administrators:(I)(F)\n       PC\\me:(I)(F)\n       BUILTIN\\Users:(I)(RX)\n\nSuccessfully processed 1 files; Failed processing 0 files\n",
            path.display()
        );
        let e = icacls::entries(&many, &path);
        assert_eq!(
            e,
            [
                "NT AUTHORITY\\SYSTEM",
                "BUILTIN\\Administrators",
                "PC\\me",
                "BUILTIN\\Users"
            ]
        );
        assert!(!icacls::only_user(&e, "PC\\me"));
        assert!(!icacls::only_user(&[], "PC\\me"));
        assert!(!icacls::only_user(&["PC\\someone-else".to_string()], "PC\\me"));
        // A non-ASCII account name: icacls prints it in the OEM code page, so only the count can be checked.
        assert!(icacls::only_user(
            &["PC\\\u{fffd}\u{fffd}\u{fffd}".to_string()],
            "PC\\\u{418}\u{432}\u{430}\u{43d}"
        ));
        assert!(!icacls::only_user(
            &["A".to_string(), "B".to_string()],
            "PC\\\u{418}\u{432}\u{430}\u{43d}"
        ));
    }

    #[test]
    fn a_new_private_file_is_private() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secret");
        let p = write_new_private(&path, b"hush").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hush");
        if cfg!(unix) {
            assert_eq!(p, Protection::Exact);
            assert!(is_restricted(&path).unwrap());
        }
        // It refuses to overwrite.
        assert!(write_new_private(&path, b"again").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"hush");
    }

    #[cfg(unix)]
    #[test]
    fn unix_modes() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b");
        assert_eq!(create_dir_all_restricted(&nested).unwrap(), Protection::Exact);
        for d in [dir.path().join("a"), nested.clone()] {
            assert_eq!(
                std::fs::metadata(&d).unwrap().permissions().mode() & 0o777,
                0o700,
                "{d:?}"
            );
        }
        let file = nested.join("f");
        std::fs::write(&file, b"x").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(!is_restricted(&file).unwrap());
        assert_eq!(restrict_file(&file).unwrap(), Protection::Exact);
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(is_restricted(&file).unwrap());
    }

    /// On Windows CI: the real `icacls` leaves exactly one entry, the current user's, on a file and on a directory.
    #[cfg(windows)]
    #[test]
    fn windows_acl_is_the_current_user_alone() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = dir.path().join("secrets");
        let p = create_dir_all_restricted(&secrets).unwrap();
        assert_eq!(p, Protection::Exact, "icacls must work on the CI runner");
        assert!(is_restricted(&secrets).unwrap(), "directory ACL");
        let file = secrets.join("auth.toml");
        let p = write_new_private(&file, b"secret").unwrap();
        assert_eq!(p, Protection::Exact);
        assert!(is_restricted(&file).unwrap(), "file ACL");
        // We can still use the file ourselves.
        assert_eq!(std::fs::read(&file).unwrap(), b"secret");
        // A file made inside the restricted directory is born with only the directory's inherited entry (ours).
        let child = secrets.join("child");
        std::fs::write(&child, b"x").unwrap();
        assert!(is_restricted(&child).unwrap(), "inherited ACL of a child");
    }
}
