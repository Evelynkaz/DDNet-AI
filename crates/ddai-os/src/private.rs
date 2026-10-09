//! Owner-only files and directories.
//!
//! * **Unix**: permission bits, `0600` for files and `0700` for directories.
//! * **Windows**: an access-control list that grants full control to the current user alone and nothing to anybody else. The user is
//!   identified by the SID that `whoami /user` prints (not by the `USERNAME` environment variable, which anything can set), the ACL is
//!   set with the system's own `icacls.exe` (`/inheritance:r /grant:r *<SID>:F`, which needs no `unsafe` and no Win32 binding) and
//!   read back with `icacls /save` (the SDDL string, which does not depend on the language of the system or on a code page).
//!
//! When the ACL cannot be set or read (`icacls` missing or failing, a file system without ACLs such as FAT32), the call does **not**
//! quietly succeed: it succeeds, reporting [`Protection::ParentInherited`] with the reason, only for a path under the user's profile
//! folder (`%USERPROFILE%`), where Windows' default ACL is "this user, the administrators and the system" and never "everyone"; for any
//! other path (a `--data-dir` on `D:\`, `C:\ddnet-ai`, ...) it fails with `PermissionDenied`, unless the environment variable
//! `DDNET_AI_ALLOW_UNRESTRICTED=1` accepts the risk explicitly. [`is_restricted`] follows the same rule when it cannot read the ACL,
//! so the load path and the write path agree. A file created by [`OwnerOnly::owner_only`] on Windows has no extra protection from that
//! call alone: call [`restrict_file`] on it **before** writing secret content.

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

    fn run_icacls(args: &[std::ffi::OsString]) -> io::Result<std::process::Output> {
        let out = Command::new(icacls::exe()).args(args).output()?;
        if out.status.success() {
            Ok(out)
        } else {
            Err(io::Error::other(format!(
                "icacls exited with {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stdout).trim()
            )))
        }
    }

    pub fn restrict(path: &Path, dir: bool) -> io::Result<Protection> {
        let result =
            icacls::current_sid().and_then(|sid| run_icacls(&icacls::restrict_args(path, &sid, dir)).map(drop));
        match result {
            Ok(()) => Ok(Protection::Exact),
            Err(e) => {
                if icacls::tolerated(path) {
                    Ok(Protection::ParentInherited { reason: e.to_string() })
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!(
                            "could not restrict {} to this user ({e}); it is outside your profile folder, so other local users may be able to read it: move the data directory under your profile, or set {}=1 to accept that",
                            path.display(),
                            icacls::ALLOW_ENV
                        ),
                    ))
                }
            }
        }
    }

    pub fn create_dir_all_restricted(path: &Path) -> io::Result<Protection> {
        fs::create_dir_all(path)?;
        restrict(path, true)
    }

    /// The ACL as the SDDL string `icacls /save` writes, compared with the current user's SID.
    fn query(path: &Path) -> io::Result<bool> {
        let sid = icacls::current_sid()?;
        let tmp = std::env::temp_dir().join(format!(
            "ddnet-ai-acl-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        let saved = run_icacls(&icacls::save_args(path, &tmp)).and_then(|_| fs::read(&tmp));
        let _ = fs::remove_file(&tmp);
        let text = icacls::decode_save(&saved?);
        let aces = icacls::sddl_aces(&text)
            .ok_or_else(|| io::Error::other("icacls /save wrote no DACL that could be read"))?;
        Ok(icacls::only_sid(&aces, &sid))
    }

    pub fn is_restricted(path: &Path) -> io::Result<bool> {
        match query(path) {
            Ok(answer) => Ok(answer),
            Err(_) if icacls::tolerated(path) => Ok(true),
            Err(e) => Err(e),
        }
    }
}

/// The command lines and the output of `whoami.exe` and `icacls.exe`, as pure functions (tested on every platform; only run on
/// Windows).
#[cfg_attr(not(windows), allow(dead_code))]
mod icacls {
    use std::ffi::OsString;
    use std::io;
    use std::path::{Path, PathBuf};

    /// The environment variable that accepts a secret path whose ACL cannot be set (see the module docs of [`super`]).
    pub const ALLOW_ENV: &str = "DDNET_AI_ALLOW_UNRESTRICTED";

    fn system32(exe: &str) -> PathBuf {
        match std::env::var_os("SystemRoot") {
            Some(root) if !root.is_empty() => PathBuf::from(root).join("System32").join(exe),
            _ => PathBuf::from(exe),
        }
    }

    /// `%SystemRoot%\System32\icacls.exe` (not looked up on `PATH`), or plain `icacls`.
    pub fn exe() -> PathBuf {
        system32("icacls.exe")
    }

    /// The SID of the user running this process: the second field of `whoami /user /fo csv /nh`, which reads the process token. Cached.
    #[cfg(windows)]
    pub fn current_sid() -> io::Result<String> {
        use std::sync::OnceLock;
        static SID: OnceLock<Result<String, String>> = OnceLock::new();
        SID.get_or_init(|| {
            let out = std::process::Command::new(system32("whoami.exe"))
                .args(["/user", "/fo", "csv", "/nh"])
                .output()
                .map_err(|e| format!("whoami: {e}"))?;
            if !out.status.success() {
                return Err(format!("whoami exited with {}", out.status));
            }
            parse_whoami_user(&String::from_utf8_lossy(&out.stdout)).ok_or_else(|| "whoami printed no SID".to_string())
        })
        .clone()
        .map_err(io::Error::other)
    }

    /// Without Windows there is no `whoami /user`; the functions that call this are never reached.
    #[cfg(not(windows))]
    pub fn current_sid() -> io::Result<String> {
        Err(io::Error::other("no SID on this platform"))
    }

    /// The SID in the output of `whoami /user /fo csv /nh`: `"DOMAIN\user","S-1-5-21-..."`.
    pub fn parse_whoami_user(output: &str) -> Option<String> {
        let line = output.lines().map(str::trim).find(|l| !l.is_empty())?;
        let field = line.rsplit(',').next()?.trim().trim_matches('"').trim();
        let well_formed = field.len() > 4
            && field.starts_with("S-1-")
            && field.bytes().all(|b| b.is_ascii_digit() || b == b'-' || b == b'S');
        well_formed.then(|| field.to_string())
    }

    /// The arguments that replace the ACL of `path` by "the SID has full control" (for a directory: inherited by what it will contain).
    pub fn restrict_args(path: &Path, sid: &str, dir: bool) -> Vec<OsString> {
        let grant = if dir {
            format!("*{sid}:(OI)(CI)F")
        } else {
            format!("*{sid}:F")
        };
        vec![
            path.as_os_str().to_owned(),
            "/inheritance:r".into(),
            "/grant:r".into(),
            grant.into(),
        ]
    }

    /// `icacls <path> /save <out>`: writes the path and its security descriptor in SDDL.
    pub fn save_args(path: &Path, out: &Path) -> Vec<OsString> {
        vec![path.as_os_str().to_owned(), "/save".into(), out.as_os_str().to_owned()]
    }

    /// `icacls /save` writes UTF-16LE with a byte order mark (UTF-8 is accepted too, in case a version does that).
    pub fn decode_save(bytes: &[u8]) -> String {
        if let Some(rest) = bytes.strip_prefix(&[0xff, 0xfe]) {
            let units: Vec<u16> = rest.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
            String::from_utf16_lossy(&units)
        } else if let Some(rest) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
            String::from_utf8_lossy(rest).into_owned()
        } else {
            String::from_utf8_lossy(bytes).into_owned()
        }
    }

    /// One access-control entry of an SDDL DACL.
    #[derive(Debug, PartialEq, Eq)]
    pub struct Ace {
        /// `A` allow, `D` deny, ...
        pub kind: String,
        /// The trustee: a SID string, or a two-letter alias for a well-known one.
        pub trustee: String,
    }

    /// The entries of the DACL in the text `icacls /save` wrote (a line `D:<flags>(<ace>)(<ace>)...`; the SACL and owner parts are not
    /// part of it). `None` when there is no DACL line.
    pub fn sddl_aces(text: &str) -> Option<Vec<Ace>> {
        let line = text.lines().map(str::trim).find(|l| l.starts_with("D:"))?;
        let dacl = line[2..].split("S:").next().unwrap_or("");
        let mut aces = Vec::new();
        let mut rest = dacl;
        while let Some(open) = rest.find('(') {
            let close = open + rest[open..].find(')')?;
            let fields: Vec<&str> = rest[open + 1..close].split(';').collect();
            if fields.len() < 6 {
                return None;
            }
            aces.push(Ace {
                kind: fields[0].to_string(),
                trustee: fields[5].to_string(),
            });
            rest = &rest[close + 1..];
        }
        Some(aces)
    }

    /// Exactly one entry, it allows, and its trustee is `sid` (compared whole, case-insensitively; the built-in Administrator account
    /// (RID 500) is printed as the alias `LA`).
    pub fn only_sid(aces: &[Ace], sid: &str) -> bool {
        match aces {
            [ace] => {
                ace.kind == "A"
                    && (ace.trustee.eq_ignore_ascii_case(sid)
                        || (ace.trustee.eq_ignore_ascii_case("LA") && sid.ends_with("-500")))
            }
            _ => false,
        }
    }

    /// Whether `path` is inside `profile` (the user's profile folder): compared on normalised text (case-insensitive, `\` and `/` the
    /// same, no `\\?\` prefix, no trailing separator), whole components only (`C:\Users\me2` is not inside `C:\Users\me`).
    pub fn is_under(path: &str, profile: &str) -> bool {
        fn norm(s: &str) -> String {
            let s = s.strip_prefix("\\\\?\\").unwrap_or(s);
            s.replace('\\', "/").trim_end_matches('/').to_lowercase()
        }
        let (path, profile) = (norm(path), norm(profile));
        !profile.is_empty() && (path == profile || path.starts_with(&format!("{profile}/")))
    }

    /// May a path whose ACL could not be set or read be used for a secret anyway? Under the user's profile folder (Windows' default ACL
    /// there is user-only), or when the user accepted the risk with [`ALLOW_ENV`].
    #[cfg(windows)]
    pub fn tolerated(path: &Path) -> bool {
        if std::env::var(ALLOW_ENV).is_ok_and(|v| v == "1") {
            return true;
        }
        let Some(profile) = std::env::var_os("USERPROFILE") else {
            return false;
        };
        let canonical = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        is_under(
            &canonical(path).to_string_lossy(),
            &canonical(Path::new(&profile)).to_string_lossy(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SID: &str = "S-1-5-21-1004336348-1177238915-682003330-1001";

    #[test]
    fn icacls_command_line_names_the_sid_not_the_user() {
        let a = icacls::restrict_args(Path::new("C:\\d\\secret.toml"), SID, false);
        let a: Vec<String> = a.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert_eq!(
            a,
            ["C:\\d\\secret.toml", "/inheritance:r", "/grant:r", &format!("*{SID}:F")]
        );
        let d = icacls::restrict_args(Path::new("C:\\d"), SID, true);
        assert_eq!(d.last().unwrap().to_string_lossy(), format!("*{SID}:(OI)(CI)F"));
        let s = icacls::save_args(Path::new("C:\\d\\f"), Path::new("C:\\t\\o.txt"));
        assert_eq!(s.len(), 3);
        assert_eq!(s[1].to_string_lossy(), "/save");
    }

    #[test]
    fn whoami_output_is_read_into_a_sid() {
        let out = format!("\"PC\\me\",\"{SID}\"\r\n");
        assert_eq!(icacls::parse_whoami_user(&out).as_deref(), Some(SID));
        // A domain account whose name has a comma-free but non-ASCII name, and a line break at the start.
        let out = format!("\n\"\u{418}\u{432}\u{430}\u{43d}\\\u{418}\u{432}\u{430}\u{43d}\",\"{SID}\"\n");
        assert_eq!(icacls::parse_whoami_user(&out).as_deref(), Some(SID));
        assert_eq!(icacls::parse_whoami_user(""), None);
        assert_eq!(icacls::parse_whoami_user("ERROR: Access is denied.\n"), None);
        assert_eq!(icacls::parse_whoami_user("\"PC\\me\",\"not-a-sid\"\n"), None);
    }

    #[test]
    fn the_saved_acl_is_read_from_utf16_and_utf8() {
        let sddl = format!("C:\\Users\\me\\secrets\nD:PAI(A;OICIID;FA;;;{SID})\n");
        let mut utf16 = vec![0xff, 0xfe];
        utf16.extend(sddl.encode_utf16().flat_map(u16::to_le_bytes));
        assert_eq!(icacls::decode_save(&utf16), sddl);
        assert_eq!(icacls::decode_save(sddl.as_bytes()), sddl);
        let aces = icacls::sddl_aces(&icacls::decode_save(&utf16)).unwrap();
        assert_eq!(
            aces,
            [icacls::Ace {
                kind: "A".into(),
                trustee: SID.into()
            }]
        );
        assert!(icacls::only_sid(&aces, SID));
        assert!(icacls::only_sid(&aces, &SID.to_lowercase()));
    }

    #[test]
    fn only_the_whole_sid_of_a_single_allow_entry_counts() {
        let parse = |s: &str| icacls::sddl_aces(s).unwrap();
        // The usual Windows default for a file under the profile: system, administrators, the user.
        let many = parse(&format!("f\nD:PAI(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;{SID})\n"));
        assert_eq!(many.len(), 3);
        assert!(!icacls::only_sid(&many, SID));
        // Two entries are never "only", even when both are ours.
        assert!(!icacls::only_sid(
            &parse(&format!("D:P(A;;FA;;;{SID})(A;;FR;;;{SID})")),
            SID
        ));
        // A different user, and a SID that merely ends with ours (the suffix bug the first version had).
        assert!(!icacls::only_sid(&parse("D:P(A;;FA;;;S-1-5-21-1-2-3-1002)"), SID));
        assert!(!icacls::only_sid(
            &parse(&format!("D:P(A;;FA;;;S-1-5-21-1-2-3-9{SID})")),
            SID
        ));
        assert!(!icacls::only_sid(&parse(&format!("D:P(A;;FA;;;{SID}0)")), SID));
        // A deny entry is not an allow for us.
        assert!(!icacls::only_sid(&parse(&format!("D:P(D;;FA;;;{SID})")), SID));
        // The built-in Administrator is printed as the alias LA.
        assert!(icacls::only_sid(&parse("D:P(A;;FA;;;LA)"), "S-1-5-21-1-2-3-500"));
        assert!(!icacls::only_sid(&parse("D:P(A;;FA;;;LA)"), SID));
        // Everyone and Authenticated Users never match.
        assert!(!icacls::only_sid(&parse("D:P(A;;FA;;;WD)"), SID));
        assert!(!icacls::only_sid(&parse("D:P(A;;0x1200a9;;;AU)"), SID));
        // No DACL, a broken one, and an empty one (nobody at all).
        assert!(icacls::sddl_aces("just a path\n").is_none());
        assert!(icacls::sddl_aces("D:P(A;;FA;;;").is_none());
        assert!(icacls::sddl_aces("D:P(A;FA)").is_none());
        assert_eq!(icacls::sddl_aces("D:P").unwrap(), []);
        assert!(!icacls::only_sid(&[], SID));
        // The SACL part of the line is not read as entries.
        let with_sacl = parse(&format!("D:P(A;;FA;;;{SID})S:(AU;SA;FA;;;WD)"));
        assert_eq!(with_sacl.len(), 1);
    }

    #[test]
    fn the_profile_test_compares_whole_components() {
        let profile = "C:\\Users\\me";
        assert!(icacls::is_under("C:\\Users\\me\\ddnet-ai\\data", profile));
        assert!(icacls::is_under("c:\\users\\ME\\x", profile));
        assert!(icacls::is_under("\\\\?\\C:\\Users\\me\\x", profile));
        assert!(icacls::is_under("C:/Users/me/", profile));
        assert!(icacls::is_under("C:\\Users\\me", profile));
        assert!(!icacls::is_under("C:\\Users\\me2\\x", profile));
        assert!(!icacls::is_under("D:\\data", profile));
        assert!(!icacls::is_under("C:\\ddnet-ai", profile));
        assert!(!icacls::is_under("C:\\Users", profile));
        assert!(!icacls::is_under("C:\\Users\\me\\x", ""));
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
