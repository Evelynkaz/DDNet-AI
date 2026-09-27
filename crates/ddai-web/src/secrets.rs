//! Everything under `<data_dir>/secrets`: the owner's password hash, the plaintext handed to the
//! owner once, and the session-cookie signing key. All three live outside the git repository (see
//! `CLAUDE.md` "Никогда не коммитить"), in files created with `0600` permissions inside a `0700`
//! directory, never logged.
//!
//! File formats are the tiny [`crate::toml_kv`] subset, not a full TOML parser (see its module
//! doc for why).

use crate::rand_util::{decode_b64, encode_b64, random_bytes};
use crate::toml_kv::{self, Value};
use argon2::password_hash::phc::PasswordHash;
use argon2::password_hash::{PasswordHasher, PasswordVerifier};
use argon2::{Algorithm, Argon2, Params, Version};
use std::fs;
use std::io;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// Explicit argon2id parameters (acceptance criterion 1: "explicit params, e.g. m=64 MiB, t=3,
/// p=1 — measure login time ≤ 0.5 s on this VPS"). Measured on this VPS (see crate tests /
/// `docs/STATUS.md`): hashing at these params takes ~0.2 s, well inside budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Argon2Params {
    pub m_cost_kib: u32,
    pub t_cost: u32,
    pub p_cost: u32,
}

impl Default for Argon2Params {
    fn default() -> Self {
        Self {
            m_cost_kib: 64 * 1024,
            t_cost: 3,
            p_cost: 1,
        }
    }
}

impl Argon2Params {
    fn context(&self) -> Result<Argon2<'static>, SecretsError> {
        let params =
            Params::new(self.m_cost_kib, self.t_cost, self.p_cost, None).map_err(SecretsError::Argon2Params)?;
        Ok(Argon2::new(Algorithm::Argon2id, Version::V0x13, params))
    }
}

/// Minimum length of a freshly generated owner password, in characters (acceptance criterion 1:
/// "≥ 20 chars, from OsRng").
pub const GENERATED_PASSWORD_MIN_LEN: usize = 20;

/// Paths of the three secrets files, all under `<data_dir>/secrets`.
#[derive(Debug, Clone)]
pub struct SecretsPaths {
    dir: PathBuf,
}

impl SecretsPaths {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            dir: data_dir.join("secrets"),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn auth_file(&self) -> PathBuf {
        self.dir.join("web-auth.toml")
    }

    pub fn session_key_file(&self) -> PathBuf {
        self.dir.join("web-session-key.toml")
    }

    pub fn password_file(&self) -> PathBuf {
        self.dir.join("web-password.txt")
    }

    /// Trusted-device records (review round 2, finding F8a). Not real TOML either — see this
    /// module's doc comment and [`encode_devices`]/[`decode_devices`].
    pub fn devices_file(&self) -> PathBuf {
        self.dir.join("web-devices.toml")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SecretsError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid argon2 parameters: {0}")]
    Argon2Params(argon2::Error),
    #[error("failed to hash password: {0}")]
    Hash(argon2::password_hash::Error),
    #[error("failed to parse stored password hash: {0}")]
    HashFormat(argon2::password_hash::phc::Error),
    #[error("{1}: {0}")]
    Parse(#[source] toml_kv::ParseError, PathBuf),
    #[error("{path}: missing or invalid field '{field}'")]
    MissingField { path: PathBuf, field: &'static str },
    #[error("{path}: invalid base64 key material: {source}")]
    BadKeyEncoding {
        path: PathBuf,
        #[source]
        source: base64::DecodeError,
    },
    #[error("{path}: key material has {actual} bytes, expected {expected}")]
    BadKeyLength {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
}

fn io_err(path: &Path, source: io::Error) -> SecretsError {
    SecretsError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Creates `<data_dir>/secrets` (and `data_dir` itself) if missing, and (re)asserts `0700` on the
/// secrets directory regardless of the umask that created it.
pub fn ensure_secrets_dir(paths: &SecretsPaths) -> Result<(), SecretsError> {
    fs::create_dir_all(paths.dir()).map_err(|e| io_err(paths.dir(), e))?;
    fs::set_permissions(paths.dir(), fs::Permissions::from_mode(0o700)).map_err(|e| io_err(paths.dir(), e))?;
    Ok(())
}

/// Checks that `path` has no group/other permission bits set; if it does, fixes it in place and
/// logs a warning rather than silently trusting a secrets path that other local users might be
/// able to read or (for the directory) list (review finding F6). This runs on every *load*, not
/// just on write, so it also catches a directory/file that predates this check, sits on a
/// misconfigured filesystem, or was loosened by an external tool.
pub(crate) fn enforce_private_permissions(path: &Path, expected_mode: u32) -> Result<(), SecretsError> {
    let metadata = fs::metadata(path).map_err(|e| io_err(path, e))?;
    let actual_mode = metadata.permissions().mode() & 0o777;
    if actual_mode & 0o077 != 0 {
        tracing::warn!(
            path = %path.display(),
            actual_mode = format!("{actual_mode:o}"),
            expected_mode = format!("{expected_mode:o}"),
            "secrets path had group/other permission bits set; fixing to private"
        );
        fs::set_permissions(path, fs::Permissions::from_mode(expected_mode)).map_err(|e| io_err(path, e))?;
    }
    Ok(())
}

fn tmp_sibling_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("secret");
    let tmp_name = format!(".{file_name}.tmp-{}", encode_b64(&random_bytes::<6>()));
    path.with_file_name(tmp_name)
}

/// Writes `contents` to `path` with exactly `0600` permissions, atomically (write to a sibling
/// temp file, then rename) so a crash never leaves a partially written or wrong-permission
/// secrets file, and no other local process ever observes a too-permissive window.
pub(crate) fn write_secret_file(path: &Path, contents: &str) -> Result<(), SecretsError> {
    use std::io::Write;

    let tmp_path = tmp_sibling_path(path);
    let write_result = (|| -> io::Result<()> {
        // `.mode(0o600)` (review finding F6) makes the file private from the moment it's
        // created, not after a separate `chmod`-equivalent call — `OpenOptions::open` with
        // `create_new` would otherwise briefly create it at the process's default mode (subject
        // to umask, potentially group/other-readable) before the follow-up `set_permissions`
        // narrowed it, a real (if narrow) window for another local user to read a partially
        // written secret. The explicit `set_permissions` below is kept as a defensive assertion
        // in case of an unusual umask.
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp_path)?;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp_path, path)?;
        Ok(())
    })();
    write_result.map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        io_err(path, e)
    })
}

pub(crate) fn read_to_string_checked(path: &Path) -> Result<String, SecretsError> {
    fs::read_to_string(path).map_err(|e| io_err(path, e))
}

// ---------------------------------------------------------------------------------------------
// Password hashing
// ---------------------------------------------------------------------------------------------

pub fn hash_password(password: &str, params: Argon2Params) -> Result<String, SecretsError> {
    let argon2 = params.context()?;
    let hash = argon2.hash_password(password.as_bytes()).map_err(SecretsError::Hash)?;
    Ok(hash.to_string())
}

/// Verifies `password` against a stored PHC hash string. Uses the params encoded in the hash
/// itself (as `argon2::Argon2::default().verify_password` does), so a config-file's recorded
/// `Argon2Params` are metadata for the audit trail, not what verification actually runs with —
/// exactly like the crate's own documented usage pattern.
pub fn verify_password(password: &str, hash_phc: &str) -> Result<bool, SecretsError> {
    let parsed = PasswordHash::new(hash_phc).map_err(SecretsError::HashFormat)?;
    match Argon2::default().verify_password(password.as_bytes(), &parsed) {
        Ok(()) => Ok(true),
        Err(argon2::password_hash::Error::PasswordInvalid) => Ok(false),
        Err(e) => Err(SecretsError::Hash(e)),
    }
}

/// Generates a random password of at least [`GENERATED_PASSWORD_MIN_LEN`] characters from the OS
/// CSPRNG. 24 random bytes base64url-encode to exactly 32 characters.
pub fn generate_password() -> String {
    let bytes = random_bytes::<24>();
    let password = encode_b64(&bytes);
    debug_assert!(password.len() >= GENERATED_PASSWORD_MIN_LEN);
    password
}

// ---------------------------------------------------------------------------------------------
// `web-auth.toml`: the password hash (never the plaintext)
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PasswordAuth {
    pub hash_phc: String,
    pub params: Argon2Params,
}

pub fn save_password_auth(paths: &SecretsPaths, auth: &PasswordAuth) -> Result<(), SecretsError> {
    ensure_secrets_dir(paths)?;
    let text = toml_kv::write_kv(
        &[
            "ddai-web owner password hash. Managed by `ddnet-ai web-passwd`.",
            "Do not edit by hand; do not commit (see CLAUDE.md).",
        ],
        &[
            ("algorithm", Value::Str("argon2id")),
            ("m_cost_kib", Value::Int(auth.params.m_cost_kib.into())),
            ("t_cost", Value::Int(auth.params.t_cost.into())),
            ("p_cost", Value::Int(auth.params.p_cost.into())),
            ("hash", Value::Str(&auth.hash_phc)),
        ],
    );
    write_secret_file(&paths.auth_file(), &text)
}

pub fn load_password_auth(paths: &SecretsPaths) -> Result<Option<PasswordAuth>, SecretsError> {
    let path = paths.auth_file();
    if !path.exists() {
        return Ok(None);
    }
    if paths.dir().exists() {
        enforce_private_permissions(paths.dir(), 0o700)?;
    }
    enforce_private_permissions(&path, 0o600)?;
    let text = read_to_string_checked(&path)?;
    let kv = toml_kv::parse_kv(&text).map_err(|e| SecretsError::Parse(e, path.clone()))?;
    let hash_phc = kv
        .str("hash")
        .ok_or_else(|| SecretsError::MissingField {
            path: path.clone(),
            field: "hash",
        })?
        .to_string();
    let field = |name: &'static str| -> Result<u32, SecretsError> {
        kv.int(name)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| SecretsError::MissingField {
                path: path.clone(),
                field: name,
            })
    };
    let params = Argon2Params {
        m_cost_kib: field("m_cost_kib")?,
        t_cost: field("t_cost")?,
        p_cost: field("p_cost")?,
    };
    Ok(Some(PasswordAuth { hash_phc, params }))
}

/// Generates a fresh password, hashes it, writes the plaintext to `web-password.txt` and the hash
/// to `web-auth.toml` (overwriting whatever was there — this is the owner-facing "(re)issue a
/// password" operation, i.e. `ddnet-ai web-passwd`).
///
/// The plaintext is written *before* the hash (review finding F6): if this function fails
/// partway through, the previous ordering could leave a brand-new hash active (so the old
/// password no longer works) with no plaintext for the new one ever written — a full owner
/// lockout. Writing the plaintext first means the worst case of a partial failure is instead "a
/// plaintext file exists for a password that isn't the active one yet", which is merely
/// confusing, not a lockout — the old password (if any) is still the one that verifies.
pub fn generate_and_store_password(
    paths: &SecretsPaths,
    params: Argon2Params,
) -> Result<GeneratedPassword, SecretsError> {
    ensure_secrets_dir(paths)?;
    let plaintext = generate_password();
    let hash_phc = hash_password(&plaintext, params)?;
    let auth = PasswordAuth { hash_phc, params };
    write_secret_file(&paths.password_file(), &format!("{plaintext}\n"))?;
    save_password_auth(paths, &auth)?;
    // Review round 2, finding F9a: wipe every persisted trusted-device record on every password
    // rotation. Not strictly the ONLY thing preventing a stale record from being trusted again
    // (that's the fingerprint-mismatch check in `auth::device::DeviceStore::is_trusted`, which
    // keeps working even if this write is somehow skipped/lost — see that module's doc comment),
    // but there is no reason to keep a superseded password's fingerprint sitting in this file for
    // up to its full TTL (90 days) once the password that ever could have matched it is gone for
    // good. Ordering relative to the two writes above is not security-load-bearing (unlike
    // plaintext-before-hash above): whichever order a crash lands in, the new hash is already
    // active and old records — cleared or not — can never match it again either way.
    save_devices(paths, &[])?;
    Ok(GeneratedPassword { plaintext, auth })
}

pub struct GeneratedPassword {
    pub plaintext: String,
    pub auth: PasswordAuth,
}

// ---------------------------------------------------------------------------------------------
// `web-session-key.toml`: HMAC key for signing the session cookie value
// ---------------------------------------------------------------------------------------------

pub const SESSION_KEY_LEN: usize = 32;

/// Loads the session-signing key, generating and persisting a fresh random one on first use
/// (acceptance criterion 2: "Session key material ... generated on first run into the secrets
/// dir").
pub fn load_or_create_session_key(paths: &SecretsPaths) -> Result<[u8; SESSION_KEY_LEN], SecretsError> {
    let path = paths.session_key_file();
    if path.exists() {
        if paths.dir().exists() {
            enforce_private_permissions(paths.dir(), 0o700)?;
        }
        enforce_private_permissions(&path, 0o600)?;
        let text = read_to_string_checked(&path)?;
        let kv = toml_kv::parse_kv(&text).map_err(|e| SecretsError::Parse(e, path.clone()))?;
        let b64 = kv.str("key_b64").ok_or_else(|| SecretsError::MissingField {
            path: path.clone(),
            field: "key_b64",
        })?;
        let bytes = decode_b64(b64).map_err(|source| SecretsError::BadKeyEncoding {
            path: path.clone(),
            source,
        })?;
        return <[u8; SESSION_KEY_LEN]>::try_from(bytes.as_slice()).map_err(|_| SecretsError::BadKeyLength {
            path: path.clone(),
            expected: SESSION_KEY_LEN,
            actual: bytes.len(),
        });
    }
    ensure_secrets_dir(paths)?;
    let key = random_bytes::<SESSION_KEY_LEN>();
    let text = toml_kv::write_kv(
        &[
            "ddai-web session-cookie signing key. Generated once on first run.",
            "Rotating this file invalidates every existing session. Do not commit.",
        ],
        &[("key_b64", Value::Str(&encode_b64(&key)))],
    );
    write_secret_file(&path, &text)?;
    Ok(key)
}

// ---------------------------------------------------------------------------------------------
// `web-devices.toml`: trusted devices that may skip the global login rate limit (review round 2,
// finding F8a — see `auth::device` for why this needs to survive a restart at all)
// ---------------------------------------------------------------------------------------------

/// One trusted-device record as persisted to disk. Keyed by [`PersistedDevice::id_hash`] — a
/// SHA-256 of the actual 256-bit device id, never the id itself, so that reading this file alone
/// is never enough to forge a valid device cookie (the signed cookie value also needs the
/// session-signing key, but hashing the id here is one more, independent layer: a leak of this
/// file specifically hands out nothing directly replayable). `auth::device` hashes an incoming
/// cookie's id the same way before every lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedDevice {
    pub id_hash: [u8; 32],
    /// The PHC hash string this device last confirmed — see `auth::device::DeviceRecord` for why
    /// comparing this against the *current* password hash on every check is the entire
    /// password-change revocation mechanism, unaffected by whether this device record came from
    /// memory or from this file.
    pub password_hash_fingerprint: String,
    pub expires_at_unix: u64,
    pub last_seen_unix: u64,
}

/// Renders `devices` as this file's line format: `<id_hash> <expires_at_unix> <last_seen_unix>
/// <password_hash_fingerprint>`, space-separated (the PHC hash format never contains a space, so
/// a plain split is unambiguous — see [`decode_devices`]).
fn encode_devices(devices: &[PersistedDevice]) -> String {
    let mut out = String::new();
    for line in [
        "ddai-web trusted devices (task 5.3, finding F8a). Managed automatically by the running",
        "server; do not edit by hand; do not commit (see CLAUDE.md).",
        "One line per device: <base64url sha256(device_id)> <expires_at_unix_s> <last_seen_unix_s> <password hash it was confirmed under>",
    ] {
        out.push_str("# ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    for device in devices {
        out.push_str(&encode_b64(&device.id_hash));
        out.push(' ');
        out.push_str(&device.expires_at_unix.to_string());
        out.push(' ');
        out.push_str(&device.last_seen_unix.to_string());
        out.push(' ');
        out.push_str(&device.password_hash_fingerprint);
        out.push('\n');
    }
    out
}

/// Parses [`encode_devices`]'s format. Deliberately lenient, unlike [`toml_kv::parse_kv`]: a
/// single malformed line (partial write racing a crash, disk corruption, hand-editing despite the
/// header comment's request not to) is logged and skipped rather than discarding every OTHER
/// device's trust along with it — this file is a convenience cache, not a security boundary in
/// itself (see `auth::device::DeviceStore::load_or_empty`'s doc comment).
fn decode_devices(text: &str) -> Vec<PersistedDevice> {
    let mut devices = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(4, ' ');
        let (Some(id_hash_b64), Some(expires_s), Some(last_seen_s), Some(hash_phc)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            tracing::warn!(
                line = idx + 1,
                "web-devices.toml: skipping malformed line (too few fields)"
            );
            continue;
        };
        let id_hash = match decode_b64(id_hash_b64).ok().and_then(|v| <[u8; 32]>::try_from(v).ok()) {
            Some(id_hash) => id_hash,
            None => {
                tracing::warn!(line = idx + 1, "web-devices.toml: skipping line with a bad id hash");
                continue;
            }
        };
        let (Ok(expires_at_unix), Ok(last_seen_unix)) = (expires_s.parse::<u64>(), last_seen_s.parse::<u64>()) else {
            tracing::warn!(line = idx + 1, "web-devices.toml: skipping line with a bad timestamp");
            continue;
        };
        devices.push(PersistedDevice {
            id_hash,
            password_hash_fingerprint: hash_phc.to_string(),
            expires_at_unix,
            last_seen_unix,
        });
    }
    devices
}

/// Overwrites `web-devices.toml` with exactly `devices` (not an append/merge — callers pass the
/// full current table, matching how small this is expected to stay for a single-owner panel).
pub fn save_devices(paths: &SecretsPaths, devices: &[PersistedDevice]) -> Result<(), SecretsError> {
    ensure_secrets_dir(paths)?;
    write_secret_file(&paths.devices_file(), &encode_devices(devices))
}

/// Loads trusted-device records. A missing file is `Ok(vec![])` (nothing has ever been trusted
/// yet), not an error — unlike the password hash or session key, there is nothing to fail
/// startup over here.
pub fn load_devices(paths: &SecretsPaths) -> Result<Vec<PersistedDevice>, SecretsError> {
    let path = paths.devices_file();
    if !path.exists() {
        return Ok(Vec::new());
    }
    if paths.dir().exists() {
        enforce_private_permissions(paths.dir(), 0o700)?;
    }
    enforce_private_permissions(&path, 0o600)?;
    let text = read_to_string_checked(&path)?;
    Ok(decode_devices(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perms(path: &Path) -> u32 {
        fs::metadata(path).expect("stat").permissions().mode() & 0o777
    }

    #[test]
    fn hash_and_verify_roundtrip() {
        let params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        let hash = hash_password("correct horse battery staple", params).expect("hash");
        assert!(hash.starts_with("$argon2id$"));
        assert!(verify_password("correct horse battery staple", &hash).expect("verify"));
        assert!(!verify_password("wrong password", &hash).expect("verify"));
    }

    #[test]
    fn verify_rejects_garbage_hash() {
        assert!(verify_password("x", "not a phc string").is_err());
    }

    #[test]
    fn generated_password_meets_length_and_charset() {
        let pw = generate_password();
        assert!(pw.len() >= GENERATED_PASSWORD_MIN_LEN, "password too short: {pw:?}");
        assert!(
            pw.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "password has unexpected characters: {pw:?}"
        );
    }

    #[test]
    fn generate_and_store_writes_0600_files_in_0700_dir_and_hash_verifies() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        let generated = generate_and_store_password(&paths, params).expect("generate");

        assert_eq!(perms(paths.dir()), 0o700);
        assert_eq!(perms(&paths.auth_file()), 0o600);
        assert_eq!(perms(&paths.password_file()), 0o600);

        let plaintext_on_disk = fs::read_to_string(paths.password_file()).expect("read password file");
        assert_eq!(plaintext_on_disk.trim_end(), generated.plaintext);

        let loaded = load_password_auth(&paths).expect("load").expect("some");
        assert_eq!(loaded.hash_phc, generated.auth.hash_phc);
        assert!(verify_password(&generated.plaintext, &loaded.hash_phc).expect("verify"));
        assert!(!verify_password("definitely not the password", &loaded.hash_phc).expect("verify"));
    }

    #[test]
    fn regenerating_overwrites_previous_password() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        let first = generate_and_store_password(&paths, params).expect("generate 1");
        let second = generate_and_store_password(&paths, params).expect("generate 2");
        assert_ne!(first.plaintext, second.plaintext);

        let loaded = load_password_auth(&paths).expect("load").expect("some");
        assert!(!verify_password(&first.plaintext, &loaded.hash_phc).expect("verify"));
        assert!(verify_password(&second.plaintext, &loaded.hash_phc).expect("verify"));
    }

    #[test]
    fn load_password_auth_missing_file_returns_none() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        assert!(load_password_auth(&paths).expect("load").is_none());
    }

    #[test]
    fn session_key_is_generated_once_and_persists() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let key1 = load_or_create_session_key(&paths).expect("create");
        assert_eq!(perms(&paths.session_key_file()), 0o600);
        let key2 = load_or_create_session_key(&paths).expect("load existing");
        assert_eq!(key1, key2, "session key must not change across loads");
    }

    #[test]
    fn session_key_file_is_0600() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let _ = load_or_create_session_key(&paths).expect("create");
        assert_eq!(perms(paths.dir()), 0o700);
    }

    #[test]
    fn loading_a_loosened_auth_file_fixes_its_permissions() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        generate_and_store_password(&paths, params).expect("generate");

        // Simulate an external tool (or a pre-fix version of this code) loosening the file.
        fs::set_permissions(paths.auth_file(), fs::Permissions::from_mode(0o644)).expect("loosen perms");
        assert_eq!(perms(&paths.auth_file()), 0o644);

        // Loading it must notice and fix it back to private, not just silently trust it.
        load_password_auth(&paths).expect("load").expect("some");
        assert_eq!(
            perms(&paths.auth_file()),
            0o600,
            "load should have re-tightened permissions"
        );
    }

    #[test]
    fn loading_a_loosened_session_key_file_fixes_its_permissions() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        load_or_create_session_key(&paths).expect("create");

        fs::set_permissions(paths.session_key_file(), fs::Permissions::from_mode(0o640)).expect("loosen perms");
        assert_eq!(perms(&paths.session_key_file()), 0o640);

        load_or_create_session_key(&paths).expect("load existing");
        assert_eq!(
            perms(&paths.session_key_file()),
            0o600,
            "load should have re-tightened permissions"
        );
    }

    #[test]
    fn loosened_secrets_dir_is_fixed_on_load() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        generate_and_store_password(&paths, params).expect("generate");

        fs::set_permissions(paths.dir(), fs::Permissions::from_mode(0o750)).expect("loosen dir perms");
        assert_eq!(perms(paths.dir()), 0o750);

        load_password_auth(&paths).expect("load").expect("some");
        assert_eq!(
            perms(paths.dir()),
            0o700,
            "load should have re-tightened the directory too"
        );
    }

    #[test]
    fn plaintext_is_written_before_the_hash_so_a_later_failure_still_leaves_it_readable() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        ensure_secrets_dir(&paths).expect("ensure dir");
        // Sabotage the hash file's path so writing web-auth.toml fails (rename onto an existing
        // directory fails with ENOTEMPTY/EISDIR), while web-password.txt — a different path — can
        // still be written normally. This simulates a mid-operation failure to check the write
        // ordering without needing a genuine disk-full/permission fault.
        fs::create_dir(paths.auth_file()).expect("pre-create web-auth.toml as a directory");

        let params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        let result = generate_and_store_password(&paths, params);
        assert!(result.is_err(), "expected the hash write to fail");

        let plaintext_on_disk = fs::read_to_string(paths.password_file()).expect("password file should still exist");
        assert!(
            !plaintext_on_disk.trim().is_empty(),
            "plaintext must have been written before the failing hash write"
        );
    }

    // -----------------------------------------------------------------------------------------
    // web-devices.toml (review round 2, finding F8a)
    // -----------------------------------------------------------------------------------------

    fn device(seed: u8) -> PersistedDevice {
        let mut id_hash = [0u8; 32];
        id_hash[0] = seed;
        PersistedDevice {
            id_hash,
            password_hash_fingerprint: format!("$argon2id$v=19$m=8,t=1,p=1$AAAA$hash-{seed}"),
            expires_at_unix: 1_000_000 + seed as u64,
            last_seen_unix: 900_000 + seed as u64,
        }
    }

    #[test]
    fn devices_roundtrip_through_encode_decode() {
        let devices = vec![device(1), device(2), device(3)];
        let decoded = decode_devices(&encode_devices(&devices));
        assert_eq!(decoded, devices);
    }

    #[test]
    fn decode_devices_skips_malformed_lines_but_keeps_the_rest() {
        let good = device(7);
        let text = format!(
            "# comment\n\ntoo few fields\nnot-base64!! 1 2 $argon2id$whatever\n{} 1 2 $argon2id$whatever\n{} {} {} {}\n",
            // A valid base64url string, but decoding to the wrong length for an id hash (5 bytes,
            // not 32) — exercises the `<[u8; 32]>::try_from` failure branch specifically, distinct
            // from `not-base64!!` above (which fails at `decode_b64` itself).
            encode_b64(&[9u8; 5]),
            encode_b64(&good.id_hash),
            good.expires_at_unix,
            good.last_seen_unix,
            good.password_hash_fingerprint,
        );
        let decoded = decode_devices(&text);
        assert_eq!(
            decoded,
            vec![good],
            "only the one well-formed line should survive: {text}"
        );
    }

    #[test]
    fn decode_devices_ignores_comments_and_blank_lines() {
        assert_eq!(decode_devices("# just a header\n\n\n").len(), 0);
        assert_eq!(decode_devices("").len(), 0);
    }

    #[test]
    fn save_and_load_devices_roundtrip_with_0600_0700() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let devices = vec![device(1), device(2)];
        save_devices(&paths, &devices).expect("save");

        assert_eq!(perms(paths.dir()), 0o700);
        assert_eq!(perms(&paths.devices_file()), 0o600);

        let loaded = load_devices(&paths).expect("load");
        assert_eq!(loaded, devices);
    }

    #[test]
    fn load_devices_missing_file_returns_empty_not_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        assert_eq!(load_devices(&paths).expect("load"), Vec::new());
    }

    #[test]
    fn saving_overwrites_the_previous_table_entirely() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        save_devices(&paths, &[device(1), device(2)]).expect("save 1");
        save_devices(&paths, &[device(3)]).expect("save 2");
        assert_eq!(load_devices(&paths).expect("load"), vec![device(3)]);
    }

    #[test]
    fn the_raw_device_id_never_appears_in_the_persisted_file_only_its_hash() {
        // Security property finding F8a asks for directly: the file stores a hash of the device
        // id, never the id itself, so reading the file alone never hands out a directly-replayable
        // value.
        let raw_id: [u8; 32] = random_bytes();
        let id_hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(raw_id);
            let out: [u8; 32] = hasher.finalize().into();
            out
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        save_devices(
            &paths,
            &[PersistedDevice {
                id_hash,
                password_hash_fingerprint: "$argon2id$v=19$m=8,t=1,p=1$AAAA$hash".to_string(),
                expires_at_unix: 1,
                last_seen_unix: 1,
            }],
        )
        .expect("save");
        let raw_contents = fs::read_to_string(paths.devices_file()).expect("read raw file");
        assert!(
            !raw_contents.contains(&encode_b64(&raw_id)),
            "the raw device id must never appear in the persisted file"
        );
        assert!(
            raw_contents.contains(&encode_b64(&id_hash)),
            "the device id's hash should be the thing actually stored"
        );
    }
}
