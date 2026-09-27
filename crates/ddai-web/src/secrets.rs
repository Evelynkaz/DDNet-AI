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
fn enforce_private_permissions(path: &Path, expected_mode: u32) -> Result<(), SecretsError> {
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
fn write_secret_file(path: &Path, contents: &str) -> Result<(), SecretsError> {
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

fn read_to_string_checked(path: &Path) -> Result<String, SecretsError> {
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
}
