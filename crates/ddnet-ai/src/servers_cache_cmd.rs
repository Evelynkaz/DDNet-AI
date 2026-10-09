//! `ddnet-ai servers-cache` (task 5.12, D-099): fetches the DDNet master list and writes the **bounded, strictly parsed cache** the
//! site's «Серверы» tab reads (`<data-dir>/servers/master.json`, `refresh.json`).
//!
//! This is the only code that talks to the master servers, and it runs in a unit of its own (`ddnet-ai-servers.service`, started by a
//! path unit when the site asks) so that the web process keeps `IPAddressDeny=any`: the unit may reach the internet but sees
//! nothing of the owner's data except the one directory it writes, and has no capability. What it fetches is untrusted:
//!
//! - HTTPS only (the four fixed masters, `https://masterN.ddnet.org/ddnet/15/servers.json`), no redirects, 8 s per request;
//! - at most [`MAX_MASTER_BYTES`] are read, whatever `Content-Length` says;
//! - the list is parsed by `ddai_client::server_list::parse_master` (no player names kept), reduced to public IPv4 rows with clean,
//!   length-capped text, and **parsed again** by the same strict reader the web uses before it is written;
//! - nothing is fetched when the cache is younger than [`MIN_REFRESH_SECS`] (one list a minute at most, however it is triggered).
//!
//! A failed run leaves the old cache in place and says why in `refresh.json` as a fixed code. Never connects to any game server.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::Args;
use ddai_client::server_list::{
    CACHE_FILE, CACHE_VERSION, MASTERS, MAX_MASTER_BYTES, MIN_REFRESH_SECS, MasterCache, REFRESH_FILE, RefreshStatus,
    parse_master,
};
use ddai_web::launch::{read_regular_nofollow, unix_now, write_atomic};

#[derive(Debug, Args)]
pub struct ServersCacheArgs {
    /// Base data directory. Default `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// The directory `master.json` and `refresh.json` are written into. Default `<data-dir>/servers`.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Test only: fetch this plain-HTTP **loopback** URL (a local fixture server) instead of the four masters.
    #[arg(long, hide = true)]
    pub fixture_url: Option<String>,
}

/// What one run did, for the terminal and the tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The cache is younger than the minimum: nothing was fetched.
    Fresh,
    /// A new cache was written (from master `n`, `count` servers).
    Written { master: u8, count: usize },
    /// Nothing usable came back (`no_master`, `bad_list`) or the file could not be written (`write_failed`); the old cache stays.
    Failed(&'static str),
}

/// Why one fetch failed (never shown with the URL or the error text: only the code reaches `refresh.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchError {
    /// No answer, an HTTP error, a redirect, a body over the size cap, bytes that are not UTF-8.
    Unavailable,
}

/// The cache as it is on disk now, if it is valid.
fn current(out: &Path) -> Option<MasterCache> {
    let bytes = read_regular_nofollow(&out.join(CACHE_FILE), ddai_client::server_list::MAX_CACHE_BYTES).ok()?;
    MasterCache::parse(&bytes).ok()
}

fn write_refresh(out: &Path, now: u64, outcome: Result<(), &'static str>) {
    let status = RefreshStatus {
        v: CACHE_VERSION,
        at: now,
        ok: outcome.is_ok(),
        reason: outcome.err().map(str::to_string),
    };
    if let Ok(bytes) = serde_json::to_vec(&status)
        && let Err(e) = write_atomic(out, REFRESH_FILE, &bytes, 0o644)
    {
        eprintln!("servers-cache: could not write {REFRESH_FILE}: {e}");
    }
}

/// One refresh. `fetch` returns the body of a master's list (already size-capped by the caller's reader); `masters` are the URLs to
/// try in order. Pure over the fetcher, so the whole policy is tested without a network.
pub fn refresh(
    out: &Path,
    now: u64,
    masters: &[String],
    fetch: &dyn Fn(&str) -> Result<String, FetchError>,
) -> Outcome {
    if let Some(c) = current(out)
        && now.saturating_sub(c.fetched_at) < MIN_REFRESH_SECS
    {
        return Outcome::Fresh;
    }
    let mut reason = "no_master";
    for (i, url) in masters.iter().enumerate() {
        let Ok(body) = fetch(url) else {
            continue;
        };
        let rows = match parse_master(&body) {
            Ok(rows) if !rows.is_empty() => rows,
            _ => {
                reason = "bad_list";
                continue;
            }
        };
        let master = u8::try_from(i + 1).unwrap_or(4).clamp(1, 4);
        let cache = MasterCache::from_rows(&rows, now, master);
        if cache.servers.is_empty() {
            reason = "bad_list";
            continue;
        }
        let Ok(bytes) = cache.to_bytes() else {
            reason = "bad_list";
            continue;
        };
        if let Err(e) = std::fs::create_dir_all(out).and_then(|()| write_atomic(out, CACHE_FILE, &bytes, 0o644)) {
            eprintln!("servers-cache: could not write {CACHE_FILE}: {e}");
            write_refresh(out, now, Err("write_failed"));
            return Outcome::Failed("write_failed");
        }
        write_refresh(out, now, Ok(()));
        return Outcome::Written {
            master,
            count: cache.servers.len(),
        };
    }
    write_refresh(out, now, Err(reason));
    Outcome::Failed(reason)
}

/// Reads at most `MAX_MASTER_BYTES` of the answer; more is a failure (a hostile or broken master must not fill the memory).
fn read_bounded(mut reader: impl Read) -> Result<String, FetchError> {
    let mut bytes = Vec::new();
    (&mut reader)
        .take(MAX_MASTER_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| FetchError::Unavailable)?;
    if bytes.len() > MAX_MASTER_BYTES {
        return Err(FetchError::Unavailable);
    }
    String::from_utf8(bytes).map_err(|_| FetchError::Unavailable)
}

/// Whether `url` may be fetched: one of the fixed masters, or (test only) a plain-HTTP loopback fixture.
fn url_allowed(url: &str, fixture: bool) -> bool {
    if fixture {
        // Parsed, not prefix-matched: `http://127.0.0.1:80@evil.example/` has the host `evil.example`.
        return reqwest::Url::parse(url).is_ok_and(|u| {
            u.scheme() == "http"
                && u.host_str() == Some("127.0.0.1")
                && u.username().is_empty()
                && u.password().is_none()
        });
    }
    MASTERS.contains(&url)
}

fn http_fetch(url: &str, fixture: bool) -> Result<String, FetchError> {
    if !url_allowed(url, fixture) {
        return Err(FetchError::Unavailable);
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(8))
        .connect_timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::none())
        .https_only(!fixture)
        .user_agent("ddnet-ai")
        .build()
        .map_err(|_| FetchError::Unavailable)?;
    let response = client.get(url).send().map_err(|_| FetchError::Unavailable)?;
    if !response.status().is_success() {
        return Err(FetchError::Unavailable);
    }
    if response.content_length().is_some_and(|n| n > MAX_MASTER_BYTES as u64) {
        return Err(FetchError::Unavailable);
    }
    read_bounded(response)
}

pub fn run(args: ServersCacheArgs) -> ExitCode {
    let data_dir = args.data_dir.clone().unwrap_or_else(|| match std::env::var_os("HOME") {
        Some(h) if !h.is_empty() => PathBuf::from(h).join("aiddnet").join("data"),
        _ => PathBuf::from("data"),
    });
    let out = args.out.clone().unwrap_or_else(|| data_dir.join("servers"));
    let fixture = args.fixture_url.is_some();
    let masters: Vec<String> = match &args.fixture_url {
        Some(u) => vec![u.clone()],
        None => MASTERS.iter().map(|m| (*m).to_string()).collect(),
    };
    let outcome = refresh(&out, unix_now(), &masters, &|url| http_fetch(url, fixture));
    match outcome {
        Outcome::Fresh => {
            println!("servers-cache: the cache is younger than {MIN_REFRESH_SECS} s, nothing fetched");
            ExitCode::SUCCESS
        }
        Outcome::Written { master, count } => {
            println!("servers-cache: wrote {count} servers (master {master})");
            ExitCode::SUCCESS
        }
        Outcome::Failed(code) => {
            eprintln!("servers-cache: failed ({code}); the old cache, if any, stays");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    const FIXTURE: &str = include_str!("../../ddai-client/tests/fixtures/master-servers.json");
    const GOOD: &str = r#"{"servers":[{"addresses":["tw-0.6+udp://93.184.216.35:8308"],"location":"eu:it","info":{"name":"S","map":{"name":"Copy Love Box"},"game_type":"Block","passworded":false,"max_clients":64,"clients":[{"name":"SECRETNICK","is_player":true}]}}]}"#;

    fn masters() -> Vec<String> {
        MASTERS.iter().map(|m| (*m).to_string()).collect()
    }

    #[test]
    fn a_good_list_becomes_a_strict_cache_and_a_note() {
        let dir = tempfile::tempdir().unwrap();
        let out = refresh(dir.path(), 1_000_000, &masters(), &|_| Ok(FIXTURE.to_string()));
        assert!(
            matches!(out, Outcome::Written { master: 1, count } if count > 0),
            "{out:?}"
        );
        let bytes = std::fs::read(dir.path().join(CACHE_FILE)).unwrap();
        let cache = MasterCache::parse(&bytes).unwrap();
        assert_eq!((cache.fetched_at, cache.master), (1_000_000, 1));
        let note: RefreshStatus =
            serde_json::from_slice(&std::fs::read(dir.path().join(REFRESH_FILE)).unwrap()).unwrap();
        assert!(note.ok && note.reason.is_none());
        for f in [CACHE_FILE, REFRESH_FILE] {
            let mode =
                std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(dir.path().join(f)).unwrap().permissions());
            assert_eq!(mode & 0o777, 0o644, "{f}");
        }
        // Player names never reach the cache.
        let out = refresh(dir.path(), 2_000_000, &masters(), &|_| Ok(GOOD.to_string()));
        assert!(matches!(out, Outcome::Written { count: 1, .. }), "{out:?}");
        assert!(
            !std::fs::read_to_string(dir.path().join(CACHE_FILE))
                .unwrap()
                .contains("SECRETNICK")
        );
    }

    #[test]
    fn a_cache_younger_than_a_minute_is_not_fetched_again() {
        let dir = tempfile::tempdir().unwrap();
        let calls = RefCell::new(0u32);
        let fetch = |_: &str| {
            *calls.borrow_mut() += 1;
            Ok(GOOD.to_string())
        };
        assert!(matches!(
            refresh(dir.path(), 1000, &masters(), &fetch),
            Outcome::Written { .. }
        ));
        assert_eq!(
            refresh(dir.path(), 1000 + MIN_REFRESH_SECS - 1, &masters(), &fetch),
            Outcome::Fresh
        );
        assert_eq!(*calls.borrow(), 1);
        assert!(matches!(
            refresh(dir.path(), 1000 + MIN_REFRESH_SECS, &masters(), &fetch),
            Outcome::Written { .. }
        ));
        assert_eq!(*calls.borrow(), 2);
    }

    #[test]
    fn the_next_master_is_tried_and_a_total_failure_keeps_the_old_cache() {
        let dir = tempfile::tempdir().unwrap();
        // Master 1 is down, master 2 answers.
        let out = refresh(dir.path(), 1000, &masters(), &|u| {
            if u == MASTERS[0] {
                Err(FetchError::Unavailable)
            } else {
                Ok(GOOD.to_string())
            }
        });
        assert!(matches!(out, Outcome::Written { master: 2, .. }), "{out:?}");
        let before = std::fs::read(dir.path().join(CACHE_FILE)).unwrap();
        // Everything down: the old cache stays, the note says why.
        let out = refresh(dir.path(), 5000, &masters(), &|_| Err(FetchError::Unavailable));
        assert_eq!(out, Outcome::Failed("no_master"));
        assert_eq!(std::fs::read(dir.path().join(CACHE_FILE)).unwrap(), before);
        let note: RefreshStatus =
            serde_json::from_slice(&std::fs::read(dir.path().join(REFRESH_FILE)).unwrap()).unwrap();
        assert_eq!(
            (note.ok, note.at, note.reason.as_deref()),
            (false, 5000, Some("no_master"))
        );
    }

    #[test]
    fn garbage_empty_and_private_only_lists_are_never_written() {
        for body in [
            "not json",
            "",
            "{}",
            r#"{"servers":[]}"#,
            r#"{"servers":[{"addresses":["udp://10.0.0.1:8303"]},{"addresses":["udp://127.0.0.1:8303"]}]}"#,
            r#"{"servers":[{"addresses":["udp://203.0.113.5:8303"]}]}"#,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let out = refresh(dir.path(), 1000, &masters(), &|_| Ok(body.to_string()));
            assert_eq!(out, Outcome::Failed("bad_list"), "{body:?}");
            assert!(!dir.path().join(CACHE_FILE).exists(), "{body:?}");
        }
    }

    #[test]
    fn the_write_goes_through_a_temp_file_and_never_follows_a_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        std::fs::write(&victim, "keep").unwrap();
        std::os::unix::fs::symlink(&victim, dir.path().join(CACHE_FILE)).unwrap();
        let out = refresh(dir.path(), 1000, &masters(), &|_| Ok(GOOD.to_string()));
        assert!(matches!(out, Outcome::Written { .. }), "{out:?}");
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep");
        assert!(MasterCache::parse(&std::fs::read(dir.path().join(CACHE_FILE)).unwrap()).is_ok());
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().all(|n| !n.contains(".tmp-")), "{names:?}");
    }

    #[test]
    fn the_body_is_read_bounded_and_only_the_fixed_masters_are_allowed() {
        let ok = vec![b'a'; MAX_MASTER_BYTES];
        assert_eq!(read_bounded(&ok[..]).unwrap().len(), MAX_MASTER_BYTES);
        let big = vec![b'a'; MAX_MASTER_BYTES + 1];
        assert_eq!(read_bounded(&big[..]), Err(FetchError::Unavailable));
        assert_eq!(read_bounded(&b"\xff\xfe"[..]), Err(FetchError::Unavailable));
        for m in MASTERS {
            assert!(url_allowed(m, false));
        }
        for bad in [
            "http://master1.ddnet.org/ddnet/15/servers.json",
            "https://evil.example/ddnet/15/servers.json",
            "https://master1.ddnet.org/ddnet/15/servers.json?x=1",
            "file:///etc/passwd",
            "http://127.0.0.1:1/x",
        ] {
            assert!(!url_allowed(bad, false), "{bad}");
        }
        // The fixture switch accepts loopback http only.
        assert!(url_allowed("http://127.0.0.1:8080/x", true));
        assert!(!url_allowed("https://master1.ddnet.org/ddnet/15/servers.json", true));
        assert!(!url_allowed("http://10.0.0.1:8080/x", true));
        assert!(!url_allowed("http://127.0.0.1:80@evil.example/x", true));
        assert!(!url_allowed("http://127.0.0.1.evil.example:80/x", true));
        assert!(!url_allowed("http://localhost:80/x", true));
    }
}
