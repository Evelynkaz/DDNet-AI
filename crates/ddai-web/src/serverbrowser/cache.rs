//! Reading what the master-list fetch wrote (`<data-dir>/servers/master.json`, `refresh.json`). Read-only: the web has no network
//! and never writes there. Everything is checked again with the same strict parser the fetch wrote it with
//! ([`ddai_client::server_list::MasterCache::parse`]).

use std::path::Path;

use ddai_client::server_list::{CACHE_FILE, MAX_CACHE_BYTES, MasterCache, REFRESH_FILE, RefreshStatus};

use crate::launch::{ReadError, read_regular_nofollow};

/// Why the cache cannot be shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheProblem {
    /// Nothing was fetched yet (or the directory does not exist).
    Missing,
    /// The file is there but not a cache this build trusts (too large, malformed, out of bounds, a symlink).
    Invalid,
}

/// The cache, or why not.
pub fn read_master(servers_dir: &Path) -> Result<MasterCache, CacheProblem> {
    match read_regular_nofollow(&servers_dir.join(CACHE_FILE), MAX_CACHE_BYTES) {
        Ok(bytes) => MasterCache::parse(&bytes).map_err(|_| CacheProblem::Invalid),
        Err(ReadError::Missing) => Err(CacheProblem::Missing),
        Err(_) => Err(CacheProblem::Invalid),
    }
}

/// What the last fetch run did, if it left a readable note.
pub fn read_refresh(servers_dir: &Path) -> Option<RefreshStatus> {
    let bytes = read_regular_nofollow(&servers_dir.join(REFRESH_FILE), 4096).ok()?;
    let status: RefreshStatus = serde_json::from_slice(&bytes).ok()?;
    (status.v == 1).then_some(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_client::server_list::{ServerRow, parse_master};

    #[test]
    fn a_missing_a_broken_and_a_good_cache() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_master(dir.path()).unwrap_err(), CacheProblem::Missing);
        assert_eq!(
            read_master(&dir.path().join("nope")).unwrap_err(),
            CacheProblem::Missing
        );
        std::fs::write(dir.path().join(CACHE_FILE), b"{\"v\":1}").unwrap();
        assert_eq!(read_master(dir.path()).unwrap_err(), CacheProblem::Invalid);
        let rows: Vec<ServerRow> = parse_master(
            r#"{"servers":[{"addresses":["tw-0.6+udp://93.184.216.35:8308"],"location":"eu:it","info":{"name":"S","map":{"name":"M"},"clients":[]}}]}"#,
        )
        .unwrap();
        let cache = MasterCache::from_rows(&rows, 77, 1);
        std::fs::write(dir.path().join(CACHE_FILE), cache.to_bytes().unwrap()).unwrap();
        assert_eq!(read_master(dir.path()).unwrap(), cache);
        // A symlink in place of the file is not followed.
        std::fs::rename(dir.path().join(CACHE_FILE), dir.path().join("real.json")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("real.json"), dir.path().join(CACHE_FILE)).unwrap();
        assert_eq!(read_master(dir.path()).unwrap_err(), CacheProblem::Invalid);
    }

    #[test]
    fn the_refresh_note_is_read_when_valid() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_refresh(dir.path()).is_none());
        std::fs::write(
            dir.path().join(REFRESH_FILE),
            br#"{"v":1,"at":5,"ok":false,"reason":"no_master"}"#,
        )
        .unwrap();
        let r = read_refresh(dir.path()).unwrap();
        assert_eq!((r.at, r.ok, r.reason.as_deref()), (5, false, Some("no_master")));
        std::fs::write(dir.path().join(REFRESH_FILE), br#"{"v":2,"at":5,"ok":true}"#).unwrap();
        assert!(read_refresh(dir.path()).is_none());
        std::fs::write(dir.path().join(REFRESH_FILE), b"garbage").unwrap();
        assert!(read_refresh(dir.path()).is_none());
    }
}
