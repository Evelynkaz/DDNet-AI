//! Persistent freeze memory (`FreezeMemory.save`/`load`, `src/plan/memory.ts:72-129`).
//!
//! The in-memory model (`note`/`notePass`/`safety`/`risk`, `SPREAD`) is `ddai_planner::memory`
//! (task 3.2); this adds the file format and its semantics: **decay on every save** (all counts
//! `* 0.97`, cells below 0.01 and passes below 0.05 are dropped), sparse `idx/val`, `pidx/pval`
//! arrays, a size mismatch loads as empty. **CHANGE against TS:** the file is keyed by the map's
//! **sha256** (two versions of one map must not share a memory; the TS keyed by name without CRC,
//! `orig-bot.md` §13.7) and lives under `~/aiddnet/data/bot/memory/` — never in git. Writes are
//! atomic (`.tmp`, rename) and the directory is owner-only.

use ddai_planner::memory::FreezeMemory;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// `DECAY` (`memory.ts:8`).
pub const DECAY: f32 = 0.97;

/// Save every this many noted freezes (`bot.ts`: `noted % 20 === 0`).
pub const SAVE_EVERY_NOTES: i64 = 20;

#[derive(Serialize, Deserialize)]
struct File {
    width: i32,
    height: i32,
    events: i64,
    idx: Vec<usize>,
    val: Vec<f64>,
    #[serde(default)]
    pidx: Vec<usize>,
    #[serde(default)]
    pval: Vec<f64>,
}

/// `<data dir>/bot/memory` (`~/aiddnet/data/bot/memory` on Linux, see `ddai_os::dirs`); `None` when there is no (absolute) home
/// directory (never a relative path: it could land inside a repository checkout).
pub fn default_memory_dir() -> Option<PathBuf> {
    ddai_os::dirs::data_root()
        .map(|d| d.join("bot/memory"))
        .filter(|p| p.is_absolute())
}

/// The memory file of the map with this sha256 (lowercase hex).
pub fn memory_path(dir: &Path, map_sha256_hex: &str) -> PathBuf {
    let safe: String = map_sha256_hex.chars().filter(char::is_ascii_alphanumeric).collect();
    dir.join(format!("{safe}.json"))
}

/// `Number(x.toFixed(digits))`.
fn fixed(v: f32, digits: usize) -> f64 {
    format!("{:.*}", digits, f64::from(v)).parse().unwrap_or(0.0)
}

/// `save(file)`: decays the memory in place, then writes it (best effort: any error is returned, the
/// caller may ignore it — the TS swallowed errors).
pub fn save(mem: &mut FreezeMemory, file: &Path) -> std::io::Result<()> {
    let (mut idx, mut val, mut pidx, mut pval) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (width, height, events) = (mem.width, mem.height, mem.noted());
    {
        let (cells, passes) = mem.grids_mut();
        for i in 0..cells.len() {
            cells[i] = (f64::from(cells[i]) * f64::from(DECAY)) as f32;
            passes[i] = (f64::from(passes[i]) * f64::from(DECAY)) as f32;
            if cells[i] < 0.01 {
                cells[i] = 0.0;
            } else {
                idx.push(i);
                val.push(fixed(cells[i], 3));
            }
            if passes[i] < 0.05 {
                passes[i] = 0.0;
            } else {
                pidx.push(i);
                pval.push(fixed(passes[i], 2));
            }
        }
    }
    if let Some(dir) = file.parent() {
        create_private_dir(dir)?;
    }
    let tmp = file.with_extension("json.tmp");
    let body = serde_json::to_vec(&File {
        width,
        height,
        events,
        idx,
        val,
        pidx,
        pval,
    })
    .map_err(std::io::Error::other)?;
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, file)
}

/// The memory directory is owner-only: mode `0700` on Unix, an ACL for the current user on Windows (`ddai_os::private`; a failure to set
/// the ACL is not fatal, the folder is inside the user's profile).
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    ddai_os::private::create_dir_all_restricted(dir).map(drop)
}

/// `FreezeMemory.load(file, width, height)`: a missing, unreadable or mismatching file is an empty memory.
pub fn load(file: &Path, width: i32, height: i32) -> FreezeMemory {
    let empty = || FreezeMemory::new(width, height);
    let Ok(bytes) = std::fs::read(file) else { return empty() };
    let Ok(raw) = serde_json::from_slice::<File>(&bytes) else {
        return empty();
    };
    if raw.width != width || raw.height != height || raw.idx.len() != raw.val.len() {
        return empty();
    }
    let n = (width * height).max(0) as usize;
    let mut cells = vec![0f32; n];
    for (k, &i) in raw.idx.iter().enumerate() {
        if i < n {
            cells[i] = raw.val[k] as f32;
        }
    }
    let mut passes = vec![0f32; n];
    for (k, &i) in raw.pidx.iter().enumerate() {
        if i < n {
            passes[i] = raw.pval.get(k).copied().unwrap_or(0.0) as f32;
        }
    }
    FreezeMemory::from_parts(width, height, cells, passes, raw.events).unwrap_or_else(empty)
}

/// The memory of one map in play: loaded at map load, noted on freezes and safe passes, saved every
/// [`SAVE_EVERY_NOTES`] freezes, at a map change and at stop.
pub struct MemoryStore {
    pub mem: FreezeMemory,
    path: PathBuf,
    dirty: bool,
}

impl MemoryStore {
    pub fn open(dir: &Path, map_sha256_hex: &str, width: i32, height: i32) -> MemoryStore {
        let path = memory_path(dir, map_sha256_hex);
        MemoryStore {
            mem: load(&path, width, height),
            path,
            dirty: false,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// We froze at `(x, y)` (px): note it and save every 20th.
    pub fn note_freeze(&mut self, x: f64, y: f64) {
        self.mem.note(x, y);
        self.dirty = true;
        if self.mem.noted() % SAVE_EVERY_NOTES == 0 {
            self.save();
        }
    }

    /// We entered a new tile free (`notePass`).
    pub fn note_pass(&mut self, x: f64, y: f64) {
        self.mem.note_pass(x, y);
        self.dirty = true;
    }

    /// Saves (decaying) when something changed since the last save.
    pub fn save(&mut self) {
        if !self.dirty {
            return;
        }
        let _ = save(&mut self.mem, &self.path);
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_save_decays_every_count_by_097_and_drops_the_tiny_ones_and_load_restores_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let file = memory_path(dir.path(), "ABCDEF0123");
        let mut m = FreezeMemory::new(10, 10);
        m.note(5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
        m.note_pass(2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0);
        save(&mut m, &file).unwrap();
        // in memory: decayed
        let centre = 5 * 10 + 5;
        assert!((m.cells()[centre] - 0.97).abs() < 1e-6);
        assert!((m.cells()[5 * 10 + 6] - 0.4 * 0.97).abs() < 1e-6, "SPREAD, decayed");
        let loaded = load(&file, 10, 10);
        assert_eq!(loaded.noted(), 1);
        assert!((f64::from(loaded.cells()[centre]) - 0.97).abs() < 1e-3, "3 decimals");
        assert!(
            (f64::from(loaded.passes()[2 * 10 + 2]) - 0.97).abs() < 1e-2,
            "2 decimals"
        );
        // a second save decays again; very small values are dropped
        let mut tiny = FreezeMemory::new(4, 4);
        tiny.note_pass(16.0, 16.0);
        let (_, passes) = tiny.grids_mut();
        passes[0] = 0.051; // 0.051 * 0.97 = 0.0495 < 0.05: dropped
        save(&mut tiny, &file).unwrap();
        assert_eq!(load(&file, 4, 4).passes()[0], 0.0);
    }

    #[test]
    fn a_size_mismatch_or_a_missing_or_broken_file_loads_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let file = memory_path(dir.path(), "00ff");
        assert_eq!(load(&file, 5, 5).noted(), 0, "missing");
        let mut m = FreezeMemory::new(5, 5);
        m.note(16.0, 16.0);
        save(&mut m, &file).unwrap();
        assert_eq!(load(&file, 6, 5).noted(), 0, "another size");
        assert_eq!(load(&file, 5, 5).noted(), 1);
        std::fs::write(&file, b"not json").unwrap();
        assert_eq!(load(&file, 5, 5).noted(), 0, "broken");
    }

    #[test]
    fn the_file_is_keyed_by_the_sha256_and_the_directory_is_owner_only() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("bot/memory");
        let a = memory_path(&sub, "aa11");
        let b = memory_path(&sub, "bb22");
        assert_ne!(a, b, "two versions of a map never share a memory");
        assert_eq!(
            memory_path(&sub, "../../etc/passwd").file_name().unwrap(),
            "etcpasswd.json",
            "no path tricks"
        );
        let mut m = FreezeMemory::new(3, 3);
        m.note(16.0, 16.0);
        save(&mut m, &a).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&sub).unwrap().permissions().mode() & 0o777, 0o700);
        }
    }

    #[test]
    fn the_store_saves_every_twentieth_freeze() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = MemoryStore::open(dir.path(), "cafe", 20, 20);
        for i in 0..19 {
            s.note_freeze(f64::from(i) * 32.0 + 16.0, 16.0);
        }
        assert!(!s.path().exists(), "19 freezes: not yet");
        s.note_freeze(16.0, 48.0);
        assert!(s.path().exists(), "the 20th saves");
        let again = MemoryStore::open(dir.path(), "cafe", 20, 20);
        assert_eq!(again.mem.noted(), 20);
    }
}
