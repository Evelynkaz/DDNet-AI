//! The clip directory (`~/aiddnet/data/bot/clips/`, never in git): names, the autoclip decision and
//! pruning — `maybeClip`, `clipCrossFail`, `writeClip` and `pruneClips` of `bot.ts` (3164-3239).
//!
//! * **Autoclip** (`CLIP_SCAN_EVERY_FRAMES = 50`): every 50 frames, unless the 45 s game-time cooldown
//!   (2250 ticks) is running and the ring holds at least 50 frames, the incidents of the ring are merged
//!   and filtered by severity (250, but 180 for `self-freeze`, `chased-into-freeze` and `goto-into-freeze`),
//!   only an incident in the **second half** of the buffer counts, and the most severe is saved.
//! * **Cross-fail** clip: when the navigator's notes say "...; trying again from the spawn" or "no way
//!   through ...", with a 60 s cooldown (3000 ticks).
//! * **Duel round-loss clips** (task 3.19, D-116): in a detected F-DDrace duel the bot saves a clip at the start of every round it ended frozen or dead in
//!   (`duel-loss-<tick>.clip`), with **no** cooldown. They are not "automatic" in the sense of [`parse_auto_name`] (no severity suffix), so the pruning above
//!   never touches them; their own bound is [`prune_duel`] ([`DUEL_KEEP`] files, [`DUEL_MAX_BYTES`] bytes), and the bot caps one session at [`DUEL_SESSION_MAX`].
//! * **Pruning**: at most [`KEEP_AUTO`] = 24 automatic clips in all and [`KEEP_PER_KIND`] = 16 of a kind,
//!   newest first; `manual-*` is never touched. The TS pattern `^(.*)-\d+-s(\d+)\.json$` never matched
//!   `cross-fail-<tick>.json`, so those piled up for ever; here `cross-fail` clips are named like every
//!   other automatic clip (`cross-fail-<tick>-s0.clip`) and pruned with them.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::incidents::{Incident, merge_overlapping};

/// `CLIP_SCAN_EVERY_FRAMES`.
pub const SCAN_EVERY_FRAMES: u32 = 50;
/// `CLIP_COOLDOWN_TICKS` = 45 s of game time.
pub const COOLDOWN_TICKS: i32 = 45 * 50;
/// `CROSS_CLIP_COOLDOWN_TICKS` = 60 s.
pub const CROSS_COOLDOWN_TICKS: i32 = 60 * 50;
/// A clip needs at least this many frames.
pub const MIN_FRAMES: usize = 50;
/// `CLIP_KEEP` and `CLIP_KEEP_PER_KIND`.
pub const KEEP_AUTO: usize = 24;
pub const KEEP_PER_KIND: usize = 16;
/// `CLIP_SEVERITY` and the lower one of the freeze kinds (`CLIP_SEVERITY_BY_KIND`).
pub const SEVERITY: i32 = 250;
pub const SEVERITY_FREEZE: i32 = 180;
/// The extension of a clip file.
pub const EXTENSION: &str = "clip";
/// Task 3.19: the kind and the file name prefix of a duel round-loss clip.
pub const DUEL_KIND: &str = "duel-loss";
/// Task 3.19: the duel clips kept on disk at most (the oldest go first), and their total size.
pub const DUEL_KEEP: usize = 60;
pub const DUEL_MAX_BYTES: u64 = 12 << 20;
/// Task 3.19: the duel clips one session of the bot saves at most (a clip is about 50 KB; a 20-round duel is 1 MB).
pub const DUEL_SESSION_MAX: usize = 40;

/// `<data dir>/bot/clips` (`~/aiddnet/data/bot/clips` on Linux, see `ddai_os::dirs`); `None` when there is no home directory (never a
/// relative path).
pub fn default_clip_dir() -> Option<PathBuf> {
    ddai_os::dirs::data_root().map(|d| d.join("bot/clips"))
}

/// The severity an incident of `kind` needs to be worth a clip.
pub fn threshold(kind: &str) -> i32 {
    match kind {
        "self-freeze" | "chased-into-freeze" | "goto-into-freeze" => SEVERITY_FREEZE,
        _ => SEVERITY,
    }
}

/// Task 3.19: `duel-loss-<tick>` (no severity suffix: not an automatic clip for [`prune`]).
pub fn duel_name(tick: i32) -> String {
    format!("{DUEL_KIND}-{tick}")
}

/// Task 3.19: deletes the oldest `duel-loss-*` clips of `dir` beyond [`DUEL_KEEP`] files or [`DUEL_MAX_BYTES`] bytes. Returns the deleted paths.
pub fn prune_duel(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<(SystemTime, PathBuf, u64)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let is_duel = path.extension().and_then(|e| e.to_str()) == Some(EXTENSION)
            && path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with(&format!("{DUEL_KIND}-")));
        if !is_duel {
            continue;
        }
        let meta = entry.metadata()?;
        files.push((meta.modified().unwrap_or(SystemTime::UNIX_EPOCH), path, meta.len()));
    }
    // Newest first; keep while both bounds hold.
    files.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let (mut kept, mut bytes) = (0usize, 0u64);
    let mut removed = Vec::new();
    for (_, path, len) in files {
        if kept < DUEL_KEEP && bytes + len <= DUEL_MAX_BYTES {
            kept += 1;
            bytes += len;
            continue;
        }
        std::fs::remove_file(&path)?;
        removed.push(path);
    }
    Ok(removed)
}

/// Characters outside `[a-zA-Z0-9_-]` become `_` (`writeClip`).
pub fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `manual-<tick>[-<note>]`.
pub fn manual_name(tick: i32, note: &str) -> String {
    let note = note.trim();
    sanitize(&if note.is_empty() {
        format!("manual-{tick}")
    } else {
        format!("manual-{tick}-{note}")
    })
}

/// `<kind>-<tick>-s<severity>`.
pub fn auto_name(kind: &str, tick: i32, severity: i32) -> String {
    sanitize(&format!("{kind}-{tick}-s{severity}"))
}

/// The worst incident worth a clip, given the ring's frames' ticks: severity over the threshold of its
/// kind and in the second half of the buffer.
pub fn pick_incident(incidents: Vec<Incident>, frame_ticks: &[i32]) -> Option<Incident> {
    if frame_ticks.len() < MIN_FRAMES {
        return None;
    }
    let half = frame_ticks[frame_ticks.len() / 2];
    merge_overlapping(incidents)
        .into_iter()
        .filter(|i| i.severity >= threshold(i.kind))
        .filter(|i| i.tick >= half)
        .max_by(|a, b| a.severity.cmp(&b.severity).then(b.tick.cmp(&a.tick)))
}

/// Frames of context kept on each side of an incident (`findIncidents`' default).
pub const CONTEXT_TICKS: i32 = 40;

/// The autoclip's verdict on a clip-shaped ring: the incident worth saving, if any.
pub fn scan(clip: &crate::format::Clip) -> Option<Incident> {
    let ticks: Vec<i32> = clip.frames.iter().map(|f| f.tick).collect();
    pick_incident(
        crate::incidents::find_incidents(&clip.frames, clip.header.own_id, CONTEXT_TICKS),
        &ticks,
    )
}

/// Writes `clip` as `<dir>/<name>.clip` (atomically) and returns the path. The caller prunes after an
/// automatic clip ([`prune`]).
pub fn save(dir: &Path, clip: &crate::format::Clip, name: &str) -> Result<PathBuf, crate::format::ClipError> {
    let path = dir.join(format!("{}.{EXTENSION}", sanitize(name)));
    clip.write(&path)?;
    Ok(path)
}

/// Whether a navigator note asks for a cross-fail clip (`/; trying again from the spawn$|^no way through /`).
pub fn is_cross_fail_note(note: &str) -> bool {
    note.ends_with("; trying again from the spawn") || note.starts_with("no way through ")
}

/// The scheduling state of the autoclip: the frame counter and the two cooldowns.
#[derive(Debug, Clone)]
pub struct AutoClip {
    pub enabled: bool,
    frames_since_scan: u32,
    last_clip_tick: i32,
    last_cross_tick: i32,
}

impl Default for AutoClip {
    fn default() -> Self {
        AutoClip {
            enabled: true,
            frames_since_scan: 0,
            last_clip_tick: i32::MIN / 2,
            last_cross_tick: i32::MIN / 2,
        }
    }
}

impl AutoClip {
    /// Counts a frame; true when the ring should be scanned now (every [`SCAN_EVERY_FRAMES`] frames).
    pub fn frame(&mut self) -> bool {
        self.frames_since_scan += 1;
        if self.frames_since_scan >= SCAN_EVERY_FRAMES {
            self.frames_since_scan = 0;
            return self.enabled;
        }
        false
    }

    /// The cooldown allows an incident clip at `tick`.
    pub fn ready(&self, tick: i32) -> bool {
        self.enabled && (tick < self.last_clip_tick || tick - self.last_clip_tick >= COOLDOWN_TICKS)
    }

    pub fn clipped(&mut self, tick: i32) {
        self.last_clip_tick = tick;
    }

    /// The cooldown allows a cross-fail clip at `tick`.
    pub fn cross_ready(&self, tick: i32) -> bool {
        self.enabled && (tick < self.last_cross_tick || tick - self.last_cross_tick >= CROSS_COOLDOWN_TICKS)
    }

    pub fn cross_clipped(&mut self, tick: i32) {
        self.last_cross_tick = tick;
    }

    /// A tick reset or a new map: the cooldowns start over.
    pub fn reset(&mut self) {
        self.frames_since_scan = 0;
        self.last_clip_tick = i32::MIN / 2;
        self.last_cross_tick = i32::MIN / 2;
    }
}

/// An automatic clip on disk, by its name `<kind>-<tick>-s<severity>.clip`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoFile {
    pub path: PathBuf,
    pub kind: String,
    pub severity: i32,
    pub modified: SystemTime,
}

/// Parses `<kind>-<tick>-s<severity>`; `manual-*` and anything else that does not match is not automatic.
pub fn parse_auto_name(stem: &str) -> Option<(String, i32)> {
    if stem.starts_with("manual-") {
        return None;
    }
    let (rest, sev) = stem.rsplit_once("-s")?;
    let severity: i32 = sev.parse().ok()?;
    let (kind, tick) = rest.rsplit_once('-')?;
    if kind.is_empty() || tick.is_empty() || !tick.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((kind.to_string(), severity))
}

/// Deletes the automatic clips of `dir` beyond [`KEEP_AUTO`] in all and [`KEEP_PER_KIND`] of a kind,
/// newest first (ties: the more severe first). Returns the deleted paths.
pub fn prune(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut auto: Vec<AutoFile> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(EXTENSION) {
            continue;
        }
        let Some((kind, severity)) = path.file_stem().and_then(|s| s.to_str()).and_then(parse_auto_name) else {
            continue;
        };
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        auto.push(AutoFile {
            path,
            kind,
            severity,
            modified,
        });
    }
    auto.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then(b.severity.cmp(&a.severity))
            .then(a.path.cmp(&b.path))
    });
    let mut per_kind: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut kept = 0;
    let mut removed = Vec::new();
    for f in auto {
        let n = per_kind.entry(f.kind.clone()).or_insert(0);
        if kept < KEEP_AUTO && *n < KEEP_PER_KIND {
            *n += 1;
            kept += 1;
            continue;
        }
        std::fs::remove_file(&f.path)?;
        removed.push(f.path);
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn inc(kind: &'static str, tick: i32, severity: i32) -> Incident {
        Incident {
            kind,
            tick,
            from: tick - 40,
            to: tick + 40,
            severity,
            note: String::new(),
        }
    }

    #[test]
    fn names_are_sanitised_and_parse_back() {
        assert_eq!(manual_name(1200, ""), "manual-1200");
        assert_eq!(manual_name(1200, "odd one/../x y"), "manual-1200-odd_one____x_y");
        assert_eq!(auto_name("self-freeze", 3000, 211), "self-freeze-3000-s211");
        assert_eq!(
            parse_auto_name("self-freeze-3000-s211"),
            Some(("self-freeze".into(), 211))
        );
        assert_eq!(parse_auto_name("cross-fail-3000-s0"), Some(("cross-fail".into(), 0)));
        assert_eq!(
            parse_auto_name("manual-3000-s5"),
            None,
            "manual clips are not automatic"
        );
        assert_eq!(parse_auto_name("manual-3000"), None);
        assert_eq!(parse_auto_name("weird"), None);
    }

    #[test]
    fn the_worst_incident_in_the_second_half_over_its_threshold_is_chosen() {
        let ticks: Vec<i32> = (0..100).map(|i| i * 2).collect(); // half at index 50 -> tick 100
        let picked = pick_incident(
            vec![
                inc("death", 20, 400),
                inc("self-freeze", 150, 200),
                inc("wall-grind", 160, 240),
                inc("jitter", 180, 255),
            ],
            &ticks,
        )
        .unwrap();
        assert_eq!(
            (picked.kind, picked.tick),
            ("jitter", 180),
            "250 for jitter; the first-half death does not count; wall-grind 240 < 250"
        );
        let freeze_only = pick_incident(
            vec![inc("self-freeze", 150, 181), inc("goto-into-freeze", 170, 179)],
            &ticks,
        )
        .unwrap();
        assert_eq!(
            freeze_only.kind, "self-freeze",
            "180 for the freeze kinds; 179 is under it"
        );
        assert!(pick_incident(vec![inc("self-freeze", 150, 179)], &ticks).is_none());
        assert!(
            pick_incident(vec![inc("death", 150, 400)], &ticks[..49]).is_none(),
            "fewer than 50 frames"
        );
        // Overlapping incidents (< 25 ticks) are one: the stronger first.
        let merged = pick_incident(vec![inc("death", 150, 300), inc("jitter", 160, 260)], &ticks).unwrap();
        assert!(merged.note.contains("also jitter"));
    }

    #[test]
    fn the_autoclip_scans_every_fifty_frames_and_keeps_its_cooldowns() {
        let mut a = AutoClip::default();
        let scans = (1..=120).filter(|_| a.frame()).count();
        assert_eq!(scans, 2, "two scans in 120 frames");
        assert!(a.ready(10_000));
        a.clipped(10_000);
        assert!(!a.ready(10_000 + COOLDOWN_TICKS - 1));
        assert!(a.ready(10_000 + COOLDOWN_TICKS));
        assert!(a.ready(5), "a tick reset (time went back) lifts it");
        assert!(a.cross_ready(0));
        a.cross_clipped(500);
        assert!(!a.cross_ready(500 + CROSS_COOLDOWN_TICKS - 1) && a.cross_ready(500 + CROSS_COOLDOWN_TICKS));
        a.enabled = false;
        assert!(!a.ready(10_000_000) && !a.cross_ready(10_000_000) && !(0..60).any(|_| a.frame()));
    }

    #[test]
    fn duel_clips_are_not_automatic_and_are_pruned_by_count_and_by_size_oldest_first() {
        assert_eq!(
            parse_auto_name(&duel_name(71936)),
            None,
            "the autoclip pruning never sees a duel clip"
        );
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, age_s: u64, len: usize| {
            let p = dir.path().join(format!("{name}.{EXTENSION}"));
            std::fs::write(&p, vec![b'x'; len]).unwrap();
            let t = SystemTime::now() - Duration::from_secs(age_s);
            std::fs::File::options()
                .write(true)
                .open(&p)
                .unwrap()
                .set_modified(t)
                .unwrap();
            p
        };
        // 70 small duel clips (age 100 + i): the 60 newest stay.
        for i in 0..70 {
            write(&duel_name(1000 + i), 100 + i as u64, 10);
        }
        let other = write("self-freeze-5-s200", 5000, 10);
        let manual = write("manual-5", 9000, 10);
        let gone = prune_duel(dir.path()).unwrap();
        assert_eq!(gone.len(), 10, "{gone:?}");
        assert!(
            gone.iter()
                .all(|p| p.file_stem().unwrap().to_str().unwrap().starts_with("duel-loss-"))
        );
        assert!(other.exists() && manual.exists(), "other clips are not its business");
        // The size bound: three 5 MiB clips do not fit in 12 MiB together, the oldest goes.
        let dir2 = tempfile::tempdir().unwrap();
        for (i, age) in [(0, 300u64), (1, 200), (2, 100)] {
            let p = dir2.path().join(format!("{}.{EXTENSION}", duel_name(i)));
            std::fs::write(&p, vec![b'x'; 5 << 20]).unwrap();
            let t = SystemTime::now() - Duration::from_secs(age);
            std::fs::File::options()
                .write(true)
                .open(&p)
                .unwrap()
                .set_modified(t)
                .unwrap();
        }
        let gone = prune_duel(dir2.path()).unwrap();
        assert_eq!(gone.len(), 1);
        assert!(
            gone[0].to_str().unwrap().contains("duel-loss-0"),
            "the oldest: {gone:?}"
        );
    }

    #[test]
    fn cross_fail_notes_are_recognised() {
        assert!(is_cross_fail_note(
            "the right freeze tube: no hop from (131,50) that clears the freeze; trying again from the spawn"
        ));
        assert!(is_cross_fail_note("no way through the left freeze tube"));
        assert!(!is_cross_fail_note("heading for (3,4), 9 tiles away"));
    }

    #[test]
    fn prune_keeps_24_in_all_16_per_kind_never_a_manual_and_covers_cross_fail() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, age_s: u64| {
            let p = dir.path().join(format!("{name}.{EXTENSION}"));
            std::fs::write(&p, b"x").unwrap();
            let t = SystemTime::now() - Duration::from_secs(age_s);
            std::fs::File::options()
                .write(true)
                .open(&p)
                .unwrap()
                .set_modified(t)
                .unwrap();
            p
        };
        // 20 self-freeze (age 1000+i), 10 death, 10 cross-fail, 5 manual (oldest of all).
        for i in 0..20 {
            write(&format!("self-freeze-{i}-s300"), 1000 + i);
        }
        for i in 0..10 {
            write(&format!("death-{i}-s400"), 2000 + i);
        }
        for i in 0..10 {
            write(&format!("cross-fail-{i}-s0"), 3000 + i);
        }
        for i in 0..5 {
            write(&format!("manual-{i}"), 99_999 + i);
        }
        std::fs::write(dir.path().join("notes.txt"), b"mine").unwrap();
        let removed = prune(dir.path()).unwrap();
        let left: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        let count = |pre: &str| {
            left.iter()
                .filter(|n| n.starts_with(pre) && n.ends_with(".clip"))
                .count()
        };
        assert_eq!(count("manual-"), 5, "manual clips are kept");
        assert!(
            left.contains(&"notes.txt".to_string()),
            "other files are not ours to delete"
        );
        assert_eq!(count("self-freeze"), 16, "16 per kind, newest kept");
        assert_eq!(
            count("self-freeze") + count("death") + count("cross-fail"),
            KEEP_AUTO,
            "24 automatic clips in all"
        );
        assert_eq!(
            count("death"),
            8,
            "the next newest after the 16 self-freezes: the deaths, then cross-fail only gets what is left"
        );
        assert_eq!(
            count("cross-fail"),
            0,
            "the oldest kind is pruned: cross-fail clips are pruned too (the TS never pruned them)"
        );
        assert_eq!(removed.len(), 40 - KEEP_AUTO);
        assert!(
            left.contains(&"self-freeze-0-s300.clip".to_string())
                && !left.contains(&"self-freeze-19-s300.clip".to_string())
        );
    }
}
