//! The duel switch "never kill ourselves" (task 4.11, D-102).
//!
//! **Why.** In an F-DDrace `/1vs1` any death of ours (`Cl_Kill`, `/kill`, a kill tile) is a point for the opponent, and a frozen tee
//! standing on the ground dies by the server's rule anyway, so a bot-initiated kill never helps there.
//!
//! **What it stops** ([`crate::Bot::set_no_selfkill`]): the unstick `Cl_Kill` (all reasons), the wayblock "lying" kill, the `/kill`
//! fallback of D-078, a navigation route's respawn step (such routes are not planned) and a trek's. **What it leaves:** the owner's own
//! console `!kill` and the lines the owner types on the website (his choice).
//!
//! **How it is switched.** The flag `--no-selfkill` (always on for the run) or the marker file `<data-dir>/bot/selfkill.off` (any
//! entry of that name counts; an error other than "not found" counts as present: fail safe), read at the start and again once a second by [`SelfKillSwitch::poll`], so the lead can toggle it
//! without a restart. The re-check is one `symlink_metadata` call per second in the runner loop, outside the decision path.

use std::path::PathBuf;
use std::time::{Duration, Instant};

/// The marker file name inside `<data-dir>/bot/`.
pub const SELFKILL_OFF_MARKER: &str = "selfkill.off";

/// Task 4.12 (D-108): the marker file that turns the automatic duel detection off, `<data-dir>/bot/duel-detect.off`. It is read exactly like
/// `selfkill.off` (same [`SelfKillSwitch`], once a second, an unreadable marker counts as present) but with the opposite fail-safe in mind: its
/// presence means "do not look for a duel", so a stuck bot can always be freed without a restart or SSH to the unit.
pub const DUEL_DETECT_OFF_MARKER: &str = "duel-detect.off";
/// Task 3.20: the marker that stops the use of the server's pre-inputs (the same one-second reader).
pub const PREINPUT_OFF_MARKER: &str = "preinput.off";

/// How often the marker is looked at.
pub const RECHECK_EVERY: Duration = Duration::from_secs(1);

/// Why the switch is on (for the one start line and the log).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfKillOff {
    Flag,
    Marker,
}

impl SelfKillOff {
    pub fn name(self) -> &'static str {
        match self {
            SelfKillOff::Flag => "flag",
            SelfKillOff::Marker => "marker",
        }
    }
}

/// Whether the marker counts as present. It fails safe on every platform: only a plain "not found" means absent; any other state (EACCES,
/// a path below a regular file, ...) leaves the marker unknown, and an unknown marker keeps the bot from killing itself
/// (`ddai_os::marker`).
fn marker_present(path: &std::path::Path) -> bool {
    ddai_os::marker::is_present(path)
}

/// The switch's state: the flag, the marker path and the time of the last look.
#[derive(Debug)]
pub struct SelfKillSwitch {
    flag: bool,
    marker: Option<PathBuf>,
    next_check: Instant,
    state: Option<SelfKillOff>,
}

impl SelfKillSwitch {
    /// Reads the marker once, now.
    pub fn new(flag: bool, marker: Option<PathBuf>, now: Instant) -> Self {
        let mut s = SelfKillSwitch {
            flag,
            marker,
            next_check: now,
            state: None,
        };
        s.state = s.look();
        s.next_check = now + RECHECK_EVERY;
        s
    }

    fn look(&self) -> Option<SelfKillOff> {
        if self.flag {
            Some(SelfKillOff::Flag)
        } else if self.marker.as_ref().is_some_and(|p| marker_present(p)) {
            Some(SelfKillOff::Marker)
        } else {
            None
        }
    }

    /// Whether the switch is on (no kill by the bot) and why.
    pub fn state(&self) -> Option<SelfKillOff> {
        self.state
    }

    /// Looks at the marker when a second has passed. `Some(new_state)` when the state changed, else `None`. With the flag set nothing
    /// is looked at (the flag cannot be lifted at run time).
    pub fn poll(&mut self, now: Instant) -> Option<Option<SelfKillOff>> {
        if self.flag || now < self.next_check {
            return None;
        }
        self.next_check = now + RECHECK_EVERY;
        let new = self.look();
        if new == self.state {
            return None;
        }
        self.state = new;
        Some(new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ddai-selfkill-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn off_without_flag_or_marker() {
        let d = tmp("none");
        let mut s = SelfKillSwitch::new(false, Some(d.join(SELFKILL_OFF_MARKER)), Instant::now());
        assert_eq!(s.state(), None);
        assert_eq!(s.poll(Instant::now() + Duration::from_secs(5)), None);
        let mut s = SelfKillSwitch::new(false, None, Instant::now());
        assert_eq!(s.state(), None);
        assert_eq!(s.poll(Instant::now() + Duration::from_secs(5)), None);
    }

    #[test]
    fn a_marker_that_cannot_be_looked_at_counts_as_present() {
        // A file where the directory should be: `symlink_metadata` fails with ENOTDIR on Linux and with a plain "not found" on Windows; both
        // must count as present (`ddai_os::marker`).
        let d = tmp("unreadable");
        let file = d.join("bot");
        std::fs::write(&file, b"").unwrap();
        let s = SelfKillSwitch::new(false, Some(file.join(SELFKILL_OFF_MARKER)), Instant::now());
        assert_eq!(s.state(), Some(SelfKillOff::Marker));
        // A missing directory is plain "not found": absent.
        let s = SelfKillSwitch::new(false, Some(d.join("nowhere").join(SELFKILL_OFF_MARKER)), Instant::now());
        assert_eq!(s.state(), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_flag_wins_and_stays() {
        let d = tmp("flag");
        let mut s = SelfKillSwitch::new(true, Some(d.join(SELFKILL_OFF_MARKER)), Instant::now());
        assert_eq!(s.state(), Some(SelfKillOff::Flag));
        assert_eq!(s.poll(Instant::now() + Duration::from_secs(5)), None);
        assert_eq!(s.state(), Some(SelfKillOff::Flag));
    }

    #[test]
    fn a_marker_at_the_start_counts_and_is_rechecked_once_a_second() {
        let d = tmp("marker");
        let m = d.join(SELFKILL_OFF_MARKER);
        std::fs::write(&m, b"").unwrap();
        let t0 = Instant::now();
        let mut s = SelfKillSwitch::new(false, Some(m.clone()), t0);
        assert_eq!(s.state(), Some(SelfKillOff::Marker));
        std::fs::remove_file(&m).unwrap();
        // Within the second nothing is looked at.
        assert_eq!(s.poll(t0 + Duration::from_millis(500)), None);
        assert_eq!(s.state(), Some(SelfKillOff::Marker));
        assert_eq!(s.poll(t0 + Duration::from_millis(1100)), Some(None));
        assert_eq!(s.state(), None);
        std::fs::write(&m, b"").unwrap();
        assert_eq!(
            s.poll(t0 + Duration::from_millis(2200)),
            Some(Some(SelfKillOff::Marker))
        );
        assert_eq!(s.poll(t0 + Duration::from_millis(3300)), None);
        let _ = std::fs::remove_dir_all(&d);
    }
}
