//! The learned window model in the live bot (task 3.17, D-111): switch, kill marker, STATUS, the log's thread.
//!
//! **What it does.** `ddai-oppnet`'s predictor guesses the opponent's inputs for the ticks the input lag hides (the window between the
//! snapshot and the tick our input lands on). The bot's decision world is `LiveWorld`'s prediction to that tick, in which every other tee holds
//! the input its snapshot shows; with the model on, the *target* plays the model's inputs in that roll instead
//! ([`ddai_world::LiveWorld::predict_local_observation_with`]). Everything after the window is as before. An online guard
//! ([`ddai_oppnet::live::guard`]) scores the model against hold on what the opponent really did and benches it when it loses.
//!
//! **Switching** (default: off, byte-identical). `--window-model <file>` or `window_model = "<file>"` in `settings.toml` loads the model at the
//! start (an unreadable or foreign file refuses the start); the marker file `<data-dir>/bot/window-model.off` switches it off while it exists, read
//! at the start and once a second by [`WindowModelRt::poll`] (any entry of that name counts; an error other than "not found" counts as present: fail
//! safe), outside the decision path. While off the model is not called, nothing is logged, and the bot decides exactly as without the flag.
//!
//! **Cost.** One forward pass per brain decision (about 10 us) plus the frame features per snapshot with a target; no allocation in the decision path
//! (the log lines are formatted into a buffer, the file is written by [`ddai_oppnet::live::writer::LogWriter`]'s thread).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ddai_brain::Observation;
use ddai_oppnet::OppPredictor;
use ddai_oppnet::live::guard::{GuardConfig, GuardState, Transition};
use ddai_oppnet::live::writer::{DEFAULT_KEEP, DEFAULT_MAX_BYTES, LogWriter};
use ddai_oppnet::live::{LiveOpp, Pair, RegimeGate, WindowUse, hex};
use ddai_physics::core::{MAX_CLIENTS, PlayerInput as PhysInput};
use ddai_physics::world::World;
use ddai_world::LiveWorld;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// The marker file name inside `<data-dir>/bot/`.
pub const WINDOW_MODEL_OFF_MARKER: &str = "window-model.off";
/// The live log's name inside `<data-dir>/bot/`.
pub const LOG_NAME: &str = "oppnet-live.jsonl";
/// How often the marker is looked at.
pub const RECHECK_EVERY: Duration = Duration::from_secs(1);

/// What the bot is told to run.
#[derive(Debug, Clone)]
pub struct WindowModelConfig {
    /// The `.oppnet` bundle (never in git).
    pub model: PathBuf,
    /// The kill marker; `None`: none.
    pub marker: Option<PathBuf>,
    /// The live log; `None`: no log.
    pub log: Option<PathBuf>,
    pub guard: GuardConfig,
    /// The situations the model drives in (see [`RegimeGate`]).
    pub gate: RegimeGate,
    /// The server's address as the log's header names it (set by the runner).
    pub server_tag: String,
    pub log_max_bytes: u64,
    pub log_keep: usize,
}

impl WindowModelConfig {
    pub fn new(model: PathBuf) -> WindowModelConfig {
        WindowModelConfig {
            model,
            marker: None,
            log: None,
            guard: GuardConfig::default(),
            gate: RegimeGate::default(),
            server_tag: String::new(),
            log_max_bytes: DEFAULT_MAX_BYTES,
            log_keep: DEFAULT_KEEP,
        }
    }

    /// The marker and the log in `<data-dir>/bot/`.
    pub fn in_data_dir(model: PathBuf, data_dir: &Path) -> WindowModelConfig {
        let dir = data_dir.join("bot");
        WindowModelConfig {
            marker: Some(dir.join(WINDOW_MODEL_OFF_MARKER)),
            log: Some(dir.join(LOG_NAME)),
            ..WindowModelConfig::new(model)
        }
    }
}

/// Seconds since the epoch (the log header's start time).
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Whether the marker counts as present: only "not found" means absent (any other error leaves it unknown, and unknown keeps the model off).
fn marker_present(path: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Ok(_) => true,
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

/// STATUS's view of the model (`window_model` and `window_guard`).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WindowGuardStatus {
    /// `"model"` while the model drives the window, `"hold"` while the guard has benched it.
    pub driver: &'static str,
    /// Windows in the judging span and the mean cost per sample of the model and of hold over it.
    pub windows: usize,
    pub model_cost: f32,
    pub hold_cost: f32,
    pub fallbacks: u32,
    pub retries: u32,
    /// Windows resolved since the start.
    pub resolved: u64,
    /// Decisions the model was run for, and of them those where it drove the roll.
    pub predicted: u64,
    pub used: u64,
    /// Decisions with no model call (an empty window or one longer than the model's; an opponent frozen through the window; our own tee frozen; outside the regime gate) and snapshots whose
    /// frame fed the history.
    pub skipped_window: u64,
    pub skipped_frozen: u64,
    pub skipped_own_frozen: u64,
    pub skipped_regime: u64,
    pub observed: u64,
    /// Lengths of the windows decisions asked about (`0..=9`, the last bin holds longer ones).
    pub window_lens: [u64; 10],
    /// Log lines lost (a full buffer or queue, a write error).
    pub log_lost: u64,
    /// The model file's sha256 (hex).
    pub sha256: String,
}

/// A fixed-size string for the opponent's tag, formatted without allocating.
struct StackStr {
    buf: [u8; 32],
    len: usize,
}

impl StackStr {
    fn new() -> StackStr {
        StackStr { buf: [0; 32], len: 0 }
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl std::fmt::Write for StackStr {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let room = self.buf.len() - self.len;
        let take = s.len().min(room);
        self.buf[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

/// The loaded model and everything around it.
pub struct WindowModelRt {
    live: LiveOpp,
    path: PathBuf,
    sha_hex: String,
    marker: Option<PathBuf>,
    killed: bool,
    next_check: Instant,
    writer: Option<LogWriter>,
    own: Vec<PhysInput>,
    victim: Vec<PhysInput>,
    transitions: Vec<Transition>,
    log_lost: u64,
}

impl WindowModelRt {
    /// Loads the model, logs its sha256 and reads the marker once. Fails (the bot refuses to start) on an unreadable, corrupt or foreign
    /// file.
    pub fn load(cfg: &WindowModelConfig, now: Instant) -> Result<WindowModelRt, String> {
        let bytes = std::fs::read(&cfg.model).map_err(|e| format!("window model {}: {e}", cfg.model.display()))?;
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let pred = OppPredictor::load(&cfg.model)?;
        let live = LiveOpp::new(pred, cfg.guard, sha)
            .map_err(|e| format!("window model: {e}"))?
            .with_gate(cfg.gate);
        let sha_hex = hex(&sha);
        let writer = match &cfg.log {
            Some(path) => Some(
                LogWriter::spawn(
                    path.clone(),
                    cfg.log_max_bytes,
                    cfg.log_keep,
                    live.header_line(unix_now(), &cfg.server_tag),
                )
                .map_err(|e| format!("window model log {}: {e}", path.display()))?,
            ),
            None => None,
        };
        let killed = cfg.marker.as_deref().is_some_and(marker_present);
        let rt = WindowModelRt {
            live,
            path: cfg.model.clone(),
            sha_hex,
            marker: cfg.marker.clone(),
            killed,
            next_check: now + RECHECK_EVERY,
            writer,
            own: Vec::with_capacity(96),
            victim: Vec::with_capacity(16),
            transitions: Vec::with_capacity(4),
            log_lost: 0,
        };
        tracing::info!(
            model = %rt.path.display(),
            sha256 = %rt.sha_hex,
            marker = ?rt.marker,
            log = ?cfg.log,
            state = if rt.killed { "off (marker)" } else { "on" },
            "window model loaded (task 3.17, D-111): the opponent plays the model's inputs in the lag window; the online guard benches it when it loses to hold"
        );
        Ok(rt)
    }

    pub fn sha256_hex(&self) -> &str {
        &self.sha_hex
    }

    /// The kill marker is on: the model is not called.
    pub fn killed(&self) -> bool {
        self.killed
    }

    /// Forgets the opponent pair (a respawn). The guard's state is kept.
    pub fn reset(&mut self) {
        self.live.reset();
    }

    /// One snapshot with a target. Does nothing while the marker is on.
    pub fn observe(&mut self, world: &World<f32>, self_id: i32, target: i32, tag: &impl std::fmt::Display) {
        if self.killed {
            return;
        }
        let mut t = StackStr::new();
        let _ = write!(t, "{tag}");
        self.live.observe(&Pair {
            world,
            self_id,
            target,
            tag: t.as_str(),
        });
    }

    /// The world a brain decision is made on: `live`'s prediction to `to_tick`, in which the target plays the model's inputs for the
    /// window when the model is on and the guard allows it, else exactly what [`LiveWorld::predict_local_observation`] gives.
    #[allow(clippy::too_many_arguments)]
    pub fn predict<'a>(
        &mut self,
        live: &'a mut LiveWorld,
        to_tick: i32,
        in_flight: &[(i32, PhysInput)],
        keep: &[bool; MAX_CLIENTS],
        target: i32,
        tag: &impl std::fmt::Display,
        obs: &mut Observation,
    ) -> &'a World<f32> {
        if self.killed {
            return live.predict_local_observation(to_tick, in_flight, keep, Some(target), obs);
        }
        live.own_inputs_over(to_tick, in_flight, &mut self.own);
        let hold = live.held_input_of(target).unwrap_or_default();
        let mut t = StackStr::new();
        let _ = write!(t, "{tag}");
        let own_id = live.own_id();
        let pair = Pair {
            world: live.base_world(),
            self_id: own_id,
            target,
            tag: t.as_str(),
        };
        let how = self.live.window(&pair, &self.own, &hold, &mut self.victim);
        match how {
            WindowUse::Model => live.predict_local_observation_with(
                to_tick,
                in_flight,
                keep,
                Some(target),
                obs,
                Some((target, &self.victim)),
            ),
            WindowUse::Hold => live.predict_local_observation(to_tick, in_flight, keep, Some(target), obs),
        }
    }

    /// Logs the guard's changes of state since the last look (also at the end of the run: a change in the last second is not lost).
    fn report_transitions(&mut self) {
        self.live.take_transitions(&mut self.transitions);
        for t in self.transitions.drain(..) {
            match t.to {
                GuardState::Fallback => tracing::warn!(
                    model_cost = t.model_cost,
                    hold_cost = t.hold_cost,
                    samples = t.samples,
                    windows = t.windows,
                    "window model: the guard benched the model (it is worse than hold over the judging span); hold drives the window, the model keeps being scored"
                ),
                GuardState::Active => tracing::info!(
                    model_cost = t.model_cost,
                    hold_cost = t.hold_cost,
                    samples = t.samples,
                    windows = t.windows,
                    "window model: the guard gave the model the window back (its shadow score is not worse than hold)"
                ),
            }
        }
    }

    /// Once a second, outside the decision path: reads the marker, hands the log to its thread, reports the guard's changes.
    pub fn poll(&mut self, now: Instant) {
        if now < self.next_check {
            return;
        }
        self.next_check = now + RECHECK_EVERY;
        let chunk = self.live.take_log();
        if let Some(w) = &self.writer
            && !w.send(chunk)
        {
            self.log_lost += 1;
        }
        self.report_transitions();
        let Some(marker) = &self.marker else { return };
        let present = marker_present(marker);
        if present == self.killed {
            return;
        }
        self.killed = present;
        // Either way the history starts afresh: after a pause the frames are stale.
        self.live.reset();
        if present {
            tracing::info!("window model: off (the marker {} exists)", marker.display());
        } else {
            tracing::info!("window model: on (the marker is gone)");
        }
    }

    /// STATUS's `window_model`: `"on"`, `"hold"` (benched by the guard) or `"killed"` (the marker).
    pub fn state_word(&self) -> &'static str {
        if self.killed {
            "killed"
        } else {
            ddai_oppnet::live::state_word(self.live.guard_status().state)
        }
    }

    /// STATUS's `window_guard`.
    pub fn guard_status(&self) -> WindowGuardStatus {
        let g = self.live.guard_status();
        let c = self.live.counts();
        let lost = self.writer.as_ref().map_or(0, |w| {
            let (d, e) = w.losses();
            d + e
        });
        WindowGuardStatus {
            driver: g.state.name(),
            windows: g.windows,
            model_cost: g.model_cost,
            hold_cost: g.hold_cost,
            fallbacks: g.fallbacks,
            retries: g.retries,
            resolved: g.resolved,
            predicted: c.predicted,
            used: c.used,
            skipped_window: c.skipped_window,
            skipped_frozen: c.skipped_frozen,
            skipped_own_frozen: c.skipped_own_frozen,
            skipped_regime: c.skipped_regime,
            observed: c.observed,
            window_lens: c.window_lens,
            log_lost: self.log_lost + c.log_dropped + lost,
            sha256: self.sha_hex.clone(),
        }
    }
}

impl Drop for WindowModelRt {
    /// The lines formatted since the last look reach the file before the thread is joined.
    fn drop(&mut self) {
        self.report_transitions();
        let chunk = self.live.take_log();
        if let Some(w) = &self.writer {
            w.send(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marker_fails_safe() {
        let dir = tempfile::tempdir().unwrap();
        let m = dir.path().join(WINDOW_MODEL_OFF_MARKER);
        assert!(!marker_present(&m));
        std::fs::write(&m, "").unwrap();
        assert!(marker_present(&m), "an empty file counts");
        std::fs::remove_file(&m).unwrap();
        std::os::unix::fs::symlink("/nonexistent", &m).unwrap();
        assert!(marker_present(&m), "a dangling symlink counts too");
        // A path below a regular file: ENOTDIR is not "not found": present.
        let f = dir.path().join("file");
        std::fs::write(&f, "x").unwrap();
        assert!(marker_present(&f.join("window-model.off")));
    }

    #[test]
    fn the_tag_buffer_formats_without_allocating_and_truncates() {
        let mut t = StackStr::new();
        let _ = write!(t, "c{}-{}", 12, 0x0a1b_2c3d_u32);
        assert_eq!(t.as_str(), "c12-169552957");
        let mut t = StackStr::new();
        let _ = write!(t, "{}", "x".repeat(100));
        assert_eq!(t.as_str().len(), 32);
    }

    #[test]
    fn a_missing_or_foreign_model_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = WindowModelConfig::new(dir.path().join("none.oppnet"));
        assert!(WindowModelRt::load(&cfg, Instant::now()).is_err());
        let junk = dir.path().join("junk.oppnet");
        std::fs::write(&junk, b"not a model").unwrap();
        let err = match WindowModelRt::load(&WindowModelConfig::new(junk), Instant::now()) {
            Ok(_) => panic!("a junk file loaded"),
            Err(e) => e,
        };
        assert!(!err.is_empty());
    }
}
