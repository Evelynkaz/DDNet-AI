//! The replay `FrameSource` (acceptance criterion 2): plays back Oracle B `trace-b` files (task
//! 1.5) from a configured directory or a single file, looping through the corpus in order.
//!
//! **Deliberately deferred from this task's `GameEvent` set** (see `crate::live::source`'s
//! `GameEvent` doc comment): `HammerHit` and `Teleport`. Both are computable from trace-b (the
//! coverage counters in `docs/formats.md` §12.7 show exactly how), but:
//! - `HammerHit` additionally needs the tuning parameters active during that run (`m_HammerHitFireDelay`/
//!   `m_HammerFireDelay`, tuning-dependent, not part of the trace-b schema itself) to reproduce
//!   the coverage script's exact "hit vs miss" formula — guessing at default tuning would silently
//!   misclassify any trace generated with a tuning override (this corpus does have some, per
//!   `docs/formats.md` §2's tuning-override section).
//! - `Teleport` needs a per-tick tile lookup against the *previous* tick's position on the
//!   now-loaded map, which this module's per-tick decode loop does not thread through today.
//!
//! Both are straightforward additions for a follow-up task; this task ships the four event kinds
//! it can derive exactly, with no heuristic guessing, from fields trace-b already records.

mod trace_b;

#[cfg(any(test, feature = "test-util"))]
pub use trace_b::testutil;
pub use trace_b::{TraceBError, TraceBReader};

use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::sync::mpsc;

use super::map_resolve::{self, MapCache};
use super::source::{
    CharacterState, FrameSource, GameEvent, MapMeta, PlayerMeta, ReplayControl, ReplayStatus, SourceEvent, WorldFrame,
};
use trace_b::{COREEVENT_HOOK_ATTACH_PLAYER, TraceBCharacterRow, TraceBTick};

/// Runs `f` (a synchronous parse of untrusted trace/map bytes), converting a panic into an `Err`
/// message instead of letting it unwind further (review round 1, finding F1: "isolate each file
/// ... so a future parser bug can't kill the source"). `decode_hex_sha256`'s own char-boundary
/// bug (fixed directly, see `trace_b.rs`) was the concrete repro, but this is defense in depth
/// against the *next* one too — every synchronous, byte-parsing call in this file's hot path
/// (`TraceBReader::open`, `reader.next_tick()`, both map-resolution paths) goes through this, so
/// a panic anywhere in any of them only ever fails the one file being processed: `play_one_file`
/// returns an `Err` exactly as it would for an ordinary parse error, the caller reports one
/// `SourceEvent::Error` and moves on to the next file, and the replay task itself — and this
/// viewer's live feed for every *other*, valid file — keeps running.
fn catch_panic_result<T, E: std::fmt::Display>(
    f: impl FnOnce() -> Result<T, E> + std::panic::UnwindSafe,
) -> Result<T, String> {
    match std::panic::catch_unwind(f) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(error.to_string()),
        // `payload.as_ref()`, NOT a bare `&payload`: `payload: Box<dyn Any + Send>` is itself
        // `'static` (its contents must be, per `catch_unwind`'s own bound) and therefore ALSO
        // satisfies `Any`'s blanket impl in its own right — `&payload` coerces to `&dyn Any` by
        // unsizing the *Box value itself* (giving back `TypeId::of::<Box<dyn Any + Send>>()`,
        // which never downcasts to `&str`/`String` no matter what actually panicked), not by
        // dereferencing through the box to the real payload inside it. `.as_ref()` (or, in the
        // caller, method-call syntax like `payload.downcast_ref()`, which auto-derefs correctly)
        // goes through `Box`'s own `Deref` impl instead and reaches the real payload. Caught by
        // this module's own `catch_panic_result_turns_a_panic_into_an_err_instead_of_unwinding`
        // test, which failed with the bare-`&payload` version every time despite compiling
        // cleanly — same "compiles, coerces to the wrong thing" trap either way.
        Err(payload) => Err(panic_payload_message(payload.as_ref())),
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("internal panic: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("internal panic: {s}")
    } else {
        "internal panic (no message available)".to_string()
    }
}

/// DDNet's server tick rate (`SERVER_TICK_SPEED`, `engine/shared/protocol.h`) — real time between
/// ticks at `speed = 1.0`.
const TICKS_PER_SECOND: f32 = 50.0;
const MIN_SPEED: f32 = 0.1;
const MAX_SPEED: f32 = 8.0;
/// How long to pause before looping back to the first trace, so a viewer watching the corpus
/// cycle sees a clear "that was the last one" beat rather than an instant, jarring cut.
const LOOP_PAUSE: Duration = Duration::from_millis(800);
/// Review round 1, finding F8: minimum real time between one failing file and the next attempt
/// (of the NEXT file — this is not a retry of the same one). See the `run` loop's own comment at
/// its one call site for the failure mode this bounds.
const ERROR_BACKOFF: Duration = Duration::from_secs(1);

/// Finds every regular file with a `.trb` extension directly inside `dir` (not recursive — every
/// real corpus directory this task points at, `~/aiddnet/data/traces/oracle-b/v1/`, is flat),
/// sorted by filename for a stable, reproducible playback order. Bounded to
/// [`MAX_TRACE_FILES`] entries so pointing this at an unexpectedly huge directory degrades
/// gracefully rather than building an unbounded `Vec` of paths.
const MAX_TRACE_FILES: usize = 100_000;

/// The name a client sees for a trace (or its `.rawmap` sibling) in `live_error` messages and
/// `ReplayStatus.file` (review round 1, finding F12): just the file name, never the full path.
/// `--replay <dir>` is an operator-chosen, possibly-absolute filesystem location (in production,
/// under `~/aiddnet/data/...`) that every authenticated client's `live_error`/`replay_status`
/// messages used to echo back verbatim — a pointless leak of the server's directory layout, since
/// `find_trace_files` never recurses, so the file name alone already uniquely identifies which
/// trace this is about to a client. Falls back to the full (`Display`) path only in the
/// practically-unreachable case of a path with no file-name component at all (e.g. `/`), which
/// none of this module's paths — all constructed by joining a directory with a `.trb`/`.rawmap`
/// file name — can actually be.
fn trace_display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Lists the `.trb` files directly inside `dir` (no recursion), sorted by filename.
///
/// **Symlinks (review round 1, finding F12).** `DirEntry::file_type` reports the on-disk entry
/// type without following a symlink, so a `.trb` that is itself a symlink previously fell through
/// `is_file()` silently — not a security issue on its own, just a confusing "why isn't my file
/// showing up" trap. Rather than leave that in place, this follows a symlink `.trb` **only when
/// its resolved target is a file inside this same directory** (canonicalized, so `../` segments
/// or a further symlink hop can't escape it) — the same "resolve only inside the configured data
/// dir" rule this codebase already applies to map resolution (`crate::live::map_resolve`'s doc
/// comment) and nowhere lets a filesystem path from outside an operator-configured directory be
/// read. A symlink pointing outside `dir`, or a dangling one, is skipped, not an error: an
/// unrelated broken symlink in the replay directory shouldn't stop `--replay` from starting.
fn find_trace_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let canonical_dir = dir.canonicalize()?;
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)?.take(MAX_TRACE_FILES) {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("trb") {
            continue;
        }
        let file_type = entry.file_type()?;
        let is_contained_symlink = file_type.is_symlink()
            && path
                .canonicalize()
                .map(|target| target.starts_with(&canonical_dir) && target.is_file())
                .unwrap_or(false);
        if file_type.is_file() || is_contained_symlink {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// `ddnet-ai web --replay <dir-or-file>` (acceptance criterion 2). A directory plays every `.trb`
/// file it contains, in sorted-filename order; a single file is just the `len() == 1` case of the
/// same list. Either way, `run`'s loop always wraps back to index 0 (after [`LOOP_PAUSE`]) once
/// the last file finishes — a single `--replay some.trb` therefore repeats that one trace
/// indefinitely rather than playing it once and going idle, matching the corpus case ("plays the
/// corpus on a loop") instead of being a special case of it. (Review round 1, finding F10: an
/// earlier revision of this comment claimed the single-file case did *not* loop; it always has —
/// this was a doc bug, not a code bug — see `docs/formats.md` §15.5 for the corrected version.)
pub struct ReplaySource {
    trace_files: Vec<PathBuf>,
    /// Directories `resolve_by_sha256` may read a real `.map` file from — never derived from any
    /// trace's own metadata (see `crate::live::map_resolve`'s doc comment).
    map_search_dirs: Vec<PathBuf>,
    map_cache: std::sync::Arc<MapCache>,
}

impl ReplaySource {
    /// `path` is either a single `.trb` file or a directory containing some. Returns an error
    /// only for "nothing to play at all" (an empty/nonexistent directory, or a path that is
    /// neither a file nor a directory) — an individual malformed trace is instead reported as a
    /// [`SourceEvent::Error`] once playback reaches it (acceptance criterion 2).
    pub fn new(
        path: &Path,
        map_search_dirs: Vec<PathBuf>,
        map_cache: std::sync::Arc<MapCache>,
    ) -> std::io::Result<Self> {
        let trace_files = if path.is_dir() {
            find_trace_files(path)?
        } else if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} is neither a file nor a directory", path.display()),
            ));
        };
        if trace_files.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no .trb files found under {}", path.display()),
            ));
        }
        Ok(ReplaySource {
            trace_files,
            map_search_dirs,
            map_cache,
        })
    }
}

/// Mutable playback state threaded through the run loop — kept as its own struct purely so
/// [`ReplaySource::run`]'s body reads as "apply a control command to `state`" rather than a wall
/// of loose local variables.
struct PlaybackState {
    playing: bool,
    speed: f32,
    /// `Some(tick)` right after a `Seek` control message, consumed (and cleared) the next time
    /// the run loop is ready to act on it.
    pending_seek: Option<u32>,
    /// Set by a `Next` control message; consumed the same way as `pending_seek`.
    skip_to_next: bool,
}

impl Default for PlaybackState {
    fn default() -> Self {
        PlaybackState {
            playing: true,
            speed: 1.0,
            pending_seek: None,
            skip_to_next: false,
        }
    }
}

impl PlaybackState {
    fn apply(&mut self, control: ReplayControl) {
        match control {
            ReplayControl::Play => self.playing = true,
            ReplayControl::Pause => self.playing = false,
            // `f32::clamp` passes a NaN `self` straight through unchanged (neither `<` nor `>`
            // against the bounds is ever true for NaN) — explicitly falling back to `MIN_SPEED`
            // for a non-finite value first, same "audit every client-supplied float" principle as
            // review round 1's finding F6, even though nothing in this task's JSON wire format
            // can currently encode a literal NaN (not valid JSON) or came up as a concrete repro
            // here.
            ReplayControl::SetSpeed(s) => {
                self.speed = if s.is_finite() { s } else { MIN_SPEED }.clamp(MIN_SPEED, MAX_SPEED)
            }
            ReplayControl::Seek(tick) => self.pending_seek = Some(tick),
            ReplayControl::Next => self.skip_to_next = true,
        }
    }
}

/// Drains every control message currently queued (never blocks) and applies them in order — the
/// last one for the same field wins, matching "the client's most recent instruction is what's
/// currently true" rather than replaying stale intermediate states.
fn drain_controls(control_rx: &mut mpsc::Receiver<ReplayControl>, state: &mut PlaybackState) {
    while let Ok(control) = control_rx.try_recv() {
        state.apply(control);
    }
}

fn to_character_state(id: u8, row: &TraceBCharacterRow) -> CharacterState {
    CharacterState {
        id,
        alive: row.ddrace_alive != 0,
        x: row.core_pos_x.round() as i32,
        y: row.core_pos_y.round() as i32,
        aim_x: row.input_target_x,
        aim_y: row.input_target_y,
        hook_state: row.core_hook_state.clamp(i8::MIN as i32, i8::MAX as i32) as i8,
        hook_x: row.core_hook_pos_x.round() as i32,
        hook_y: row.core_hook_pos_y.round() as i32,
        hooked_id: if row.core_hooked_player >= 0 {
            Some(row.core_hooked_player as u8)
        } else {
            None
        },
        weapon: row.core_active_weapon.clamp(0, 255) as u8,
        team: row.ddrace_team.clamp(0, 255) as u8,
        frozen: row.ddrace_is_in_freeze != 0,
        deep_frozen: row.ddrace_deep_frozen != 0,
        live_frozen: row.ddrace_live_frozen != 0,
    }
}

/// Derives this tick's [`GameEvent`]s by comparing `row` against `previous` (the same character's
/// row on the previous tick) — `None` for the very first tick of a trace (no previous state to
/// diff against; `died_this_tick`/`respawned_this_tick` below are transitions trace-b itself
/// already computed, so they don't need a `previous` row at all, but freeze transitions are
/// derived here from the two most recent samples for symmetry and because `is_in_freeze` has no
/// equivalent pre-computed "just changed" field of its own).
fn derive_events(id: u8, row: &TraceBCharacterRow, previous: Option<&TraceBCharacterRow>) -> Vec<GameEvent> {
    let mut events = Vec::new();
    if row.ddrace_died_this_tick != 0 {
        events.push(GameEvent::Death { id });
    }
    if row.ddrace_respawned_this_tick != 0 {
        events.push(GameEvent::Respawn { id });
    }
    if let Some(previous) = previous {
        let was_frozen = previous.ddrace_is_in_freeze != 0;
        let is_frozen = row.ddrace_is_in_freeze != 0;
        if !was_frozen && is_frozen {
            events.push(GameEvent::Freeze { id });
        } else if was_frozen && !is_frozen {
            events.push(GameEvent::Unfreeze { id });
        }
    }
    if row.core_triggered_events & COREEVENT_HOOK_ATTACH_PLAYER != 0 {
        events.push(GameEvent::HookGrab {
            id,
            target: if row.core_hooked_player >= 0 {
                Some(row.core_hooked_player as u8)
            } else {
                None
            },
        });
    }
    events
}

impl ReplaySource {
    async fn run(self: Box<Self>, events_tx: mpsc::Sender<SourceEvent>, mut control_rx: mpsc::Receiver<ReplayControl>) {
        let mut state = PlaybackState::default();
        let mut file_index = 0usize;

        loop {
            let path = &self.trace_files[file_index];
            if events_tx
                .send(SourceEvent::ReplayStatus(ReplayStatus {
                    file: trace_display_name(path),
                    tick: 0,
                    tick_count: 0,
                    playing: state.playing,
                    speed: state.speed,
                }))
                .await
                .is_err()
            {
                return;
            }

            match self.play_one_file(path, &events_tx, &mut control_rx, &mut state).await {
                Ok(()) => {}
                Err(message) => {
                    if events_tx.send(SourceEvent::Error(message)).await.is_err() {
                        return;
                    }
                    // Review round 1, finding F8: a file that fails immediately (e.g. every
                    // `real-map` trace erroring at map resolution because `--maps-dir` wasn't
                    // passed at all) used to move on to the next file with no delay whatsoever —
                    // a corpus where every file fails the same way raced through the whole
                    // directory (and then looped back to the start) about as fast as the CPU
                    // could spawn tasks and open file handles, reported as "one client received
                    // 2600 `live_error` messages in 3s". A short, fixed backoff after every
                    // failing file is enough on its own to bound that rate to roughly one error
                    // per file per `ERROR_BACKOFF` — a healthy corpus (the common case) is
                    // completely unaffected, since this only runs on the `Err` path.
                    tokio::time::sleep(ERROR_BACKOFF).await;
                }
            }

            if state.skip_to_next {
                state.skip_to_next = false;
            }
            file_index += 1;
            if file_index >= self.trace_files.len() {
                file_index = 0;
                tokio::time::sleep(LOOP_PAUSE).await;
            }
        }
    }

    /// Plays one trace file tick-by-tick, honoring pause/speed/seek/next as it goes. Returns
    /// `Err(message)` for a malformed trace (acceptance criterion 2: "a malformed trace gives an
    /// error event, not a panic") rather than propagating a panic — the caller turns this into a
    /// [`SourceEvent::Error`] and moves on to the next file.
    async fn play_one_file(
        &self,
        path: &Path,
        events_tx: &mpsc::Sender<SourceEvent>,
        control_rx: &mut mpsc::Receiver<ReplayControl>,
        state: &mut PlaybackState,
    ) -> Result<(), String> {
        let mut reader = catch_panic_result(std::panic::AssertUnwindSafe(|| TraceBReader::open(path)))
            .map_err(|e| format!("{}: {e}", trace_display_name(path)))?;
        let header = reader.header().clone();

        let map_name = map_display_name(path, &header.metadata);
        let scene = if let Some(cached) = self.map_cache.get(&header.metadata.map_sha256) {
            cached
        } else {
            // Review round 1, finding F11: resolving a map means reading a whole `.map` (or
            // `.rawmap`) file — up to `map_resolve::MAX_MAP_FILE_BYTES` — and hashing/parsing it,
            // which used to run synchronously on this task's own async runtime worker thread
            // (shared with every other connection's `select!` loop, since this whole source runs
            // inside the same multi-threaded Tokio runtime). `spawn_blocking` moves it onto the
            // dedicated blocking-task pool instead, so a slow map read never competes with
            // anyone else's scheduling. This also *replaces* `catch_panic_result` at these two
            // call sites (not both): a panic inside a `spawn_blocking` closure becomes a
            // `JoinError` here, never an unwind into this task, which is the exact same isolation
            // guarantee finding F1 asked for — just delivered by a mechanism this task already
            // needs for an unrelated reason, rather than by layering two wrappers doing
            // overlapping jobs.
            let resolved = if header.metadata.mode == "real-map" {
                let search_dirs = self.map_search_dirs.clone();
                let hint = header.metadata.real_map_path.clone().unwrap_or_default();
                let sha256 = header.metadata.map_sha256;
                match tokio::task::spawn_blocking(move || {
                    map_resolve::resolve_by_sha256(&search_dirs, &hint, sha256).map(|(_, scene)| scene)
                })
                .await
                {
                    Ok(Ok(scene)) => scene,
                    Ok(Err(e)) => return Err(format!("{}: resolving real map: {e}", trace_display_name(path))),
                    Err(_join_error) => {
                        return Err(format!(
                            "{}: resolving real map: panicked while resolving",
                            trace_display_name(path)
                        ));
                    }
                }
            } else {
                let path_owned = path.to_path_buf();
                match tokio::task::spawn_blocking(move || resolve_synthetic_map(&path_owned)).await {
                    Ok(Ok(scene)) => scene,
                    Ok(Err(e)) => return Err(format!("{}: resolving synthetic map: {e}", trace_display_name(path))),
                    Err(_join_error) => {
                        return Err(format!(
                            "{}: resolving synthetic map: panicked while resolving",
                            trace_display_name(path)
                        ));
                    }
                }
            };
            self.map_cache.insert(header.metadata.map_sha256, resolved)
        };

        events_tx
            .send(SourceEvent::MapChanged(MapMeta {
                sha256: header.metadata.map_sha256,
                name: map_name,
                width: scene.width,
                height: scene.height,
            }))
            .await
            .map_err(|_| "hub closed".to_string())?;

        let players: Vec<PlayerMeta> = header
            .character_ids
            .iter()
            .enumerate()
            .map(|(slot, _id)| PlayerMeta {
                id: slot as u8,
                // Oracle B has no client/network layer, so there is no real display name to
                // report here (see `crate::live::source`'s `PlayerMeta` field docs) — a
                // synthesized, clearly-a-placeholder Russian label instead of inventing a fake
                // one that could be mistaken for a real player's name.
                name: format!("Игрок {slot}"),
                team: 0, // refined below, once the first tick's real team value is known
            })
            .collect();
        events_tx
            .send(SourceEvent::Players(players))
            .await
            .map_err(|_| "hub closed".to_string())?;

        let mut previous_rows: Vec<Option<TraceBCharacterRow>> = vec![None; header.character_ids.len()];
        let mut sent_team_update = false;
        // Every path through the loop below assigns this before it's next read (either the seek
        // branch or the normal tick-read branch) — the compiler can't see that across a `loop`,
        // so this initial value is a deliberate, harmless dead store, not a bug.
        #[allow(unused_assignments)]
        let mut current_tick: u32 = 0;

        loop {
            drain_controls(control_rx, state);
            if state.skip_to_next {
                return Ok(());
            }
            if let Some(seek_tick) = state.pending_seek.take() {
                // No random-access index exists (trace-b's per-tick byte length varies with
                // `entity_count`, see `trace_b`'s doc comment), so seeking means: reopen the file
                // and re-read forward from tick 0, discarding every tick before the target
                // instead of emitting it — but still folding each discarded tick's rows into
                // `previous_rows`, so the first tick actually shown after a seek has a correct
                // "was this character already frozen" baseline for `derive_events` rather than
                // spuriously reporting a Freeze/Unfreeze edge that didn't really just happen.
                reader = catch_panic_result(std::panic::AssertUnwindSafe(|| TraceBReader::open(path)))
                    .map_err(|e| format!("{}: {e}", trace_display_name(path)))?;
                previous_rows = vec![None; header.character_ids.len()];
                current_tick = 0;
                while current_tick < seek_tick {
                    match catch_panic_result(std::panic::AssertUnwindSafe(|| reader.next_tick()))
                        .map_err(|e| format!("{}: {e}", trace_display_name(path)))?
                    {
                        Some(skipped) => {
                            for (slot, row) in skipped.characters.iter().enumerate() {
                                previous_rows[slot] = Some(*row);
                            }
                            current_tick = skipped.game_tick as u32;
                        }
                        None => break, // seek target is past the end of this trace; stop here
                    }
                }
            }
            if !state.playing {
                // Still report status while paused (throttled to a sane rate by the hub, see
                // `crate::live::hub::STATUS_THROTTLE`) — otherwise a client has no way to learn
                // "we're paused at tick X" at all, since the tick-read path below (the only other
                // place a `ReplayStatus` is sent) never runs while paused.
                if events_tx
                    .send(SourceEvent::ReplayStatus(ReplayStatus {
                        file: trace_display_name(path),
                        tick: current_tick,
                        tick_count: header.tick_count,
                        playing: false,
                        speed: state.speed,
                    }))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }

            let tick_data: TraceBTick = match catch_panic_result(std::panic::AssertUnwindSafe(|| reader.next_tick()))
                .map_err(|e| format!("{}: {e}", trace_display_name(path)))?
            {
                Some(t) => t,
                None => return Ok(()),
            };

            let mut characters = Vec::with_capacity(tick_data.characters.len());
            let mut all_events = Vec::new();
            for (slot, row) in tick_data.characters.iter().enumerate() {
                let id = slot as u8;
                characters.push(to_character_state(id, row));
                all_events.extend(derive_events(id, row, previous_rows[slot].as_ref()));
                previous_rows[slot] = Some(*row);
            }

            if !sent_team_update && tick_data.characters.iter().any(|r| r.ddrace_team != 0) {
                sent_team_update = true;
                let players: Vec<PlayerMeta> = tick_data
                    .characters
                    .iter()
                    .enumerate()
                    .map(|(slot, row)| PlayerMeta {
                        id: slot as u8,
                        name: format!("Игрок {slot}"),
                        team: row.ddrace_team.clamp(0, 255) as u8,
                    })
                    .collect();
                if events_tx.send(SourceEvent::Players(players)).await.is_err() {
                    return Ok(());
                }
            }

            let tick = tick_data.game_tick as u32;
            current_tick = tick; // kept current so a later `Seek` starts skipping from the right place
            if events_tx
                .send(SourceEvent::Frame(WorldFrame { tick, characters }))
                .await
                .is_err()
            {
                return Ok(());
            }
            if !all_events.is_empty()
                && events_tx
                    .send(SourceEvent::Events {
                        tick,
                        events: all_events,
                    })
                    .await
                    .is_err()
            {
                return Ok(());
            }
            if events_tx
                .send(SourceEvent::ReplayStatus(ReplayStatus {
                    file: trace_display_name(path),
                    tick: current_tick,
                    tick_count: header.tick_count,
                    playing: state.playing,
                    speed: state.speed,
                }))
                .await
                .is_err()
            {
                return Ok(());
            }

            // `state.speed` is clamped to `[MIN_SPEED, MAX_SPEED]` by `PlaybackState::apply`, so
            // this is always in-range in practice — `try_from_secs_f32` (not the panicking
            // `from_secs_f32`) anyway, same "audit every float-to-duration conversion" principle
            // as the WS `sub{live}` fix (review round 1, finding F6): a future change to that
            // clamp, or to `TICKS_PER_SECOND`, must not turn this into a panic that kills the
            // whole replay task.
            let real_seconds_per_tick = 1.0 / (TICKS_PER_SECOND * state.speed);
            let sleep_duration =
                Duration::try_from_secs_f32(real_seconds_per_tick.max(0.0)).unwrap_or(Duration::from_millis(20));
            tokio::time::sleep(sleep_duration).await;
        }
    }
}

/// A human-readable map label for the `map` WS message: the trace filename's map-name segment
/// when it follows this corpus's own `realmap_<Name>__seed<N>.trb`/`recipe_<name>_seed<N>.trb`
/// naming convention, falling back to the bare filename stem otherwise — cosmetic only, never
/// used for any filesystem lookup (that's `map_sha256`'s job, checked independently).
fn map_display_name(path: &Path, metadata: &trace_b::TraceBMetadata) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("map");
    for prefix in ["realmap_", "recipe_"] {
        if let Some(rest) = stem.strip_prefix(prefix) {
            if let Some((name, _seed)) = rest.split_once("__seed") {
                return name.to_string();
            }
            if let Some((name, _seed)) = rest.rsplit_once("_seed") {
                return name.to_string();
            }
            return rest.to_string();
        }
    }
    let _ = metadata; // reserved for a future fallback that reads a name out of metadata
    stem.to_string()
}

/// Resolves the map for a `mode == "rawmap-scenario"` trace (review round 1, finding F2: the
/// previous version only tried the `.rawmap` sibling below, which none of this corpus's 160
/// `recipe_*` traces actually have — every one of them errored out forever). Two strategies, in
/// order:
///
/// 1. A `.rawmap` file sitting alongside `trace_path` (same directory, same stem, `.rawmap`
///    extension) — kept as the first attempt for a future/different trace source that *does*
///    write one (some of this task's own test fixtures do, via `--emit-scenario-v3`-style
///    tooling described in `docs/formats.md` §12.6); this only ever reads a path *derived from
///    the replay directory itself* (fixed, operator-configured, never request/trace-content-
///    derived), never a filename taken from untrusted metadata.
/// 2. Rebuilding the map from `ddai_trace::synthetic` directly, using the recipe name parsed out
///    of the trace's OWN filename (`recipe_<name>_seed<N>.trb`, this corpus's fixed naming
///    convention — the only place the recipe name exists at all: trace-b's `rawmap-scenario`
///    metadata has no recipe-name field of its own, per `docs/formats.md` §11.1, and this
///    corpus's `recipe_*` traces have no companion scenario/rawmap file to read one from either,
///    confirmed by listing the corpus directly). **No sha256 cross-check is possible for this
///    path** — unlike real-map mode, `map_sha256` here is the hash of a `.map` file that only
///    ever existed transiently inside the C++ harness (produced by `raw2map` from the recipe,
///    then hashed and discarded — see `docs/formats.md` §11.1/§12.1.1), never persisted anywhere
///    this Rust code could re-derive without reimplementing `raw2map`'s real-`.map`-datafile
///    writer wholesale. The integrity guarantee here is narrower, and different in kind, from the
///    real-map path's: the recipe NAME must match one of [`ddai_trace::synthetic::RECIPES`] (a
///    fixed set of 4 pure, parameterless, non-attacker-controlled builders baked into this same
///    binary), not an arbitrary byte match — there is no untrusted content to verify against,
///    only a closed enum to accept-or-reject.
fn resolve_synthetic_map(trace_path: &Path) -> Result<super::scene::MapScene, String> {
    let rawmap_path = trace_path.with_extension("rawmap");
    if let Ok(bytes) = std::fs::read(&rawmap_path) {
        let data =
            ddai_trace::rawmap::read(&bytes).map_err(|e| format!("{}: {e}", trace_display_name(&rawmap_path)))?;
        return Ok(super::scene::MapScene::build(&data));
    }

    let recipe_name = recipe_name_from_filename(trace_path).ok_or_else(|| {
        format!(
            "{}: no .rawmap sibling and filename doesn't match recipe_<name>_seed<N>",
            trace_display_name(trace_path)
        )
    })?;
    let map_data = ddai_trace::synthetic::build(&recipe_name).ok_or_else(|| {
        format!(
            "{}: {recipe_name:?} is not a known synthetic recipe ({:?})",
            trace_display_name(trace_path),
            ddai_trace::synthetic::RECIPES
        )
    })?;
    Ok(super::scene::MapScene::build(&map_data))
}

/// Parses `recipe_<name>_seed<N>.trb`'s `<name>` and normalizes it to
/// [`ddai_trace::synthetic::RECIPES`]'s own spelling (`_` -> `-`: this corpus's filenames spell
/// the fourth recipe `tele_speedup`, `RECIPES` spells it `tele-speedup` — the same normalization
/// `map_display_name`'s own doc comment does not need, since that function only ever shows the
/// name to a human, never looks it up).
fn recipe_name_from_filename(trace_path: &Path) -> Option<String> {
    let stem = trace_path.file_stem()?.to_str()?;
    let rest = stem.strip_prefix("recipe_")?;
    let (name, _seed) = rest.rsplit_once("_seed")?;
    Some(name.replace('_', "-"))
}

impl FrameSource for ReplaySource {
    fn spawn(
        self: Box<Self>,
        events_tx: mpsc::Sender<SourceEvent>,
        control_rx: mpsc::Receiver<ReplayControl>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move { self.run(events_tx, control_rx).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------------------------
    // `catch_panic_result` (review round 1, finding F1's defense-in-depth)
    // -----------------------------------------------------------------------------------------

    #[test]
    fn catch_panic_result_passes_through_ok() {
        let result: Result<i32, String> = catch_panic_result(|| Ok::<i32, std::io::Error>(42));
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn catch_panic_result_passes_through_a_normal_error() {
        let result: Result<i32, String> = catch_panic_result(|| Err::<i32, _>(std::io::Error::other("boom")));
        assert_eq!(result, Err("boom".to_string()));
    }

    #[test]
    fn catch_panic_result_turns_a_panic_into_an_err_instead_of_unwinding() {
        // Suppress the default panic hook's stderr dump for this one deliberately-panicking
        // test, so a normal `cargo test` run doesn't print a scary-looking backtrace for
        // something that is, in fact, the expected/covered behavior.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result: Result<i32, String> =
            catch_panic_result(|| -> Result<i32, std::io::Error> { panic!("simulated parser bug") });
        std::panic::set_hook(previous_hook);

        let error = result.expect_err("a panic must become an Err, not unwind past this call");
        assert!(error.contains("simulated parser bug"), "{error}");
    }

    #[test]
    fn new_rejects_a_nonexistent_path() {
        let result = ReplaySource::new(
            Path::new("/does/not/exist/at/all"),
            Vec::new(),
            std::sync::Arc::new(MapCache::new()),
        );
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_an_empty_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let result = ReplaySource::new(tmp.path(), Vec::new(), std::sync::Arc::new(MapCache::new()));
        assert!(result.is_err());
    }

    #[test]
    fn new_accepts_a_single_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let file = tmp.path().join("one.trb");
        std::fs::write(&file, b"not a real trace, just needs to exist").unwrap();
        let source = ReplaySource::new(&file, Vec::new(), std::sync::Arc::new(MapCache::new())).expect("should accept");
        assert_eq!(source.trace_files, vec![file]);
    }

    #[test]
    fn new_finds_and_sorts_trb_files_in_a_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        for name in ["b.trb", "a.trb", "c.scn", "a.rawmap"] {
            std::fs::write(tmp.path().join(name), b"x").unwrap();
        }
        let source =
            ReplaySource::new(tmp.path(), Vec::new(), std::sync::Arc::new(MapCache::new())).expect("should find files");
        assert_eq!(
            source.trace_files,
            vec![tmp.path().join("a.trb"), tmp.path().join("b.trb")]
        );
    }

    /// Regression test for review round 1, finding F12: a `.trb` symlink whose target resolves
    /// *inside* the replay directory is followed, not silently skipped.
    #[test]
    #[cfg(unix)]
    fn a_symlinked_trb_file_pointing_inside_the_directory_is_included() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real.trb");
        std::fs::write(&real, b"x").unwrap();
        let link = tmp.path().join("link.trb");
        std::os::unix::fs::symlink(&real, &link).expect("create symlink");

        let source =
            ReplaySource::new(tmp.path(), Vec::new(), std::sync::Arc::new(MapCache::new())).expect("should find files");
        assert_eq!(source.trace_files, vec![link, real]);
    }

    /// Regression test for review round 1, finding F12: a `.trb` symlink whose target resolves
    /// *outside* the replay directory must not be followed — `--replay some-dir` must never end
    /// up reading a file the operator didn't put inside `some-dir`.
    #[test]
    #[cfg(unix)]
    fn a_symlinked_trb_file_pointing_outside_the_directory_is_skipped() {
        let outside = tempfile::tempdir().expect("outside tempdir");
        let target = outside.path().join("secret.trb");
        std::fs::write(&target, b"x").unwrap();

        let tmp = tempfile::tempdir().expect("tempdir");
        let link = tmp.path().join("link.trb");
        std::os::unix::fs::symlink(&target, &link).expect("create symlink");
        // A directory with only an escaping symlink in it has, from `find_trace_files`'s point of
        // view, zero usable trace files — `ReplaySource::new` must reject it exactly like a truly
        // empty directory (see `new_rejects_an_empty_directory` above), not silently follow the
        // symlink out.
        let result = ReplaySource::new(tmp.path(), Vec::new(), std::sync::Arc::new(MapCache::new()));
        assert!(
            result.is_err(),
            "a symlink escaping the replay dir must not be followed, so this directory has no usable trace files"
        );
    }

    /// Regression test for review round 1, finding F12: a dangling symlink (nothing at all where
    /// the target points) is skipped rather than erroring — an unrelated broken symlink in the
    /// replay directory shouldn't stop the whole source from starting.
    #[test]
    #[cfg(unix)]
    fn a_dangling_trb_symlink_is_skipped_without_erroring() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real.trb");
        std::fs::write(&real, b"x").unwrap();
        let dangling = tmp.path().join("dangling.trb");
        std::os::unix::fs::symlink(tmp.path().join("does-not-exist"), &dangling).expect("create symlink");

        let source =
            ReplaySource::new(tmp.path(), Vec::new(), std::sync::Arc::new(MapCache::new())).expect("should find files");
        assert_eq!(source.trace_files, vec![real]);
    }

    /// Regression test for review round 1, finding F8: a corpus where every file fails (here:
    /// several syntactically-invalid `.trb` files, but a directory of otherwise-valid traces with
    /// no `--maps-dir` configured at all would fail the exact same way, at map resolution instead
    /// of at parsing) must not flood the event channel — the finding's own repro measured 2600
    /// `live_error` messages in 3s before this fix.
    #[tokio::test]
    async fn a_corpus_of_entirely_broken_files_does_not_flood_error_events() {
        let tmp = tempfile::tempdir().expect("tempdir");
        for i in 0..5 {
            std::fs::write(tmp.path().join(format!("broken_{i}.trb")), b"not a trace-b file").unwrap();
        }
        let source =
            ReplaySource::new(tmp.path(), Vec::new(), std::sync::Arc::new(MapCache::new())).expect("construct");

        let (events_tx, mut events_rx) = mpsc::channel(64);
        let (_control_tx, control_rx) = mpsc::channel(4);
        let task = Box::new(source).spawn(events_tx, control_rx);

        let mut error_count = 0u32;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, events_rx.recv()).await {
                Ok(Some(SourceEvent::Error(_))) => error_count += 1,
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => break, // timeout elapsed
            }
        }
        task.abort();

        // With `ERROR_BACKOFF` (1s) between failing files, 3 seconds bounds this to roughly 3-5
        // errors, not hundreds/thousands — generous slack for scheduling jitter, but nowhere near
        // the pre-fix "as fast as the CPU can spawn tasks" rate.
        assert!(
            error_count <= 10,
            "expected a bounded error rate, got {error_count} errors in 3s"
        );
        assert!(error_count >= 1, "should have gotten at least one error at all");
    }

    #[test]
    fn map_display_name_strips_the_corpus_naming_convention() {
        assert_eq!(
            map_display_name(
                Path::new("/x/realmap_BlmapChill__seed10001.trb"),
                &trace_b::TraceBMetadata {
                    map_sha256: [0; 32],
                    mode: "real-map".to_string(),
                    real_map_path: None,
                },
            ),
            "BlmapChill"
        );
        assert_eq!(
            map_display_name(
                Path::new("/x/recipe_arena_seed20001.trb"),
                &trace_b::TraceBMetadata {
                    map_sha256: [0; 32],
                    mode: "rawmap-scenario".to_string(),
                    real_map_path: None,
                },
            ),
            "arena"
        );
    }

    #[test]
    fn derive_events_reports_freeze_transition() {
        let mut previous = zero_row();
        previous.ddrace_is_in_freeze = 0;
        let mut current = zero_row();
        current.ddrace_is_in_freeze = 1;
        let events = derive_events(3, &current, Some(&previous));
        assert_eq!(events, vec![GameEvent::Freeze { id: 3 }]);
    }

    #[test]
    fn derive_events_reports_unfreeze_transition() {
        let mut previous = zero_row();
        previous.ddrace_is_in_freeze = 1;
        let current = zero_row();
        let events = derive_events(3, &current, Some(&previous));
        assert_eq!(events, vec![GameEvent::Unfreeze { id: 3 }]);
    }

    #[test]
    fn derive_events_reports_death_and_respawn_directly_from_trace_fields() {
        let mut died = zero_row();
        died.ddrace_died_this_tick = 1;
        assert_eq!(derive_events(1, &died, None), vec![GameEvent::Death { id: 1 }]);

        let mut respawned = zero_row();
        respawned.ddrace_respawned_this_tick = 1;
        assert_eq!(derive_events(1, &respawned, None), vec![GameEvent::Respawn { id: 1 }]);
    }

    #[test]
    fn derive_events_reports_hook_grab_on_the_player_attach_bit() {
        let mut row = zero_row();
        row.core_triggered_events = COREEVENT_HOOK_ATTACH_PLAYER;
        row.core_hooked_player = 4;
        assert_eq!(
            derive_events(0, &row, None),
            vec![GameEvent::HookGrab { id: 0, target: Some(4) }]
        );
    }

    #[test]
    fn derive_events_is_empty_when_nothing_changed() {
        let row = zero_row();
        assert!(derive_events(0, &row, Some(&row)).is_empty());
    }

    fn zero_row() -> TraceBCharacterRow {
        TraceBCharacterRow {
            input_target_x: 0,
            input_target_y: 0,
            core_pos_x: 0.0,
            core_pos_y: 0.0,
            core_hook_pos_x: 0.0,
            core_hook_pos_y: 0.0,
            core_hook_state: 0,
            core_hooked_player: -1,
            core_active_weapon: 0,
            core_triggered_events: 0,
            ddrace_alive: 1,
            ddrace_died_this_tick: 0,
            ddrace_respawned_this_tick: 0,
            ddrace_is_in_freeze: 0,
            ddrace_deep_frozen: 0,
            ddrace_live_frozen: 0,
            ddrace_team: 0,
        }
    }
}
