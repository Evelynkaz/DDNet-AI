//! `ddnet-ai rec inspect|reconstruct|anonymize` (task 8.4a acceptance criterion 3/2): offline
//! tooling over a rec v1 recording (`ddai-recorder`) — no network, no live server.

use clap::{Args, Subcommand};
use ddai_recorder::anonymize::Anonymizer;
use ddai_recorder::format::{Frame, RecordedGameMessage};
use ddai_recorder::reader::{RecordingReader, verify_whole_file_sha256};
use ddai_recorder::reconstruct::{self, Confidence};
use ddai_recorder::writer::RecordingWriter;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Args)]
pub struct RecArgs {
    #[command(subcommand)]
    pub command: RecCommand,
}

#[derive(Debug, Subcommand)]
pub enum RecCommand {
    /// Prints a summary of a recording: header, frame counts, tick range, players seen, and
    /// (with `--verify`) whether the whole-file sha256 sidecar still matches.
    Inspect {
        recording: PathBuf,
        /// Also recomputes and checks the whole-file sha256 against `<recording>.sha256`.
        #[arg(long)]
        verify: bool,
    },
    /// Task acceptance criterion 3: reconstructs every player's trajectory and estimated inputs,
    /// writing them as JSON to `--out`. With `--validate`, additionally compares the named
    /// player's reconstructed inputs against a `ddnet-ai play --input-log` ground-truth file and
    /// prints a per-field accuracy report instead of (or alongside) writing `--out`.
    Reconstruct {
        recording: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Ground-truth JSON-lines file from `ddnet-ai play --input-log` (or `ddnet-ai record
        /// --input-log`) — enables the accuracy report.
        #[arg(long)]
        validate: Option<PathBuf>,
        /// Which recorded player to validate against `--validate`'s ground truth (its
        /// `PlayerInfo`/`Character` client id in the recording). Required when `--validate` is
        /// given and the recording has more than one `(client_id, stint)` track, unless
        /// `--player-name` is given instead. Review round 1, finding F5: a client id slot can be
        /// occupied by more than one real person across a recording (a stint each) — with no
        /// `--player-name` to disambiguate, this picks that id's *most recently active* stint.
        #[arg(long)]
        client_id: Option<i32>,
        /// Alternative to `--client-id`: the recorded player's `Cl_StartInfo` nick (e.g. the name
        /// a `ddnet-ai play --brain random-scripted` validation bot joined under) — resolved to an
        /// exact `(client_id, stint)` track by scanning the recording's own `ClientInfo` (finding
        /// F5): the *most recently active* stint that ever carried this name, matching
        /// `load_input_log`'s own last-epoch-only ground-truth semantics. Convenient for tooling
        /// (`tools/e2e/record.sh`) that knows the bot's name but not its server-assigned id ahead
        /// of time.
        #[arg(long)]
        player_name: Option<String>,
    },
    /// Task acceptance criterion 2's `--anonymize` export: writes a copy of `recording` with
    /// every nickname/clan replaced by a stable per-recording id and every chat message's text
    /// redacted — safe to share or commit (still never actually committed unless it is also
    /// synthetic, per `CLAUDE.md`).
    Anonymize {
        recording: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

pub fn run(args: RecArgs) -> ExitCode {
    match args.command {
        RecCommand::Inspect { recording, verify } => inspect(&recording, verify),
        RecCommand::Reconstruct {
            recording,
            out,
            validate,
            client_id,
            player_name,
        } => reconstruct_cmd(
            &recording,
            out.as_deref(),
            validate.as_deref(),
            client_id,
            player_name.as_deref(),
        ),
        RecCommand::Anonymize { recording, out } => anonymize_cmd(&recording, &out),
    }
}

fn read_all_frames(path: &std::path::Path) -> Result<(ddai_recorder::format::Header, Vec<Frame>), String> {
    let mut reader = RecordingReader::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let header = reader.header().clone();
    let mut frames = Vec::new();
    loop {
        match reader.next_frame() {
            Ok(Some(frame)) => frames.push(frame),
            Ok(None) => break,
            Err(e) => return Err(format!("failed to read {}: {e}", path.display())),
        }
    }
    Ok((header, frames))
}

fn inspect(path: &std::path::Path, verify: bool) -> ExitCode {
    let (header, frames) = match read_all_frames(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    println!("rec v1: {}", path.display());
    println!("  server_address:  {}", header.server_address);
    println!("  map_name:        {}", header.map_name);
    println!(
        "  map_sha256:      {}",
        header.map_sha256.iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
    println!("  client_version:  {}", header.client_version);
    println!("  start_time:      {} ms since epoch", header.start_time_unix_ms);
    println!("  observer_nick:   {}", header.observer_nick);

    let mut snapshot_count = 0u64;
    let mut game_event_counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut min_tick = i32::MAX;
    let mut max_tick = i32::MIN;
    for frame in &frames {
        match frame {
            Frame::Snapshot { tick, .. } => {
                snapshot_count += 1;
                min_tick = min_tick.min(*tick);
                max_tick = max_tick.max(*tick);
            }
            Frame::GameEvent { message, .. } => {
                let kind = match message {
                    RecordedGameMessage::Kill { .. } => "kill",
                    RecordedGameMessage::Broadcast { .. } => "broadcast",
                    RecordedGameMessage::Chat { .. } => "chat",
                    RecordedGameMessage::Tuning(_) => "tuning",
                    RecordedGameMessage::Other { .. } => "other",
                };
                *game_event_counts.entry(kind).or_insert(0) += 1;
            }
        }
    }

    println!("  snapshot frames: {snapshot_count}");
    if snapshot_count > 0 {
        println!("  tick range:      {min_tick}..={max_tick}");
    }
    println!("  game events:     {game_event_counts:?}");
    // Review round 1, finding F5: keyed by `(client_id, stint)`, not bare `client_id` — a slot
    // reused by a different person must not silently collapse into one name here either. One
    // single canonical pass over `frames` (`stint_names`), not one scan per player per frame.
    let players: BTreeMap<(i32, u32), String> = reconstruct::stint_names(&frames)
        .into_iter()
        .map(|(key, info)| (key, info.name))
        .collect();
    println!("  players seen:    {players:?}");

    if verify {
        match verify_whole_file_sha256(path) {
            Ok(true) => println!("  sha256 sidecar:  OK"),
            Ok(false) => {
                println!("  sha256 sidecar:  MISMATCH");
                return ExitCode::FAILURE;
            }
            Err(e) => {
                println!("  sha256 sidecar:  error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    ExitCode::SUCCESS
}

fn confidence_json(c: Confidence) -> &'static str {
    match c {
        Confidence::Exact => "exact",
        Confidence::Estimated => "estimated",
    }
}

fn reconstruct_cmd(
    path: &std::path::Path,
    out: Option<&std::path::Path>,
    validate: Option<&std::path::Path>,
    client_id: Option<i32>,
    player_name: Option<&str>,
) -> ExitCode {
    let (_header, frames) = match read_all_frames(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let players = reconstruct::reconstruct(&frames);

    // Review round 1, finding F5: `--player-name` resolves to an exact `(client_id, stint)` pair
    // (the most recently active one under that name, matching `load_input_log`'s own
    // last-epoch-only semantics on the ground-truth side) rather than a bare client id — a plain
    // `--client-id` with no name carries no stint information of its own, so `run_validation`
    // itself picks that id's most recently active stint when more than one exists for it.
    let (client_id, stint): (Option<i32>, Option<u32>) = match (client_id, player_name) {
        (_, Some(name)) => match reconstruct::resolve_name_to_stints(&frames, name).last() {
            Some(&(id, stint)) => (Some(id), Some(stint)),
            None => {
                eprintln!("no player named {name:?} found in this recording");
                return ExitCode::FAILURE;
            }
        },
        (Some(id), None) => (Some(id), None),
        (None, None) => (None, None),
    };

    if let Some(out) = out {
        let json = serde_json::json!({
            "players": players.iter().map(|p| serde_json::json!({
                "client_id": p.client_id,
                "stint": p.stint,
                "confidence": {
                    "direction": confidence_json(p.confidence.direction),
                    "aim": confidence_json(p.confidence.aim),
                    "jump": confidence_json(p.confidence.jump),
                    "hook": confidence_json(p.confidence.hook),
                    "fire": confidence_json(p.confidence.fire),
                },
                "trajectory": p.trajectory.iter().map(|s| serde_json::json!({
                    "tick": s.tick, "x": s.x, "y": s.y, "vel_x": s.vel_x, "vel_y": s.vel_y,
                    "hook_state": s.hook_state, "hooked_player": s.hooked_player,
                    "weapon": s.weapon, "direction": s.direction, "freeze": s.freeze,
                    "in_freeze_tile": s.in_freeze_tile, "aim_x": s.aim_x, "aim_y": s.aim_y,
                    "jumped": s.jumped,
                })).collect::<Vec<_>>(),
                "inputs": p.inputs.iter().map(|i| serde_json::json!({
                    "tick": i.tick, "direction": i.direction, "aim_x": i.aim_x, "aim_y": i.aim_y,
                    "jump": i.jump, "hook": i.hook,
                })).collect::<Vec<_>>(),
                "fire_ticks": p.fire_ticks,
            })).collect::<Vec<_>>(),
        });
        if let Err(e) = std::fs::write(out, serde_json::to_string_pretty(&json).unwrap_or_default()) {
            eprintln!("failed to write {}: {e}", out.display());
            return ExitCode::FAILURE;
        }
        println!("wrote {} players' reconstruction to {}", players.len(), out.display());
    }

    if let Some(validate_path) = validate {
        return run_validation(&players, validate_path, client_id, stint);
    }

    ExitCode::SUCCESS
}

#[derive(Debug, serde::Deserialize)]
struct LoggedInput {
    tick: i32,
    direction: i32,
    target_x: i32,
    target_y: i32,
    jump: i32,
    fire: i32,
    hook: i32,
}

/// How far a later entry's `tick` must drop below the running maximum seen so far to count as a
/// fresh connection epoch's start, not ordinary same-epoch jitter — see [`load_input_log`]'s doc
/// comment. Comfortably above any same-epoch reordering this project's own bounded/droppable
/// event channels could produce, comfortably below "the server's own tick counter reset to near 0
/// after a restart" (which this exists to detect).
const EPOCH_RESET_SLACK: i32 = 50;

/// Loads `ddnet-ai play`/`record --input-log`'s JSON lines, keeping only the **last connection
/// epoch**. `tick` is `NETMSG_INPUT`'s `pred_tick` — a small integer close to the *server's* own
/// tick counter, not a value unique to one TCP-like connection — so a log spanning a reconnect
/// (this task's own e2e test restarts the local server mid-session, task acceptance criterion 4)
/// can have **two different sends, from two different connections, land on the same numeric
/// tick**: the server process restarting resets its own tick counter back down near zero, and the
/// reconnecting client's own predicted-tick bootstrap follows it right back down. A validator that
/// blindly sorted the whole file by `tick` would silently interleave two unrelated connections'
/// entries — a real bug this project's own local e2e run caught (see the crate's BUILD REPORT):
/// found as a `rising_edges` output that made no physical sense (two "presses" one tick apart)
/// until traced back to exactly this. `crate::record_cmd` always starts a fresh recording segment
/// on every `MapLoaded` (including the one after a reconnect), so a recording's *last* segment is
/// already exactly the last connection epoch — this keeps the same slice on the ground-truth side.
fn load_input_log(path: &std::path::Path) -> Result<Vec<LoggedInput>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut all = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: LoggedInput =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), line_no + 1))?;
        all.push(entry);
    }
    // File order is chronological (each line is appended live, in the order it was sent) — find
    // where the *last* epoch starts: the last point a tick drops more than `EPOCH_RESET_SLACK`
    // below the running maximum seen so far.
    let mut running_max = i32::MIN;
    let mut last_epoch_start = 0usize;
    for (i, e) in all.iter().enumerate() {
        if e.tick < running_max.saturating_sub(EPOCH_RESET_SLACK) {
            // A fresh epoch: rebase `running_max` on *this* entry, not the previous epoch's
            // (otherwise every following entry looks like "a drop" against the stale old
            // maximum too, and `last_epoch_start` would keep creeping forward one entry at a
            // time instead of settling on the true start of the last epoch).
            last_epoch_start = i;
            running_max = e.tick;
        } else {
            running_max = running_max.max(e.tick);
        }
    }
    let mut last_epoch = all.split_off(last_epoch_start);
    last_epoch.sort_by_key(|e| e.tick);
    Ok(last_epoch)
}

/// Rising edges of a 0/1 field in `log`, sorted by tick — the ground-truth "event" ticks a
/// scripted brain's held/pulsed input produces (see `ddnet-ai`'s `play_cmd::random_scripted_input`
/// and `EstimatedInput::jump`'s doc comment for why this, not the raw per-tick value, is the fair
/// comparison unit for an edge-triggered field).
fn rising_edges(log: &[LoggedInput], field: impl Fn(&LoggedInput) -> i32) -> Vec<i32> {
    let mut out = Vec::new();
    let mut prev = 0;
    for entry in log {
        let v = field(entry);
        if v != 0 && prev == 0 {
            out.push(entry.tick);
        }
        prev = v;
    }
    out
}

/// Same matching rule `set_accuracy` uses (a truth/reconstructed pair "matches" within
/// `tolerance` ticks of each other), but returns raw counts instead of fractions — review round 3,
/// finding F20 step (3) needs raw counts to correctly *pool* several phase-2 sub-runs by summing
/// counts (not averaging already-computed fractions, which would weight a small-sample sub-run
/// equally to a large one). Returns `(matched_truth, truth.len(), matched_recon,
/// reconstructed.len())`.
fn matched_counts(truth: &[i32], reconstructed: &[i32], tolerance: i32) -> (usize, usize, usize, usize) {
    let matched_truth = truth
        .iter()
        .filter(|t| reconstructed.iter().any(|r| (*r - **t).abs() <= tolerance))
        .count();
    let matched_recon = reconstructed
        .iter()
        .filter(|r| truth.iter().any(|t| (**r - *t).abs() <= tolerance))
        .count();
    (matched_truth, truth.len(), matched_recon, reconstructed.len())
}

/// Fraction of `truth`'s ticks that have a matching reconstructed event within `tolerance` ticks
/// of each other, symmetrized (also counts how many reconstructed events had no nearby truth
/// event) — reported as (recall, precision). A vacuous 1.0 (not 0.0/NaN) when a side is empty,
/// same convention as `Bucket::frac`'s own `NAN`-on-empty everywhere else in this file does *not*
/// share — deliberately: an empty truth set trivially has perfect recall (nothing was missed) and
/// an empty reconstructed set trivially has perfect precision (nothing false was reported).
fn set_accuracy(truth: &[i32], reconstructed: &[i32], tolerance: i32) -> (f64, f64) {
    let (matched_truth, truth_total, matched_recon, recon_total) = matched_counts(truth, reconstructed, tolerance);
    let recall = if truth_total == 0 {
        1.0
    } else {
        matched_truth as f64 / truth_total as f64
    };
    let precision = if recon_total == 0 {
        1.0
    } else {
        matched_recon as f64 / recon_total as f64
    };
    (recall, precision)
}

/// The freeze state of `trajectory`'s sample nearest `tick` (`trajectory` must be sorted by tick —
/// true of every `PlayerReconstruction::trajectory`, see `reconstruct::reconstruct`'s own doc
/// comment) — used to bucket a jump/fire ground-truth *event* tick (which will rarely land on an
/// exactly-observed trajectory tick — dead reckoning means most ticks have no sample at all, see
/// `run_validation`'s own "frozen ticks" line) by whether the player was frozen around then,
/// exactly like the exact-tick direction/hook comparison already does for ticks that *are*
/// directly observed.
fn nearest_freeze(trajectory: &[reconstruct::TrajectorySample], tick: i32) -> Option<bool> {
    let idx = trajectory.partition_point(|s| s.tick < tick);
    let after = trajectory.get(idx);
    let before = idx.checked_sub(1).and_then(|i| trajectory.get(i));
    [before, after]
        .into_iter()
        .flatten()
        .min_by_key(|s| (s.tick - tick).abs())
        .and_then(|s| s.freeze)
}

/// The trajectory sample *strictly before* `tick` (`trajectory` must be sorted by tick, same
/// precondition as [`nearest_freeze`]) — `None` if `tick` is at or before every observed sample.
///
/// Review round 2, finding F18 as originally worded said "the nearest observed sample at or
/// before it" (inclusive) — implemented that way first, then corrected to *strictly* before after
/// live testing on `ChillBlock5` (this finding's own e2e phase 2, see the crate's BUILD REPORT)
/// exposed why inclusive is wrong: DDNet's server resends a fresh `Character` netobj on a jump
/// event far more often than its usual periodic cadence (`m_TriggeredEvents`-driven — matches this
/// finding's own live data: the overwhelming majority of jump-press ticks had an *exact* observed
/// sample, `gap_before=0`), so a same-tick sample is common — and that sample already reflects the
/// *outcome* of the very press being judged: with the default tuning (`m_Jumps == 2`, double jump
/// enabled), an *air* jump sets `m_Jumped |= 3` — bit 1 included — in the same tick it executes
/// (`gamecore.cpp:249-253`; the grounded-jump branch, `gamecore.cpp:241-247`, only sets bit 1 too
/// when `m_Jumps <= 1`, i.e. double jump disabled — round 2's own citation named this branch
/// instead of the air-jump one; round 3 corrected it, the conclusion itself was never in
/// question). Reading bit 1 from that same-tick sample and calling it "already used before this
/// press" is circular:
/// it counts a press's own success as proof the press could not have happened. Strictly-before
/// asks the physically correct question — the tee's jump-availability state as of just *before*
/// this input was applied — and, measured on the same live recording, raised phase 2's executable
/// jump count from 12 to 16 (with a correspondingly higher recall) purely by removing this
/// self-exclusion; see `jump_could_execute`'s own doc comment for what's left after this fix.
fn strictly_before(trajectory: &[reconstruct::TrajectorySample], tick: i32) -> Option<&reconstruct::TrajectorySample> {
    let idx = trajectory.partition_point(|s| s.tick < tick);
    idx.checked_sub(1).and_then(|i| trajectory.get(i))
}

/// Review round 2, finding F18: a jump *press* (already known non-frozen — this is only ever
/// called on `jump_truth_free`, see `run_validation`) still cannot execute if the tee had already
/// used its second/air jump this flight: `gamecore.cpp:227-256`'s own gating checks exactly
/// `jumped & 2` (the "second jump used" bit) before allowing an air jump, independent of whether
/// the button was just freshly pressed. `None` (no observed sample strictly before `tick` at all)
/// is treated as executable — the same lenient "assume yes absent contrary evidence" default this
/// crate already uses for missing/sparse data elsewhere (e.g. `nearest_freeze` returning `None`).
///
/// Residual, *not* fixed by the strictly-before correction above (documented, not silently
/// claimed away): review round 3, finding F20 measured the actual cause of remaining misses
/// directly (a per-event classification across 8 live recordings, `docs/formats.md` §19.5) — it
/// is **not** dead-reckoning sparsity (samples here in fact arrive densely enough, ~every 2
/// ticks, to catch a 2-3-tick press 97/97 times). The real cause was the *validator*'s own jump
/// pulse being defined in wall-clock milliseconds against an independently-clocked ~50Hz send
/// loop, which could shrink an intended 2-tick press down to a single real input tick on the
/// wire; a genuinely 1-tick-long press is then invisible whenever DDNet's own every-2nd-tick
/// snapshot cadence, plus its release-tick resend, happens to land on the *other* tick. Fixed on
/// the validator side (`play_cmd::PULSE_TICKS`, now a tick-quantized minimum of 3 ticks, never
/// wall-clock milliseconds) and partly mitigated here too (`reconstruct::EstimatedInput::jump`
/// also fires on a rising edge of `jumped` bit 1, which a genuine air jump sets and — unlike bit
/// 0 — holds until landing, catching some 1-tick air-jump presses bit 0 alone would still miss).
/// What remains structurally unobservable at this snapshot rate, and is not a gap in this
/// function's own logic: a 1-tick *ground* jump with double-jump tuning enabled sets only bit 0,
/// so if dead-reckoning happens to miss that one tick, no later sample can reveal it after the
/// fact.
fn jump_could_execute(trajectory: &[reconstruct::TrajectorySample], tick: i32) -> bool {
    strictly_before(trajectory, tick).is_none_or(|s| s.jumped & 2 == 0)
}

/// Splits `events` into (non-frozen, frozen) by [`nearest_freeze`].
fn split_by_freeze(events: &[i32], trajectory: &[reconstruct::TrajectorySample]) -> (Vec<i32>, Vec<i32>) {
    let mut free = Vec::new();
    let mut frozen = Vec::new();
    for &tick in events {
        if nearest_freeze(trajectory, tick) == Some(true) {
            frozen.push(tick);
        } else {
            free.push(tick);
        }
    }
    (free, frozen)
}

fn run_validation(
    players: &[reconstruct::PlayerReconstruction],
    validate_path: &std::path::Path,
    client_id: Option<i32>,
    stint: Option<u32>,
) -> ExitCode {
    let log = match load_input_log(validate_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if log.is_empty() {
        eprintln!("{}: no logged inputs", validate_path.display());
        return ExitCode::FAILURE;
    }

    // Review round 1, finding F5: a bare `--client-id` (no `--player-name`, hence no already-
    // resolved `stint`) can match more than one `(client_id, stint)` track if that slot was
    // reused — pick the most recently active stint for that id, the same "most recent" choice
    // `--player-name` itself makes (see `reconstruct_cmd`).
    let player = match (client_id, stint) {
        (Some(id), Some(stint)) => players.iter().find(|p| p.client_id == id && p.stint == stint),
        (Some(id), None) => players.iter().filter(|p| p.client_id == id).max_by_key(|p| p.stint),
        (None, _) if players.len() == 1 => players.first(),
        (None, _) => {
            eprintln!(
                "recording has {} (client_id, stint) tracks; pass --client-id to pick one (tracks: {:?})",
                players.len(),
                players.iter().map(|p| (p.client_id, p.stint)).collect::<Vec<_>>()
            );
            return ExitCode::FAILURE;
        }
    };
    let Some(player) = player else {
        eprintln!("no such --client-id in this recording");
        return ExitCode::FAILURE;
    };

    let by_tick: BTreeMap<i32, &LoggedInput> = log.iter().map(|e| (e.tick, e)).collect();
    // `reconstruct::reconstruct` builds `trajectory`/`inputs` in lockstep (one push per dedup'd
    // `character.tick`, same loop iteration — see that function's own source) — same length, same
    // tick order, so a plain zip lines each `EstimatedInput` up with its `TrajectorySample`. Used
    // here purely for `freeze`, which `EstimatedInput` itself does not carry (see the accuracy
    // split below for why this matters).
    let freeze_by_tick: BTreeMap<i32, Option<bool>> = player.trajectory.iter().map(|s| (s.tick, s.freeze)).collect();

    // direction/hook: exact-tick agreement, only over ticks present in both series — split by
    // freeze state (`DDRaceTick`, `game/server/entities/character.cpp:2235-2253`: the *server*
    // forcibly zeroes `Direction`/`Jump` — and `Hook` too, once `m_FreezeTime > 0`, though not
    // during a bare `m_LiveFrozen` — before physics ever sees them while a character is frozen).
    // A ground-truth "the bot pressed direction/hook" while frozen has *no way* to show up in the
    // snapshot at all — that is not a reconstruction shortfall, it is the real, correct state the
    // server actually simulated, so lumping frozen and non-frozen ticks into one number
    // understates accuracy on exactly the ticks where the field genuinely is recoverable.
    struct Bucket {
        total: u64,
        matched: u64,
    }
    impl Bucket {
        fn frac(&self) -> f64 {
            if self.total == 0 {
                f64::NAN
            } else {
                self.matched as f64 / self.total as f64
            }
        }
    }
    let mut direction_frozen = Bucket { total: 0, matched: 0 };
    let mut direction_free = Bucket { total: 0, matched: 0 };
    let mut aim_total = 0u64;
    let mut aim_match = 0u64;
    let mut hook_frozen = Bucket { total: 0, matched: 0 };
    let mut hook_free = Bucket { total: 0, matched: 0 };
    for input in &player.inputs {
        let Some(truth) = by_tick.get(&input.tick) else {
            continue;
        };
        // `Some(true)` only — `Some(false)`/`None` (no DDNet extension at all, or not frozen) are
        // both treated as "not frozen" here: the accuracy-relevant distinction is specifically
        // "was `Direction`/`Jump`/`Hook` server-forced to zero", which only `Some(true)` means.
        let frozen = freeze_by_tick.get(&input.tick).copied().flatten().unwrap_or(false);
        let direction_bucket = if frozen {
            &mut direction_frozen
        } else {
            &mut direction_free
        };
        direction_bucket.total += 1;
        if input.direction == truth.direction {
            direction_bucket.matched += 1;
        }
        aim_total += 1;
        if input.aim_x == truth.target_x && input.aim_y == truth.target_y {
            aim_match += 1;
        }
        let hook_bucket = if frozen { &mut hook_frozen } else { &mut hook_free };
        hook_bucket.total += 1;
        if input.hook == (truth.hook != 0) {
            hook_bucket.matched += 1;
        }
    }
    let direction_total = direction_frozen.total + direction_free.total;
    let direction_match = direction_frozen.matched + direction_free.matched;

    // Jump/fire truth events are also split by freeze state (task 8.4a's own local e2e run found
    // this matters a great deal in practice — `DDRaceTick` forces `Jump` to 0, exactly like
    // `Direction`, while frozen; see `direction`'s identical split above and its doc comment) —
    // `set_accuracy`'s `tolerance` accounts for dead-reckoning sparsity (an event's true tick
    // rarely lands on an *observed* trajectory sample), not for freeze, which is why this split is
    // still necessary on top of it.
    let jump_truth = rising_edges(&log, |e| e.jump);
    let (jump_truth_free, jump_truth_frozen) = split_by_freeze(&jump_truth, &player.trajectory);
    // Review round 2, finding F18: a free (non-frozen) jump *press* still cannot execute if the
    // tee had already used its second/air jump this flight — `jumped & 2` in the trajectory
    // sample at-or-before the press tick (see `jump_could_execute`'s own doc comment for the
    // exact DDNet physics this mirrors). Excluded presses are reported explicitly (the finding's
    // own "report the exclusions"), not silently dropped from the denominator with no trace.
    let (jump_truth_executable, jump_truth_excluded_double_jump): (Vec<i32>, Vec<i32>) = jump_truth_free
        .iter()
        .copied()
        .partition(|&t| jump_could_execute(&player.trajectory, t));
    let jump_reconstructed: Vec<i32> = player.inputs.iter().filter(|i| i.jump).map(|i| i.tick).collect();
    let (jump_recall, jump_precision) = set_accuracy(&jump_truth, &jump_reconstructed, 2);
    // Review round 3, finding F20 step (2): `tools/e2e/record.sh`'s gate needs the raw executable
    // count (`jump_executable_n`) to enforce its own minimum-sample-size rule, not just the recall
    // fraction — `matched_counts` gives both without duplicating `set_accuracy`'s own matching
    // logic a second time.
    let (jump_executable_matched, jump_executable_n, _, _) =
        matched_counts(&jump_truth_executable, &jump_reconstructed, 2);
    let jump_recall_free = if jump_executable_n == 0 {
        1.0
    } else {
        jump_executable_matched as f64 / jump_executable_n as f64
    };

    // Review round 2, finding F17: `fire` became a monotonically-increasing counter (review round
    // 1, finding F14) — bit 0 (`& 1`) is the actual held/pressed level (matching DDNet's own
    // `CountInput` encoding this crate's `play_cmd::fire_counter` mirrors), the rest of the value
    // is just an incrementing count. Round 1's own `rising_edges(&log, |e| e.fire)` fed the raw
    // counter straight in — since a monotonic counter is "!= 0" from the moment it first ticks up
    // and never returns to exactly `0` again, that found the rising edge *once* for the whole
    // session and never again (exactly the reported symptom: "only 1 truth event is found per
    // run"). Fixed by extracting bit 0 first, exactly like `jump` already does for its own 0/1
    // level field.
    let fire_truth = rising_edges(&log, |e| e.fire & 1);
    let (fire_truth_free, fire_truth_frozen) = split_by_freeze(&fire_truth, &player.trajectory);
    let (fire_recall, fire_precision) = set_accuracy(&fire_truth, &player.fire_ticks, 1);
    // Review round 1's stricter e2e gate needs a free-only *precision* too (not just recall) —
    // bucket the reconstructed side by freeze state the same way the truth side already is, so a
    // reconstructed fire tick that genuinely matches a *frozen* truth event is not counted as a
    // free-precision false positive just because it is compared against the free truth subset.
    let (fire_recon_free, _fire_recon_frozen) = split_by_freeze(&player.fire_ticks, &player.trajectory);
    // Review round 3, finding F20 step (3): raw counts for pooling, same reasoning as jump above.
    let (fire_free_truth_matched, fire_free_truth_n, fire_free_recon_matched, fire_free_recon_n) =
        matched_counts(&fire_truth_free, &fire_recon_free, 1);
    let fire_recall_free = if fire_free_truth_n == 0 {
        1.0
    } else {
        fire_free_truth_matched as f64 / fire_free_truth_n as f64
    };
    let fire_precision_free = if fire_free_recon_n == 0 {
        1.0
    } else {
        fire_free_recon_matched as f64 / fire_free_recon_n as f64
    };

    let frac = |m: u64, t: u64| if t == 0 { f64::NAN } else { m as f64 / t as f64 };

    println!("input-reconstruction accuracy vs {}", validate_path.display());
    println!("  player client_id:    {} (stint {})", player.client_id, player.stint);
    println!("  compared ticks:      {direction_total} (of {} logged)", log.len());
    println!(
        "  frozen ticks:        {} of {direction_total} ({:.0}%) — Direction/Jump/(Hook while fully frozen) \
         are forced to 0 by the server itself on these, not just estimated poorly",
        direction_frozen.total,
        frac(direction_frozen.total, direction_total) * 100.0
    );
    println!(
        "  direction: {:.4} ({}/{}) overall [{}] — non-frozen only: {:.4} ({}/{}); frozen only: {:.4} ({}/{})",
        frac(direction_match, direction_total),
        direction_match,
        direction_total,
        confidence_json(player.confidence.direction),
        direction_free.frac(),
        direction_free.matched,
        direction_free.total,
        direction_frozen.frac(),
        direction_frozen.matched,
        direction_frozen.total,
    );
    println!(
        "  aim:       {:.4} ({}/{}) [{}]",
        frac(aim_match, aim_total),
        aim_match,
        aim_total,
        confidence_json(player.confidence.aim)
    );
    println!(
        "  hook:      {:.4} ({}/{}) overall [{}] — non-frozen only: {:.4} ({}/{}); frozen only: {:.4} ({}/{})",
        frac(
            hook_frozen.matched + hook_free.matched,
            hook_frozen.total + hook_free.total
        ),
        hook_frozen.matched + hook_free.matched,
        hook_frozen.total + hook_free.total,
        confidence_json(player.confidence.hook),
        hook_free.frac(),
        hook_free.matched,
        hook_free.total,
        hook_frozen.frac(),
        hook_frozen.matched,
        hook_frozen.total,
    );
    println!(
        "  jump:      recall {jump_recall:.4}, precision {jump_precision:.4} ({} truth events, {} reconstructed) [{}] \
         — executable only (non-frozen, second jump not already used): recall {jump_recall_free:.4} \
         ({} truth events); {} were frozen, {} excluded (already double-jumped)",
        jump_truth.len(),
        jump_reconstructed.len(),
        confidence_json(player.confidence.jump),
        jump_truth_executable.len(),
        jump_truth_frozen.len(),
        jump_truth_excluded_double_jump.len(),
    );
    println!(
        "  fire:      recall {fire_recall:.4}, precision {fire_precision:.4} ({} truth events, {} reconstructed) [{}] \
         — non-frozen only: recall {fire_recall_free:.4}, precision {fire_precision_free:.4} ({} truth events); \
         {} truth events were frozen",
        fire_truth.len(),
        player.fire_ticks.len(),
        confidence_json(player.confidence.fire),
        fire_truth_free.len(),
        fire_truth_frozen.len(),
    );

    // Machine-readable summary (review round 1's stricter e2e gate, `tools/e2e/record.sh`, reads
    // these `key: value` lines by name rather than parsing the human-readable prose above) — every
    // number here is restricted to non-frozen ticks/events specifically because that is the
    // population the gate's own accuracy thresholds are defined over: while frozen, the *server*
    // itself forces `Direction`/`Jump`/`Hook` to zero (`DDRaceTick`, see this file's own comments
    // above), so a frozen tick can never demonstrate this crate's reconstruction quality either
    // way, and folding frozen ticks into the denominator would only dilute a real regression (or
    // hide one, if a validation run happened to spend unusually long frozen).
    println!("free_ticks: {}", direction_free.total);
    println!("direction_free_acc: {:.4}", direction_free.frac());
    println!("hook_free_acc: {:.4}", hook_free.frac());
    // Review round 2, finding F18: `jump_free_recall` is computed over *executable* jump presses
    // only (non-frozen AND second jump not already used) — the key name is kept as-is (rather than
    // renamed to e.g. `jump_executable_recall`) so `tools/e2e/record.sh`'s existing gate/grep needs
    // no changes; what the number *means* is simply corrected to what it should have measured all
    // along. The exclusion count is reported explicitly, per the finding's own "report the
    // exclusions".
    println!("jump_free_recall: {jump_recall_free:.4}");
    println!("jump_excluded_double_jump: {}", jump_truth_excluded_double_jump.len());
    println!("fire_free_precision: {fire_precision_free:.4}");
    println!("fire_free_recall: {fire_recall_free:.4}");
    // Review round 3, finding F20 steps (2)/(3): raw matched/total counts, one line each, so
    // `tools/e2e/record.sh` can *pool* several phase-2 sub-runs by summing these across recordings
    // and recomputing the final fractions from the sums (statistically correct — averaging
    // already-computed fractions across sub-runs of different sizes would not be), and enforce its
    // own minimum-sample-size rule on `jump_executable_n` directly (never derivable from a
    // fraction alone: 16/16 and 2/2 both read `1.0000`).
    println!("direction_free_matched: {}", direction_free.matched);
    println!("direction_free_total: {}", direction_free.total);
    println!("hook_free_matched: {}", hook_free.matched);
    println!("hook_free_total: {}", hook_free.total);
    println!("jump_executable_n: {jump_executable_n}");
    println!("jump_executable_matched: {jump_executable_matched}");
    println!("fire_free_truth_n: {fire_free_truth_n}");
    println!("fire_free_truth_matched: {fire_free_truth_matched}");
    println!("fire_free_recon_n: {fire_free_recon_n}");
    println!("fire_free_recon_matched: {fire_free_recon_matched}");

    ExitCode::SUCCESS
}

fn anonymize_cmd(path: &std::path::Path, out: &std::path::Path) -> ExitCode {
    let (header, frames) = match read_all_frames(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let mut anonymizer = Anonymizer::new();
    let mut writer = match RecordingWriter::create(out, &header) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("failed to create {}: {e}", out.display());
            return ExitCode::FAILURE;
        }
    };
    let mut count = 0u64;
    for mut frame in frames {
        anonymizer.anonymize_frame(&mut frame);
        if let Err(e) = writer.write_frame(&frame) {
            eprintln!("failed to write frame: {e}");
            return ExitCode::FAILURE;
        }
        count += 1;
    }
    match writer.finish() {
        Ok(summary) => {
            println!(
                "wrote {} anonymized frames to {} ({} bytes)",
                count,
                out.display(),
                summary.bytes_written
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to finish {}: {e}", out.display());
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_net::generated::objects;

    fn player(id: i32, name: &str) -> ddai_recorder::format::PlayerRecord {
        ddai_recorder::format::PlayerRecord {
            id,
            info: objects::PlayerInfo {
                local: 0,
                client_id: id,
                team: 0,
                score: 0,
                latency: 0,
            },
            client_info: Some(objects::ClientInfo {
                name: name.to_string(),
                clan: String::new(),
                country: -1,
                skin: "default".to_string(),
                use_custom_color: 0,
                color_body: 0,
                color_feet: 0,
            }),
            ddnet: None,
        }
    }

    fn traj_sample(tick: i32, freeze: Option<bool>) -> reconstruct::TrajectorySample {
        reconstruct::TrajectorySample {
            tick,
            x: 0,
            y: 0,
            vel_x: 0,
            vel_y: 0,
            hook_state: 0,
            hooked_player: -1,
            weapon: 1,
            direction: 0,
            freeze,
            in_freeze_tile: freeze,
            aim_x: 0,
            aim_y: 0,
            jumped: 0,
        }
    }

    /// Review round 2, finding F18: like [`traj_sample`], plus an explicit `jumped` bitfield —
    /// only the tests that actually exercise [`jump_could_execute`] need this; every other test
    /// keeps using the plain 2-argument helper unchanged.
    fn traj_sample_jumped(tick: i32, freeze: Option<bool>, jumped: i32) -> reconstruct::TrajectorySample {
        reconstruct::TrajectorySample {
            jumped,
            ..traj_sample(tick, freeze)
        }
    }

    #[test]
    fn nearest_freeze_picks_the_closer_sample() {
        let traj = vec![
            traj_sample(0, Some(false)),
            traj_sample(10, Some(true)),
            traj_sample(20, Some(false)),
        ];
        assert_eq!(nearest_freeze(&traj, 0), Some(false));
        assert_eq!(nearest_freeze(&traj, 3), Some(false)); // closer to tick 0 than tick 10
        assert_eq!(nearest_freeze(&traj, 8), Some(true)); // closer to tick 10
        assert_eq!(nearest_freeze(&traj, 20), Some(false));
        assert_eq!(nearest_freeze(&traj, 100), Some(false)); // past the end -> nearest is tick 20
    }

    #[test]
    fn nearest_freeze_of_empty_trajectory_is_none() {
        assert_eq!(nearest_freeze(&[], 5), None);
    }

    #[test]
    fn nearest_freeze_none_when_the_nearest_sample_has_no_ddnet_extension() {
        let traj = vec![traj_sample(0, None)];
        assert_eq!(nearest_freeze(&traj, 0), None);
    }

    #[test]
    fn split_by_freeze_buckets_events_correctly() {
        let traj = vec![traj_sample(0, Some(false)), traj_sample(10, Some(true))];
        let (free, frozen) = split_by_freeze(&[0, 10, 5], &traj);
        // tick 5 is equidistant between the two samples; `nearest_freeze` resolves a tie to the
        // earlier (`before`) sample — tick 0, non-frozen.
        assert_eq!(free, vec![0, 5]);
        assert_eq!(frozen, vec![10]);
    }

    #[test]
    fn split_by_freeze_of_no_events_is_empty() {
        let traj = vec![traj_sample(0, Some(false))];
        let (free, frozen) = split_by_freeze(&[], &traj);
        assert!(free.is_empty());
        assert!(frozen.is_empty());
    }

    #[test]
    fn strictly_before_finds_the_last_sample_before_tick_never_at_it() {
        let traj = vec![traj_sample(0, None), traj_sample(10, None), traj_sample(20, None)];
        assert_eq!(
            strictly_before(&traj, 0),
            None,
            "nothing is strictly before the very first sample's own tick"
        );
        assert_eq!(strictly_before(&traj, 5).map(|s| s.tick), Some(0));
        assert_eq!(
            strictly_before(&traj, 10).map(|s| s.tick),
            Some(0),
            "a sample AT tick 10 must not count as before tick 10"
        );
        assert_eq!(strictly_before(&traj, 15).map(|s| s.tick), Some(10));
        assert_eq!(strictly_before(&traj, 100).map(|s| s.tick), Some(20));
    }

    #[test]
    fn strictly_before_is_none_when_tick_is_at_or_before_every_sample() {
        let traj = vec![traj_sample(10, None)];
        assert_eq!(strictly_before(&traj, 5), None);
        assert_eq!(strictly_before(&traj, 10), None);
    }

    /// Review round 2, finding F18: `jumped & 2` set in the sample strictly before the press tick
    /// means the second/air jump was already used — that press cannot execute even though free.
    #[test]
    fn jump_could_execute_is_false_when_the_second_jump_bit_is_already_set() {
        let traj = vec![traj_sample_jumped(0, Some(false), 0b11)]; // held + second-jump-used
        assert!(!jump_could_execute(&traj, 5));
    }

    #[test]
    fn jump_could_execute_is_true_when_the_second_jump_bit_is_clear() {
        let traj = vec![traj_sample_jumped(0, Some(false), 0b01)]; // held, second jump still available
        assert!(jump_could_execute(&traj, 5));
    }

    /// No observed sample strictly before the press tick at all — lenient default, per
    /// `jump_could_execute`'s own doc comment.
    #[test]
    fn jump_could_execute_defaults_to_true_with_no_earlier_sample() {
        let traj = vec![traj_sample_jumped(100, Some(false), 0b11)];
        assert!(jump_could_execute(&traj, 5));
    }

    /// Review round 2, finding F18 — the exact bug this fix corrects, reproduced from a live
    /// `ChillBlock5` recording (see the crate's BUILD REPORT): DDNet resends a fresh `Character`
    /// far more often right on a jump event, so a sample commonly lands *exactly on* the press
    /// tick — and that sample already reflects the press's own outcome (an air jump sets bit 1 in
    /// the very same tick it executes, `gamecore.cpp:249-253` — round 2's own citation named the
    /// grounded-jump branch, `gamecore.cpp:241-247`, instead; corrected in round 3). Judging *that*
    /// press by its own same-tick result is circular; `jump_could_execute` must look strictly
    /// before it.
    #[test]
    fn jump_could_execute_is_not_fooled_by_the_press_s_own_same_tick_outcome() {
        let traj = vec![
            traj_sample_jumped(0, Some(false), 0b00), // before the press: second jump not used yet
            traj_sample_jumped(5, Some(false), 0b11), // the press's own tick: sets bit 1 as a RESULT
        ];
        assert!(
            jump_could_execute(&traj, 5),
            "must look strictly before tick 5, not at it"
        );
    }

    /// Companion: a *prior*, genuinely earlier sample with bit 1 already set (from an earlier
    /// press, not this one) must still correctly exclude a later press.
    #[test]
    fn jump_could_execute_is_false_when_second_jump_was_used_by_a_genuinely_earlier_press() {
        let traj = vec![
            traj_sample_jumped(0, Some(false), 0b11), // an earlier press already used the second jump
            traj_sample_jumped(10, Some(false), 0b11), // this later press's own tick
        ];
        assert!(!jump_could_execute(&traj, 10));
    }

    /// Review round 2, finding F17: `rising_edges` on the raw fire counter (round 1's bug) finds
    /// only one "rising edge" ever, since a monotonic counter never returns to exactly 0 — the
    /// fix (`fire & 1`) must find every press.
    #[test]
    fn fire_counter_rising_edges_via_bit0_finds_every_press_not_just_the_first() {
        let log = vec![
            logged(0, 0, 0),
            logged(1, 0, 1), // first press: counter 0 -> 1 (bit0 set)
            logged(2, 0, 1), // still held
            logged(3, 0, 2), // released: counter -> 2 (bit0 clear)
            logged(4, 0, 3), // second press: counter -> 3 (bit0 set again)
        ];
        // The raw counter itself must NOT be used directly (the bug this regression test guards
        // against) — only one edge would be found that way.
        assert_eq!(rising_edges(&log, |e| e.fire).len(), 1);
        // The fix: extract bit 0 first.
        assert_eq!(rising_edges(&log, |e| e.fire & 1), vec![1, 4]);
    }

    /// Review round 1, finding F5: `reconstruct_cmd`'s `--player-name` resolution now goes through
    /// `reconstruct::resolve_name_to_stints` (its own tests cover the stint-tracking algorithm in
    /// depth) — this exercises the same integration point `--player-name` actually uses, picking
    /// the *last* (most recently active) match exactly like `reconstruct_cmd` itself does.
    #[test]
    fn resolve_name_to_stints_finds_the_matching_client_id_via_the_last_match() {
        let frames = vec![Frame::Snapshot {
            tick: 0,
            characters: vec![],
            players: vec![player(0, "Alice"), player(1, "RSValid")],
        }];
        assert_eq!(
            reconstruct::resolve_name_to_stints(&frames, "RSValid").last(),
            Some(&(1, 0))
        );
        assert_eq!(reconstruct::resolve_name_to_stints(&frames, "Bob").last(), None);
    }

    #[test]
    fn resolve_name_to_stints_skips_game_event_frames() {
        let frames = vec![Frame::GameEvent {
            tick_hint: 0,
            message: RecordedGameMessage::Broadcast {
                message: "hi".to_string(),
            },
        }];
        assert!(reconstruct::resolve_name_to_stints(&frames, "anyone").is_empty());
    }

    fn logged(tick: i32, jump: i32, fire: i32) -> LoggedInput {
        LoggedInput {
            tick,
            direction: 0,
            target_x: 0,
            target_y: -1,
            jump,
            fire,
            hook: 0,
        }
    }

    #[test]
    fn rising_edges_finds_only_the_press_tick_not_every_held_tick() {
        let log = vec![
            logged(0, 0, 0),
            logged(1, 1, 0),
            logged(2, 1, 0),
            logged(3, 0, 0),
            logged(4, 1, 0),
        ];
        assert_eq!(rising_edges(&log, |e| e.jump), vec![1, 4]);
    }

    #[test]
    fn rising_edges_of_all_zero_field_is_empty() {
        let log = vec![logged(0, 0, 0), logged(1, 0, 0)];
        assert!(rising_edges(&log, |e| e.fire).is_empty());
    }

    /// Review round 3, finding F20 step (3): `matched_counts` is what pooling across phase-2
    /// sub-runs actually sums — must agree with `set_accuracy`'s own fractions on the same input.
    #[test]
    fn matched_counts_agrees_with_set_accuracy_on_the_same_input() {
        let (matched_truth, truth_n, matched_recon, recon_n) = matched_counts(&[10, 20, 30], &[10, 500], 0);
        assert_eq!((matched_truth, truth_n, matched_recon, recon_n), (1, 3, 1, 2));
        let (recall, precision) = set_accuracy(&[10, 20, 30], &[10, 500], 0);
        assert_eq!(recall, matched_truth as f64 / truth_n as f64);
        assert_eq!(precision, matched_recon as f64 / recon_n as f64);
    }

    #[test]
    fn matched_counts_of_two_empty_sets_is_zero_over_zero() {
        assert_eq!(matched_counts(&[], &[], 0), (0, 0, 0, 0));
    }

    #[test]
    fn set_accuracy_is_perfect_for_identical_sets() {
        let (recall, precision) = set_accuracy(&[10, 20, 30], &[10, 20, 30], 0);
        assert_eq!(recall, 1.0);
        assert_eq!(precision, 1.0);
    }

    #[test]
    fn set_accuracy_tolerance_matches_nearby_ticks() {
        let (recall, precision) = set_accuracy(&[10], &[11], 2);
        assert_eq!(recall, 1.0);
        assert_eq!(precision, 1.0);
        let (recall, precision) = set_accuracy(&[10], &[15], 2);
        assert_eq!(recall, 0.0);
        assert_eq!(precision, 0.0);
    }

    #[test]
    fn set_accuracy_empty_truth_and_reconstructed_is_vacuously_perfect() {
        let (recall, precision) = set_accuracy(&[], &[], 0);
        assert_eq!(recall, 1.0);
        assert_eq!(precision, 1.0);
    }

    #[test]
    fn set_accuracy_extra_false_positives_hurt_precision_not_recall() {
        let (recall, precision) = set_accuracy(&[10], &[10, 500], 0);
        assert_eq!(recall, 1.0);
        assert_eq!(precision, 0.5);
    }

    #[test]
    fn load_input_log_parses_jsonl_and_sorts_by_tick() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truth.jsonl");
        std::fs::write(
            &path,
            "{\"tick\":5,\"direction\":1,\"target_x\":0,\"target_y\":-1,\"jump\":0,\"fire\":0,\"hook\":0,\"player_flags\":1,\"wanted_weapon\":0,\"next_weapon\":0,\"prev_weapon\":0}\n\
             {\"tick\":1,\"direction\":-1,\"target_x\":0,\"target_y\":-1,\"jump\":1,\"fire\":0,\"hook\":0,\"player_flags\":1,\"wanted_weapon\":0,\"next_weapon\":0,\"prev_weapon\":0}\n\
             \n",
        )
        .unwrap();
        let log = load_input_log(&path).unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].tick, 1);
        assert_eq!(log[1].tick, 5);
    }

    fn input_line(tick: i32) -> String {
        format!(
            "{{\"tick\":{tick},\"direction\":0,\"target_x\":0,\"target_y\":-1,\"jump\":0,\"fire\":0,\
             \"hook\":0,\"player_flags\":1,\"wanted_weapon\":0,\"next_weapon\":0,\"prev_weapon\":0}}\n"
        )
    }

    /// Task 8.4a: a reconnect (this project's own e2e test restarts the local server mid-session)
    /// resets the server's tick counter back down near zero — a naive whole-file sort by `tick`
    /// would interleave the pre-restart and post-restart connections' entries. Only the last
    /// epoch (ticks 5, 6, 7 here, after the drop back to 5 following the much larger 900s) must
    /// survive.
    #[test]
    fn load_input_log_keeps_only_the_last_connection_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truth.jsonl");
        let mut text = String::new();
        for tick in [10, 20, 900, 910] {
            text.push_str(&input_line(tick));
        }
        // The reconnect: tick drops from 910 back down to 5 — well past `EPOCH_RESET_SLACK`.
        for tick in [5, 6, 7] {
            text.push_str(&input_line(tick));
        }
        std::fs::write(&path, text).unwrap();

        let log = load_input_log(&path).unwrap();
        assert_eq!(log.iter().map(|e| e.tick).collect::<Vec<_>>(), vec![5, 6, 7]);
    }

    /// A drop within `EPOCH_RESET_SLACK` (ordinary same-epoch jitter/reordering) must *not* be
    /// treated as a new epoch — every entry survives.
    #[test]
    fn load_input_log_tolerates_small_same_epoch_tick_jitter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truth.jsonl");
        let mut text = String::new();
        for tick in [100, 101, 99, 102, 103] {
            text.push_str(&input_line(tick));
        }
        std::fs::write(&path, text).unwrap();

        let log = load_input_log(&path).unwrap();
        assert_eq!(log.len(), 5);
        assert_eq!(
            log.iter().map(|e| e.tick).collect::<Vec<_>>(),
            vec![99, 100, 101, 102, 103]
        );
    }

    #[test]
    fn load_input_log_rejects_malformed_json_with_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truth.jsonl");
        std::fs::write(&path, "not json\n").unwrap();
        assert!(load_input_log(&path).is_err());
    }
}
