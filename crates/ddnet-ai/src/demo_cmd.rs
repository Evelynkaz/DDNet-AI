//! Implementation of the `ddnet-ai demo` subcommand group: thin CLI glue around `ddai-demo`
//! (task 8.4b) for inspecting real DDNet `.demo` files. See `docs/formats.md` for the format and
//! `crates/ddai-demo/README.md` for the crate's own docs.
//!
//! Nicknames (`ClientInfo::name`) only ever appear in this CLI's own local terminal output —
//! never written to a file this tool controls the destination of, and `--anonymize` replaces
//! them even there, per task 8.4b acceptance criterion 3 and D-040. `--raw` (machine-comparable
//! output for exactness cross-checking) prints the snapshot's raw ints, which include the
//! `ClientInfo` name/clan as packed ints an `IntsToStr` decode trivially recovers — review round 1
//! finding F3 — so `--raw` and `--anonymize` are mutually exclusive (see [`DemoCommand::Dump`]),
//! rather than silently ignoring the flag.

use clap::{Args, Subcommand};
use ddai_demo::Demo;
use ddai_net::generated::enums::characterflagflag;
use ddai_net::generated::objects;
use ddai_net::view::View;
use ddai_physics::core::{HOOK_FLYING, HOOK_GRABBED, SERVER_TICK_SPEED};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Args)]
pub struct DemoArgs {
    #[command(subcommand)]
    pub command: DemoCommand,
}

#[derive(Debug, Subcommand)]
pub enum DemoCommand {
    /// Prints header, embedded-map, and tick-range info for one `.demo` file.
    Info { file: PathBuf },
    /// Dumps per-tick content. Human-readable typed summary by default; `--raw` instead prints
    /// one JSON line per tick with the raw snapshot items (key + data ints, in file order) —
    /// exactly the shape `tools/ddnet-oracle/demo2json` prints, for exactness cross-checking.
    /// `--raw` and `--anonymize` conflict (see the module docs) rather than one silently winning.
    Dump {
        file: PathBuf,
        /// Replace nicknames with a stable, first-appearance-order pseudonym instead of printing
        /// them. Conflicts with `--raw`.
        #[arg(long, conflicts_with = "raw")]
        anonymize: bool,
        /// Raw snapshot items (machine format) instead of a typed, human-readable summary.
        #[arg(long)]
        raw: bool,
        /// Stop after this many ticks.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Prints aggregate stats over one `.demo` file or a directory of them (searched
    /// recursively): duration, map names/sha256, distinct players, players on screen per tick
    /// (min/mean/max), `DDNetCharacter` presence, and freeze/hook/hammer event counts.
    Stats {
        path: PathBuf,
        /// Replace nicknames with a stable, first-appearance-order pseudonym instead of printing
        /// them.
        #[arg(long)]
        anonymize: bool,
    },
}

pub fn run(args: DemoArgs) -> ExitCode {
    match args.command {
        DemoCommand::Info { file } => info(&file),
        DemoCommand::Dump {
            file,
            anonymize,
            raw,
            limit,
        } => dump(&file, anonymize, raw, limit),
        DemoCommand::Stats { path, anonymize } => stats(&path, anonymize),
    }
}

/// Reads `path`, rejecting it on size *before* loading the bytes (review round 1 finding F9:
/// `std::fs::read` used to load the whole file first and only `Demo::parse` checked
/// `MAX_DEMO_FILE_SIZE` afterward — a multi-gigabyte hostile file would already have been fully
/// read into memory by then).
fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
        .len();
    if len > ddai_demo::header::MAX_DEMO_FILE_SIZE as u64 {
        return Err(format!(
            "{}: {len} bytes exceeds the {}-byte limit this reader accepts",
            path.display(),
            ddai_demo::header::MAX_DEMO_FILE_SIZE
        ));
    }
    std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))
}

fn hex32(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(64);
    for b in bytes {
        write!(out, "{b:02x}").expect("writing to a String cannot fail");
    }
    out
}

/// The map's SHA256 as a hex string: the header extension if present (version >= 6), else a hash
/// of the embedded bytes computed here — matching `CDemoPlayer::ExtractMap`'s own fallback
/// (`demo.cpp:940-950`) for pre-v6 demos, which never carry the extension at all (review round 1
/// finding F7: printing "(none)" for those was misleading — DDNet itself always ends up with a
/// real hash for the map, it just doesn't always ship one) — a demo with no embedded map at all
/// (`map.size == 0`, real and common: 30 files in the ChillerDragon corpus) has no bytes to hash
/// and is reported as such rather than as the hash of an empty buffer.
fn map_sha256_string(demo: &Demo) -> String {
    if let Some(sha) = &demo.map.sha256 {
        return hex32(sha);
    }
    if demo.map.size == 0 {
        return "(no embedded map)".to_string();
    }
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(demo.map_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    hex32(&digest)
}

/// Non-reversible, first-appearance-order pseudonym assignment for `--anonymize` (review round 1
/// finding F8: a fixed-key `DefaultHasher` is stable across machines/runs and reversible against
/// a name list — a determined reader could just hash every candidate nickname and match the
/// output). Numbering by first appearance instead carries no relationship to the original name at
/// all, cryptographic or otherwise; the price is that join order across a run is observable,
/// which this task's threat model (D-040: don't leak *which* real person played) doesn't care
/// about.
#[derive(Default)]
struct Anonymizer {
    seen: HashMap<String, usize>,
}

impl Anonymizer {
    fn label(&mut self, name: &str) -> String {
        let next = self.seen.len() + 1;
        let id = *self.seen.entry(name.to_string()).or_insert(next);
        format!("player_{id}")
    }
}

fn display_name(anon: &mut Option<Anonymizer>, name: &str) -> String {
    match anon {
        Some(a) => a.label(name),
        None => name.to_string(),
    }
}

fn info(path: &Path) -> ExitCode {
    let bytes = match read_file(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let demo = match Demo::parse(&bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to parse {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    println!("version: {}", demo.header.version);
    println!("netversion: {}", demo.header.netversion);
    println!("type: {}", demo.header.demo_type);
    println!("timestamp: {}", demo.header.timestamp);
    println!("recorded length (header): {} s", demo.header.length_seconds);
    println!(
        "map: {} ({} bytes, crc {:08x})",
        demo.map.name, demo.map.size, demo.map.crc
    );
    println!("map sha256: {}", map_sha256_string(&demo));
    println!("timeline markers: {}", demo.timeline_markers.len());

    let mut first_tick = None;
    let mut last_tick = None;
    let mut tick_count = 0usize;
    let mut error: Option<String> = None;
    for tick in demo.ticks() {
        match tick {
            Ok(t) => {
                first_tick.get_or_insert(t.tick);
                last_tick = Some(t.tick);
                tick_count += 1;
            }
            Err(e) => {
                error = Some(e.to_string());
                break;
            }
        }
    }
    match (first_tick, last_tick) {
        (Some(first), Some(last)) => {
            let seconds = (last - first) as f64 / f64::from(SERVER_TICK_SPEED);
            println!(
                "ticks: {tick_count} (first={first}, last={last}, ~{seconds:.1} s at {SERVER_TICK_SPEED} ticks/s)"
            );
        }
        _ => println!("ticks: 0"),
    }
    if let Some(e) = error {
        println!("stopped early: {e}");
    }
    ExitCode::SUCCESS
}

fn dump(path: &Path, anonymize: bool, raw: bool, limit: Option<usize>) -> ExitCode {
    let bytes = match read_file(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let demo = match Demo::parse(&bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to parse {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    // `raw` and `anonymize` are mutually exclusive at the clap level (see `DemoCommand::Dump`);
    // this is not reachable, but the type still lets us build the same `Option<Anonymizer>`
    // shape `stats` uses.
    let mut anon = anonymize.then(Anonymizer::default);

    for (i, tick) in demo.ticks().enumerate() {
        if let Some(limit) = limit
            && i >= limit
        {
            break;
        }
        let tick = match tick {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error at tick #{i}: {e}");
                return ExitCode::FAILURE;
            }
        };

        let Some(snap) = &tick.snapshot else {
            continue;
        };

        if raw {
            print!("{{\"tick\":{},\"items\":[", tick.tick);
            for (i, item) in snap.items.iter().enumerate() {
                if i > 0 {
                    print!(",");
                }
                print!("{{\"key\":{},\"data\":[", item.key);
                for (j, v) in item.data.iter().enumerate() {
                    if j > 0 {
                        print!(",");
                    }
                    print!("{v}");
                }
                print!("]}}");
            }
            println!("]}}");
            continue;
        }

        let view = View::new(snap);
        let characters = view.characters();
        let players = view.players();
        print!("tick {}: {} characters", tick.tick, characters.len());
        for c in &characters {
            let name = players
                .iter()
                .find(|p| p.id == c.id)
                .and_then(|p| p.client_info.as_ref())
                .map(|ci| display_name(&mut anon, &ci.name));
            print!(
                " [id={} pos=({},{}) ddnet={}{}]",
                c.id,
                c.character.x,
                c.character.y,
                c.ddnet.is_some(),
                name.map(|n| format!(" name={n}")).unwrap_or_default()
            );
        }
        if !tick.messages.is_empty() {
            print!(" messages={}", tick.messages.len());
        }
        println!();
    }
    ExitCode::SUCCESS
}

/// Recursively collects every `*.demo` file under `path` (or just `path` itself, if it is a
/// file), skipping dot-directories (e.g. a `.git` checkout, as in ChillerDragon's archive).
/// Sorted for reproducible output.
fn collect_demo_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries =
            std::fs::read_dir(&dir).map_err(|e| format!("failed to read directory {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("failed to read directory entry in {}: {e}", dir.display()))?;
            let p = entry.path();
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("demo")) {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

#[derive(Default)]
struct RunningStats {
    min: Option<u64>,
    max: Option<u64>,
    sum: u128,
    count: u64,
}

impl RunningStats {
    fn add(&mut self, v: u64) {
        self.min = Some(self.min.map_or(v, |m| m.min(v)));
        self.max = Some(self.max.map_or(v, |m| m.max(v)));
        self.sum += v as u128;
        self.count += 1;
    }
    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum as f64 / self.count as f64
        }
    }
}

/// Records `id`'s current boolean state (freezing or hooking) in `prev` and reports whether this
/// is a false→true transition — i.e. a new "event" to count, not a continuation of one already
/// counted.
///
/// Review round 2 finding F12: the previous inline form only called `prev.insert` while `now` was
/// `true` (`if now && !prev.insert(...)`), so a player's recorded state could reach `true` but
/// then never go back to `false` on unfreeze/un-hook — every later re-freeze then saw `Some(true)`
/// already in the map and was silently not counted. That undercounted real transitions by roughly
/// two orders of magnitude on real traffic (18 hook-grab events reported vs. an independent count
/// of 2317 on the same file). The fix is to always record the new state first, then check the
/// transition against what was there before.
fn is_new_event(prev: &mut HashMap<i32, bool>, id: i32, now: bool) -> bool {
    let was = prev.insert(id, now).unwrap_or(false);
    now && !was
}

fn stats(path: &Path, anonymize: bool) -> ExitCode {
    let files = match collect_demo_files(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if files.is_empty() {
        eprintln!("no .demo files found under {}", path.display());
        return ExitCode::FAILURE;
    }

    // (map name, sha256-hex-or-a-"(no embedded map)" label) -> number of demos with that map.
    let mut maps: BTreeMap<(String, String), u64> = BTreeMap::new();
    let mut anon = anonymize.then(Anonymizer::default);
    let mut player_names: HashSet<String> = HashSet::new();
    let mut on_screen = RunningStats::default();
    let mut total_duration_ticks: i64 = 0;
    let mut total_characters: u64 = 0;
    let mut ddnet_characters: u64 = 0;
    // Transition-based event counts (review round 1 finding F7: a per-tick "character is
    // currently frozen/hooking" count conflates a state's *duration* with how often it actually
    // *starts* — e.g. one player frozen for 500 ticks would dwarf 50 players each briefly frozen
    // once). A "freeze event" is a `false -> true` edge on `IN_FREEZE` for one client id; a "hook
    // grab event" likewise for entering `HOOK_FLYING`/`HOOK_GRABBED` (not any nonzero
    // `hook_state` — `HOOK_RETRACTED` is `-1`, a real, "not attached" state, not a grab).
    let mut freeze_events: u64 = 0;
    let mut hook_grab_events: u64 = 0;
    let mut hammer_hits: u64 = 0;
    let mut files_ok = 0u64;
    let mut files_failed = 0u64;
    let mut ticks_total: u64 = 0;
    let mut tick_errors: u64 = 0;

    for path in &files {
        let bytes = match read_file(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping {}: {e}", path.display());
                files_failed += 1;
                continue;
            }
        };
        let demo = match Demo::parse(&bytes) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping {}: parse failed: {e}", path.display());
                files_failed += 1;
                continue;
            }
        };
        files_ok += 1;

        let sha = map_sha256_string(&demo);
        *maps.entry((demo.map.name.clone(), sha)).or_insert(0) += 1;

        // Reset per file: client ids are only meaningful within one recording session (a demo
        // that outlives a disconnect/reconnect could reassign the same id to a different player,
        // which would misattribute one transition — an accepted, documented limitation, not
        // fixed here since it needs player-identity tracking this report doesn't otherwise do).
        let mut prev_freeze: HashMap<i32, bool> = HashMap::new();
        let mut prev_hooking: HashMap<i32, bool> = HashMap::new();

        let mut first_tick = None;
        let mut last_tick = None;
        for tick in demo.ticks() {
            let tick = match tick {
                Ok(t) => t,
                Err(_) => {
                    tick_errors += 1;
                    break;
                }
            };
            ticks_total += 1;
            first_tick.get_or_insert(tick.tick);
            last_tick = Some(tick.tick);

            let Some(snap) = &tick.snapshot else {
                continue;
            };
            let view = View::new(snap);
            let characters = view.characters();
            on_screen.add(characters.len() as u64);
            for c in &characters {
                total_characters += 1;

                let mut freezing_now = false;
                if let Some(ddnet) = &c.ddnet {
                    ddnet_characters += 1;
                    freezing_now = ddnet.flags & characterflagflag::IN_FREEZE != 0;
                }
                if is_new_event(&mut prev_freeze, c.id, freezing_now) {
                    freeze_events += 1;
                }

                let hooking_now = matches!(c.character.hook_state, HOOK_FLYING | HOOK_GRABBED);
                if is_new_event(&mut prev_hooking, c.id, hooking_now) {
                    hook_grab_events += 1;
                }
            }
            for player in view.players() {
                if let Some(ci) = &player.client_info {
                    player_names.insert(display_name(&mut anon, &ci.name));
                }
            }
            hammer_hits += snap
                .items
                .iter()
                .filter(|it| it.internal_type() == objects::HammerHit::ID)
                .count() as u64;
        }
        if let (Some(first), Some(last)) = (first_tick, last_tick) {
            total_duration_ticks += (last - first) as i64;
        }
    }

    println!("files: {files_ok} ok, {files_failed} failed to parse");
    println!(
        "duration: {:.1} s total ({SERVER_TICK_SPEED} ticks/s), {ticks_total} ticks, {tick_errors} files stopped early on a decode error",
        total_duration_ticks as f64 / f64::from(SERVER_TICK_SPEED)
    );
    println!("maps ({}):", maps.len());
    for ((name, sha), count) in &maps {
        println!("  {name} sha256={sha} ({count} demos)");
    }
    println!("distinct players: {}", player_names.len());
    if anonymize {
        println!("  (nicknames replaced with --anonymize pseudonyms)");
    } else {
        let mut names: Vec<&String> = player_names.iter().collect();
        names.sort();
        for n in names {
            println!("  {n}");
        }
    }
    println!(
        "players on screen per tick: min={} mean={:.2} max={}",
        on_screen.min.unwrap_or(0),
        on_screen.mean(),
        on_screen.max.unwrap_or(0)
    );
    let ddnet_pct = if total_characters == 0 {
        0.0
    } else {
        100.0 * ddnet_characters as f64 / total_characters as f64
    };
    println!("DDNetCharacter presence: {ddnet_characters}/{total_characters} character-snapshots ({ddnet_pct:.1}%)");
    println!(
        "freeze events (entered IN_FREEZE): {freeze_events}; hook grab events (entered HOOK_FLYING/HOOK_GRABBED): {hook_grab_events}; hammer hit events: {hammer_hits}"
    );

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review round 2 finding F12, freeze counter: freeze → unfreeze → freeze must be counted as
    /// 2 events (one per false→true transition), not 1 (the pre-fix bug: the state never returned
    /// to `false` once a player had frozen once, so a later re-freeze went uncounted).
    #[test]
    fn freeze_then_unfreeze_then_freeze_counts_two_events() {
        let mut prev_freeze: HashMap<i32, bool> = HashMap::new();
        let player = 1;
        let mut freeze_events = 0;

        // Tick 1: enters freeze.
        if is_new_event(&mut prev_freeze, player, true) {
            freeze_events += 1;
        }
        // Tick 2: still frozen — not a new event.
        if is_new_event(&mut prev_freeze, player, true) {
            freeze_events += 1;
        }
        // Tick 3: unfreezes — not itself an event.
        if is_new_event(&mut prev_freeze, player, false) {
            freeze_events += 1;
        }
        // Tick 4: still unfrozen — not an event.
        if is_new_event(&mut prev_freeze, player, false) {
            freeze_events += 1;
        }
        // Tick 5: freezes again — a second event.
        if is_new_event(&mut prev_freeze, player, true) {
            freeze_events += 1;
        }

        assert_eq!(
            freeze_events, 2,
            "one event per freeze, not one total for the whole demo"
        );
    }

    /// Same pattern for the hook-grab counter (hooking → released → hooking again = 2 events).
    #[test]
    fn hook_then_release_then_hook_counts_two_events() {
        let mut prev_hooking: HashMap<i32, bool> = HashMap::new();
        let player = 1;
        let mut hook_grab_events = 0;

        if is_new_event(&mut prev_hooking, player, true) {
            hook_grab_events += 1;
        }
        if is_new_event(&mut prev_hooking, player, false) {
            hook_grab_events += 1;
        }
        if is_new_event(&mut prev_hooking, player, true) {
            hook_grab_events += 1;
        }

        assert_eq!(
            hook_grab_events, 2,
            "one event per hook grab, not one total for the whole demo"
        );
    }

    /// Different players' transitions must not interfere with each other.
    #[test]
    fn is_new_event_tracks_each_id_independently() {
        let mut prev: HashMap<i32, bool> = HashMap::new();
        assert!(is_new_event(&mut prev, 1, true), "player 1's first freeze is an event");
        assert!(
            is_new_event(&mut prev, 2, true),
            "player 2's first freeze is an event too"
        );
        assert!(!is_new_event(&mut prev, 1, true), "player 1 still frozen: no new event");
        assert!(!is_new_event(&mut prev, 2, true), "player 2 still frozen: no new event");
    }

    /// An id never seen before defaults to "was not in the state" rather than panicking or
    /// treating a first `true` observation as a non-event.
    #[test]
    fn is_new_event_first_observation_of_true_is_an_event() {
        let mut prev: HashMap<i32, bool> = HashMap::new();
        assert!(is_new_event(&mut prev, 42, true));
    }

    /// An id whose very first observation is `false` (e.g. a player who joins already unfrozen)
    /// must not be counted as an event.
    #[test]
    fn is_new_event_first_observation_of_false_is_not_an_event() {
        let mut prev: HashMap<i32, bool> = HashMap::new();
        assert!(!is_new_event(&mut prev, 42, false));
    }
}
