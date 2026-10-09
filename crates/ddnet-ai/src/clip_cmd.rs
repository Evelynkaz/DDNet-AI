//! `ddnet-ai clip info|incidents|replay <file>...` (task 4.3): look at the bot's clips offline. Nothing here
//! touches a network or a server. `replay` loads the map from the map cache by the name and sha256 in the clip's
//! header (the clip never holds the map) and runs the bot's own physics over the recorded frames
//! (`ddai_clip::replay`), reporting how many steps reproduce the recording bit for bit and the first
//! divergence with its cause.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Args, Subcommand};
use ddai_clip::replay::{Mode, Report, replay_with};
use ddai_clip::{Clip, find_incidents, summarise};

#[derive(Debug, Args)]
pub struct ClipArgs {
    #[command(subcommand)]
    pub cmd: ClipCmd,
}

#[derive(Debug, Subcommand)]
pub enum ClipCmd {
    /// The header, the frames and the events of a clip.
    Info { file: PathBuf },
    /// The incidents the clip holds, by kind.
    Incidents { file: PathBuf },
    /// Task 3.10: what became of every block we made in the clips (held 5 s / escaped, and why escaped), per clip and in total.
    Held {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Also print the victim's track after each block (every 10 ticks: position, velocity, frozen, ticks left, our distance and target).
        #[arg(long)]
        track: bool,
    },
    /// Frames `from..=to` of a clip: our tee, the sent inputs, the events (to look at a divergence).
    Dump {
        file: PathBuf,
        #[arg(long, default_value_t = 0)]
        from: usize,
        #[arg(long, default_value_t = usize::MAX)]
        to: usize,
    },
    /// Task 4.12 (D-108): replays the self-kills the clips recorded under the smart policy and says which it would have skipped (and
    /// whether the tee would have been left stuck) or sent sooner; also runs the duel detector over every clip.
    Selfkill {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Where the maps are cached (default `<data-dir>/maps/cache`).
        #[arg(long)]
        map_cache: Option<PathBuf>,
        /// Print the smart policy's call at every frame of each replayed run.
        #[arg(long)]
        trace: bool,
        /// The clips are from an F-DDrace server (joniTee): the duel detector gets the F-DDrace evidence a clip (which holds no chat) lacks.
        /// Without it a two-player team is, rightly, no duel.
        #[arg(long)]
        f_ddrace: bool,
    },
    /// Task 4.12 (D-108): how good the smart policy's thaw forecast is on the frozen runs the clips show ending by themselves.
    Forecast {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Where the maps are cached (default `<data-dir>/maps/cache`).
        #[arg(long)]
        map_cache: Option<PathBuf>,
    },
    /// Replays clips offline and reports where the physics and the recording agree.
    Replay {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// `resync` (every step from the recorded state) or `free` (our tee runs free from the first frame).
        #[arg(long, default_value = "resync")]
        mode: String,
        /// Where the maps are cached (default `<data-dir>/maps/cache`).
        #[arg(long)]
        map_cache: Option<PathBuf>,
        /// Print every divergent step, not only the first.
        #[arg(long)]
        all: bool,
        /// A/B measurement of task 2.4c: rebuild the tees the way `LiveWorld` did before (`m_PrevPos` snapped to the
        /// current position, no `m_FrozenLastTick`).
        #[arg(long)]
        legacy_prev_pos: bool,
    },
}

fn default_cache() -> PathBuf {
    ddai_os::dirs::data_root_or_relative().join("maps/cache")
}

fn read(path: &Path) -> Result<Clip, String> {
    Clip::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn run(args: ClipArgs) -> ExitCode {
    let result = match args.cmd {
        ClipCmd::Info { file } => info(&file),
        ClipCmd::Incidents { file } => incidents(&file),
        ClipCmd::Held { files, track } => held(&files, track),
        ClipCmd::Dump { file, from, to } => dump(&file, from, to),
        ClipCmd::Forecast { files, map_cache } => forecast(&files, &map_cache.unwrap_or_else(default_cache)),
        ClipCmd::Selfkill {
            files,
            map_cache,
            trace,
            f_ddrace,
        } => selfkill(&files, &map_cache.unwrap_or_else(default_cache), trace, f_ddrace),
        ClipCmd::Replay {
            files,
            mode,
            map_cache,
            all,
            legacy_prev_pos,
        } => replay_files(
            &files,
            &mode,
            &map_cache.unwrap_or_else(default_cache),
            all,
            legacy_prev_pos,
        ),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn info(file: &Path) -> Result<ExitCode, String> {
    let c = read(file)?;
    let h = &c.header;
    println!(
        "{}: map {} ({}), brain {}, tee {}, {} frames{}",
        file.display(),
        h.map_name,
        h.map_sha256
            .iter()
            .take(4)
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        h.brain,
        h.own_id,
        c.frames.len(),
        c.tick_range()
            .map_or(String::new(), |(a, b)| format!(", ticks {a}..{b}")),
    );
    println!(
        "reason: {} (severity {}, tick {}) {}",
        h.reason.kind, h.reason.severity, h.reason.tick, h.reason.note
    );
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for e in c.frames.iter().flat_map(|f| f.events.iter()) {
        let name = match e {
            ddai_clip::ClipEvent::HammerHit { .. } => "hammer-hit",
            ddai_clip::ClipEvent::HammerFire { .. } => "hammer-fire",
            ddai_clip::ClipEvent::HookAttach { .. } => "hook-attach",
            ddai_clip::ClipEvent::HookRelease { .. } => "hook-release",
            ddai_clip::ClipEvent::FreezeOnset { .. } => "freeze-onset",
            ddai_clip::ClipEvent::Kill { .. } => "kill",
            ddai_clip::ClipEvent::Respawn { .. } => "respawn",
            ddai_clip::ClipEvent::KillSent { .. } => "kill-sent",
        };
        *counts.entry(name).or_default() += 1;
    }
    println!("events: {counts:?}");
    println!(
        "players: {}",
        h.players.iter().map(|p| p.tag.as_str()).collect::<Vec<_>>().join(" ")
    );
    Ok(ExitCode::SUCCESS)
}

fn dump(file: &Path, from: usize, to: usize) -> Result<ExitCode, String> {
    let c = read(file)?;
    for (i, f) in c
        .frames
        .iter()
        .enumerate()
        .skip(from)
        .take(to.saturating_sub(from).saturating_add(1))
    {
        let own = f.tee(c.header.own_id);
        println!(
            "#{i} tick {} alive {} own {} sent {:?} events {:?}",
            f.tick,
            f.own_alive,
            own.map_or("-".to_string(), |t| format!(
                "xy ({},{}) v ({},{}) hook {}/{} tick {} frozen {} (freeze_end {:?})",
                t.ch.x,
                t.ch.y,
                t.ch.vel_x,
                t.ch.vel_y,
                t.ch.hook_state,
                t.ch.hooked_player,
                t.ch.tick,
                t.frozen,
                t.dd.map(|d| d.freeze_end)
            )),
            f.sent
                .iter()
                .map(|s| (s.tick, s.input.direction, s.input.jump, s.input.hook, s.timing_known))
                .collect::<Vec<_>>(),
            f.events
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn held(files: &[PathBuf], track: bool) -> Result<ExitCode, String> {
    use ddai_clip::held::{Fate, FateCounts, block_fates};
    let mut total = FateCounts::default();
    // Clips overlap (two incidents a few seconds apart share frames): one block is counted once.
    let mut seen = std::collections::HashSet::new();
    for f in files {
        let c = read(f)?;
        let name = f
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().to_string());
        for b in block_fates(&c.frames, c.header.own_id) {
            let tag = c
                .header
                .players
                .iter()
                .find(|p| p.id == b.victim)
                .map_or("?", |p| p.tag.as_str());
            if !seen.insert((b.tick, tag.to_string())) {
                continue;
            }
            let fate = match b.fate {
                Fate::Held => "held".to_string(),
                Fate::Killed { weapon } => format!("killed (weapon {weapon})"),
                Fate::Escaped { after, why } => format!("escaped after {after} ticks ({why:?})"),
                Fate::Unknown(u) => format!("unknown ({u:?})"),
            };
            println!(
                "{name}: block at tick {} victim {tag}: {fate}; in view out {} ticks, target at block {}, frames off target {}, touches {}",
                b.tick, b.out_ticks, b.target_at_block, b.frames_off_target, b.touches
            );
            total.add(&b);
            if track {
                let mut next = 0;
                for fr in &c.frames[b.frame..] {
                    let dt = fr.tick - b.tick;
                    if dt > ddai_clip::held::HELD_TICKS + 20 {
                        break;
                    }
                    if dt < next {
                        continue;
                    }
                    next = dt + 10;
                    let (me, v) = (fr.tee(c.header.own_id), fr.tee(b.victim));
                    let dist = me.zip(v).map(|(m, v)| {
                        let (a, b2) = (m.pos(), v.pos());
                        ddai_libm::hypot(a.0 - b2.0, a.1 - b2.1).round()
                    });
                    println!(
                        "    +{dt:>3}: victim {} our dist {:?} target {} our hook {}/{}",
                        v.map_or("not in view".to_string(), |t| format!(
                            "({:.0},{:.0}) v ({:.1},{:.1}) frozen {} left {}",
                            t.ch.x,
                            t.ch.y,
                            t.vel().0,
                            t.vel().1,
                            t.frozen,
                            t.freeze_left
                        )),
                        dist,
                        fr.bot.target,
                        me.map_or(-1, |t| t.ch.hook_state),
                        me.map_or(-1, |t| t.ch.hooked_player)
                    );
                }
            }
        }
    }
    println!("{total:?} (escaped {})", total.escaped());
    Ok(ExitCode::SUCCESS)
}

fn incidents(file: &Path) -> Result<ExitCode, String> {
    let c = read(file)?;
    let found = find_incidents(&c.frames, c.header.own_id, ddai_clip::store::CONTEXT_TICKS);
    for s in summarise(&found) {
        println!("{:<20} x{:<3} worst {}", s.kind, s.count, s.worst);
    }
    for i in &found {
        println!("  {} at tick {} (severity {}): {}", i.kind, i.tick, i.severity, i.note);
    }
    Ok(ExitCode::SUCCESS)
}

fn print_report(name: &str, r: &Report, all: bool) {
    println!(
        "{name}: {} frames, {} steps, {} bit-exact ({:.1}%), fresh-core steps {}/{} exact, nobody near {}/{} exact, freeze changes {}/{} exact (fresh {}/{}; {:?}), {} skipped at deaths; not reproduced by cause {:?}",
        r.frames,
        r.steps,
        r.exact,
        100.0 * r.exact as f64 / r.steps.max(1) as f64,
        r.fresh_exact,
        r.fresh_steps,
        r.isolated_exact,
        r.isolated_steps,
        r.freeze_exact,
        r.freeze_steps,
        r.freeze_fresh_exact,
        r.freeze_fresh_steps,
        r.freeze_breakdown(),
        r.skipped_deaths,
        r.by_cause()
    );
    let show = |d: &ddai_clip::replay::Divergence| {
        println!(
            "  frame {} tick {}: {} reconstructed {} replayed {} (fields (name, reconstructed, replayed) {:?}), cause {:?}{}{}",
            d.frame,
            d.tick,
            d.field,
            d.recorded,
            d.replayed,
            d.fields,
            d.cause,
            if d.fresh { ", fresh core" } else { "" },
            if d.freeze_change { ", FREEZE CHANGE" } else { "" }
        );
    };
    if all {
        r.divergences.iter().for_each(show);
    } else if let Some(d) = &r.first_divergence {
        println!("  first divergence:");
        show(d);
    }
}

fn replay_files(files: &[PathBuf], mode: &str, cache: &Path, all: bool, legacy: bool) -> Result<ExitCode, String> {
    let mode = match mode {
        "resync" => Mode::Resync,
        "free" | "free-run" => Mode::FreeRun,
        other => return Err(format!("--mode {other:?}: resync | free")),
    };
    let mut all_exact = true;
    for f in files {
        let c = read(f)?;
        let bytes =
            ddai_client::map_cache::read_cached(cache, &c.header.map_name, &c.header.map_sha256).ok_or_else(|| {
                format!(
                    "{}: the map {} is not in {}",
                    f.display(),
                    c.header.map_name,
                    cache.display()
                )
            })?;
        let map = ddai_map::load_map(&bytes)
            .map_err(|e| format!("{}: {e}", c.header.map_name))?
            .data;
        let r = replay_with(&c, Arc::new(map), mode, legacy);
        all_exact &= r.unexplained().next().is_none()
            && r.first_divergence
                .as_ref()
                .is_none_or(|d| d.cause != ddai_clip::replay::Cause::ServerCorrection);
        print_report(
            &f.file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().to_string()),
            &r,
            all,
        );
    }
    Ok(if all_exact {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

/// `clip selfkill`: the recorded self-kills of the clips under the smart policy (task 4.12, `ddai_bot::kill_replay`).
fn selfkill(files: &[PathBuf], cache: &Path, trace: bool, f_ddrace: bool) -> Result<ExitCode, String> {
    use ddai_bot::kill_replay::{Fate, Outcome, analyse, scan_duel};
    let mut cases = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for f in files {
        let c = read(f)?;
        let name = f
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().to_string());
        let bytes = ddai_client::map_cache::read_cached(cache, &c.header.map_name, &c.header.map_sha256)
            .ok_or_else(|| format!("{name}: the map {} is not in {}", c.header.map_name, cache.display()))?;
        let map = Arc::new(
            ddai_map::load_map(&bytes)
                .map_err(|e| format!("{}: {e}", c.header.map_name))?
                .data,
        );
        let duel = scan_duel(&c, f_ddrace);
        println!(
            "{name}: map {} | duel detector: {}",
            c.header.map_name,
            match duel.first {
                Some((tick, why)) => format!(
                    "ON from tick {tick} by {} ({} of {} frames); kills inside the duel {:?}, outside {:?}",
                    why.name(),
                    duel.frames_in_duel,
                    duel.frames,
                    duel.kills_in_duel,
                    duel.kills_outside
                ),
                None => format!("never ({} frames)", duel.frames),
            }
        );
        println!("    own DDRace team (tick, team, others in it): {:?}", duel.own_team);
        // Clips overlap: one kill is counted once (the same map and tick).
        for case in analyse(&c, &map, &name) {
            if seen.insert((c.header.map_name.clone(), case.tick)) {
                cases.push(case);
            }
        }
    }
    println!();
    println!(
        "{:<44} {:>9} {:>10} {:>7}  outcome under the smart policy",
        "clip", "tick", "kind", "frozen"
    );
    let (mut same, mut earlier, mut skipped, mut not_judged, mut unknown) = (0, 0, 0, 0, 0);
    let (mut thawed, mut died, mut later, mut still, mut ended) = (0, 0, 0, 0, 0);
    let mut saved = Vec::new();
    for c in &cases {
        let text = match &c.verdict {
            Outcome::NotJudged(w) => {
                not_judged += 1;
                format!("not judged ({w})")
            }
            Outcome::LegacyNotReproduced => {
                unknown += 1;
                "legacy timers not reproduced (the clip does not hold the whole run)".to_string()
            }
            Outcome::Same { why } => {
                same += 1;
                format!("SAME kill ({})", why.name())
            }
            Outcome::Earlier { ticks, why } => {
                earlier += 1;
                saved.push(*ticks);
                format!("EARLIER by {ticks} ticks ({})", why.name())
            }
            Outcome::Skipped { why, fate } => {
                skipped += 1;
                let w = why.map_or("-", |w| w.name());
                match fate {
                    Fate::Thawed { after } => {
                        thawed += 1;
                        format!("SKIPPED ({w}): thawed by itself {after} ticks later")
                    }
                    Fate::Died { after } => {
                        died += 1;
                        format!("SKIPPED ({w}): died by itself {after} ticks later")
                    }
                    Fate::KilledLater { after, why } => {
                        later += 1;
                        format!("SKIPPED ({w}), then killed {after} ticks later ({})", why.name())
                    }
                    Fate::StillFrozen { ticks } => {
                        still += 1;
                        format!("SKIPPED ({w}): STILL FROZEN when the clip ends, {ticks} ticks later")
                    }
                    Fate::ClipEnded { ticks } => {
                        ended += 1;
                        format!("SKIPPED ({w}): free, clip ends {ticks} ticks later")
                    }
                }
            }
        };
        println!(
            "{:<44} {:>9} {:>10} {:>7}  {text}",
            c.clip,
            c.tick,
            c.why_name(),
            c.frozen_for
        );
        if trace {
            for l in &c.trace {
                println!("{l}");
            }
        }
    }
    println!();
    println!(
        "{} recorded kills: judged {} = same {same} + earlier {earlier} + skipped {skipped}; not judged {not_judged}; legacy not reproduced {unknown}",
        cases.len(),
        same + earlier + skipped
    );
    if !saved.is_empty() {
        let mean = f64::from(saved.iter().sum::<i32>()) / saved.len() as f64;
        println!(
            "earlier: {} kills sent {mean:.0} ticks sooner on average (min {}, max {})",
            saved.len(),
            saved.iter().min().unwrap_or(&0),
            saved.iter().max().unwrap_or(&0)
        );
    }
    println!(
        "skipped {skipped}: thawed by itself {thawed}, died by itself {died}, killed later {later}, STILL FROZEN at the clip's end {still}, free at the clip's end {ended}"
    );
    Ok(ExitCode::SUCCESS)
}

/// `clip forecast`: the thaw forecast against what the clips show (task 4.12, `ddai_bot::kill_replay::forecast_samples`).
fn forecast(files: &[PathBuf], cache: &Path) -> Result<ExitCode, String> {
    use ddai_bot::kill_replay::forecast_samples;
    const AGES: [i32; 4] = [25, 50, 75, 100];
    let mut samples = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for f in files {
        let c = read(f)?;
        let name = f
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().to_string());
        let bytes = ddai_client::map_cache::read_cached(cache, &c.header.map_name, &c.header.map_sha256)
            .ok_or_else(|| format!("{name}: the map {} is not in {}", c.header.map_name, cache.display()))?;
        let map = Arc::new(
            ddai_map::load_map(&bytes)
                .map_err(|e| format!("{}: {e}", c.header.map_name))?
                .data,
        );
        for s in forecast_samples(&c, &map, &name, &AGES) {
            // Clips overlap: one run is counted once.
            if seen.insert((c.header.map_name.clone(), s.run_start, s.age)) {
                samples.push(s);
            }
        }
    }
    println!(
        "{:<44} {:>9} {:>4} {:>9} {:>7} {:>8}",
        "clip", "run", "age", "predicted", "actual", "touched"
    );
    for s in &samples {
        println!(
            "{:<44} {:>9} {:>4} {:>9} {:>7} {:>8}",
            s.clip,
            s.run_start,
            s.age,
            if s.died {
                "dies".to_string()
            } else {
                s.predicted.map_or("held".to_string(), |p| p.to_string())
            },
            s.actual,
            if s.touched { "yes" } else { "no" }
        );
    }
    for (label, touched) in [("untouched", false), ("touched (hooked or within 60 px)", true)] {
        let set: Vec<_> = samples.iter().filter(|s| s.touched == touched).collect();
        let finite: Vec<_> = set.iter().filter(|s| s.predicted.is_some() && !s.died).collect();
        let within = finite
            .iter()
            .filter(|s| (s.predicted.unwrap_or(0) - s.actual).abs() <= 4)
            .count();
        let later = finite
            .iter()
            .filter(|s| s.actual > s.predicted.unwrap_or(0) + 4)
            .count();
        let earlier = finite
            .iter()
            .filter(|s| s.actual < s.predicted.unwrap_or(0) - 4)
            .count();
        let held = set.iter().filter(|s| s.predicted.is_none() && !s.died).count();
        let dies = set.iter().filter(|s| s.died).count();
        println!(
            "{label}: {} probes of thawed runs: forecast within 4 ticks {within}, thawed later than forecast {later}, earlier {earlier}; forecast `held` (never) {held}, `dies` {dies}",
            set.len()
        );
    }
    Ok(ExitCode::SUCCESS)
}
