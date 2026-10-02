//! `ddnet-ai clip info|incidents|replay <file>...` (task 4.3): look at the bot's clips offline. Nothing here
//! touches a network or a server. `replay` loads the map from the map cache by the name and sha256 in the clip's
//! header (the clip never holds the map) and runs the bot's own physics over the recorded frames
//! (`ddai_clip::replay`), reporting how many steps reproduce the recording bit for bit and the first
//! divergence with its cause.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Args, Subcommand};
use ddai_clip::replay::{Mode, Report, replay};
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
    /// Frames `from..=to` of a clip: our tee, the sent inputs, the events (to look at a divergence).
    Dump {
        file: PathBuf,
        #[arg(long, default_value_t = 0)]
        from: usize,
        #[arg(long, default_value_t = usize::MAX)]
        to: usize,
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
    },
}

fn default_cache() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache")
}

fn read(path: &Path) -> Result<Clip, String> {
    Clip::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn run(args: ClipArgs) -> ExitCode {
    let result = match args.cmd {
        ClipCmd::Info { file } => info(&file),
        ClipCmd::Incidents { file } => incidents(&file),
        ClipCmd::Dump { file, from, to } => dump(&file, from, to),
        ClipCmd::Replay {
            files,
            mode,
            map_cache,
            all,
        } => replay_files(&files, &mode, &map_cache.unwrap_or_else(default_cache), all),
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
        "{name}: {} frames, {} steps, {} bit-exact ({:.1}%), fresh-core steps {}/{} exact, nobody near {}/{} exact, freeze changes {}/{} exact (fresh {}/{}), {} skipped at deaths; not reproduced by cause {:?}",
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
        r.skipped_deaths,
        r.by_cause()
    );
    let show = |d: &ddai_clip::replay::Divergence| {
        println!(
            "  frame {} tick {}: {} reconstructed {} replayed {} (fields (name, reconstructed, replayed) {:?}), cause {:?}{}",
            d.frame,
            d.tick,
            d.field,
            d.recorded,
            d.replayed,
            d.fields,
            d.cause,
            if d.fresh { ", fresh core" } else { "" }
        );
    };
    if all {
        r.divergences.iter().for_each(show);
    } else if let Some(d) = &r.first_divergence {
        println!("  first divergence:");
        show(d);
    }
}

fn replay_files(files: &[PathBuf], mode: &str, cache: &Path, all: bool) -> Result<ExitCode, String> {
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
        let r = replay(&c, Arc::new(map), mode);
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
