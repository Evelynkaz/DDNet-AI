//! `ddnet-ai dataset`: the human-play dataset (task 8.4c, `ddai-dataset`).
//!
//! - `from-demos`: DDNet demos -> `(Observation, Action, meta)` dataset + report (heavy, run by
//!   hand, never part of `cargo test`);
//! - `info`: print the report of a finished dataset (optionally verify the chunk checksums);
//! - `show`: print a window of reconstructed frames around a `(demo, tick)` hit, anonymously;
//! - `locate`: find the local demo file for a sha256 prefix (local output only, so a hit from the
//!   report can be opened in the replay).
//!
//! Nothing this command prints or writes contains a nickname; demos are addressed by sha256.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Subcommand};
use ddai_dataset::config::Config;
use ddai_dataset::dataset::{DatasetReader, sha256_hex};
use ddai_dataset::report;
use ddai_dataset::run::{self, Options};

#[derive(Debug, Args)]
pub struct DatasetArgs {
    #[command(subcommand)]
    pub command: DatasetCommand,
}

#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// Builds a dataset from a directory of DDNet client demos.
    FromDemos {
        /// Directory searched recursively for `*.demo`.
        #[arg(long)]
        demos: PathBuf,
        /// Output directory (created).
        #[arg(long)]
        out: PathBuf,
        /// Directories searched for `.map` files when a demo has no usable embedded map (matched
        /// by the header's crc and size). Repeatable.
        #[arg(long = "maps")]
        maps: Vec<PathBuf>,
        /// Worker threads (at most 6).
        #[arg(long, default_value_t = 6)]
        threads: usize,
        /// Recorded in the manifest; default `git rev-parse HEAD` (+dirty).
        #[arg(long)]
        code_commit: Option<String>,
        #[arg(long, default_value = "human-dataset")]
        name: String,
        /// Free-text source description for the manifest (no names).
        #[arg(long, default_value = "DDNet client demos")]
        source: String,
        /// Process only the first N demos (sha256 order); smoke tests.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Prints the report of a dataset; `--verify` also checks every chunk's sha256.
    Info {
        dir: PathBuf,
        #[arg(long)]
        verify: bool,
    },
    /// Prints frames of one demo (sha256 prefix) in a tick window, anonymously.
    Show {
        dir: PathBuf,
        /// sha256 prefix of the demo (>= 6 hex characters).
        demo: String,
        #[arg(long)]
        tick: i32,
        /// Half-width of the window in ticks.
        #[arg(long, default_value_t = 30)]
        window: i32,
        /// Show only these anonymous player labels (repeatable); default all.
        #[arg(long = "player")]
        players: Vec<u16>,
    },
    /// Finds the local demo file(s) whose sha256 starts with the prefix.
    Locate {
        #[arg(long)]
        demos: PathBuf,
        prefix: String,
    },
}

fn git_commit() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    match run(&["rev-parse", "HEAD"]) {
        Some(h) => {
            let dirty = run(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
            if dirty { format!("{h}+dirty") } else { h }
        }
        None => "unknown".to_string(),
    }
}

pub fn run(args: DatasetArgs) -> ExitCode {
    match run_inner(args.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_inner(cmd: DatasetCommand) -> Result<(), String> {
    match cmd {
        DatasetCommand::FromDemos {
            demos,
            out,
            maps,
            threads,
            code_commit,
            name,
            source,
            limit,
        } => {
            let opts = Options {
                demos_dir: demos,
                out_dir: out,
                map_dirs: maps,
                threads,
                code_commit: code_commit.unwrap_or_else(git_commit),
                name,
                source,
                limit,
                top_players: 20,
            };
            let rep = run::from_demos(&opts, &Config::default(), &|m| eprintln!("{m}")).map_err(|e| e.to_string())?;
            println!("{}", report::render(&rep));
            Ok(())
        }
        DatasetCommand::Info { dir, verify } => {
            let reader = DatasetReader::open(&dir).map_err(|e| e.to_string())?;
            if verify {
                reader.verify().map_err(|e| e.to_string())?;
                println!("all {} chunks verified", reader.manifest.chunks.len());
            }
            let text = std::fs::read(dir.join(ddai_dataset::dataset::REPORT)).map_err(|e| e.to_string())?;
            let rep: report::Report = serde_json::from_slice(&text).map_err(|e| e.to_string())?;
            println!("{}", report::render(&rep));
            Ok(())
        }
        DatasetCommand::Show {
            dir,
            demo,
            tick,
            window,
            players,
        } => show(&dir, &demo, tick, window, &players),
        DatasetCommand::Locate { demos, prefix } => {
            for f in run::discover(&demos, "demo").map_err(|e| e.to_string())? {
                let bytes = std::fs::read(&f).map_err(|e| e.to_string())?;
                if sha256_hex(&bytes).starts_with(&prefix) {
                    println!("{}", f.display());
                }
            }
            Ok(())
        }
    }
}

fn show(dir: &Path, prefix: &str, tick: i32, window: i32, players: &[u16]) -> Result<(), String> {
    let reader = DatasetReader::open(dir).map_err(|e| e.to_string())?;
    let di = reader
        .manifest
        .demos
        .iter()
        .position(|d| d.sha256.starts_with(prefix))
        .ok_or("no demo with that sha256 prefix")?;
    for ci in 0..reader.manifest.chunks.len() {
        if reader.manifest.chunks[ci].demo as usize != di {
            continue;
        }
        let chunk = reader.read_chunk(ci).map_err(|e| e.to_string())?;
        for (fi, f) in chunk.frames.iter().enumerate() {
            if (f.tick - tick).abs() > window {
                continue;
            }
            println!("tick {}", f.tick);
            for c in f
                .chars
                .iter()
                .filter(|c| players.is_empty() || players.contains(&c.player))
            {
                let s = chunk
                    .samples
                    .iter()
                    .find(|s| s.frame as usize == fi && f.chars[s.slot as usize].id == c.id);
                let act = s.map(|s| {
                    format!(
                        "dir {:2} jump {} hook {} fire {} tags {:#x} replay {}",
                        s.action.direction,
                        u8::from(s.action.jump),
                        u8::from(s.action.hook),
                        u8::from(s.action.fire),
                        s.tags,
                        s.replay().name()
                    )
                });
                println!(
                    "  p{:<3} id {:2} pos ({:8.1},{:8.1}) vel ({:6.2},{:6.2}) hook {} -> {:3} flags {:#04x} {}",
                    c.player,
                    c.id,
                    c.pos[0],
                    c.pos[1],
                    c.vel[0],
                    c.vel[1],
                    c.hook_state,
                    c.hooked_player,
                    c.flags,
                    act.unwrap_or_default()
                );
            }
        }
    }
    Ok(())
}
