//! `ddnet-ai oppnet-live report` (task 3.17, D-111): the accuracy of the live window model against hold, from the bot's `oppnet-live.jsonl`.
//!
//! Reads files only. A log that was rotated (`oppnet-live.jsonl.1`, `.2`, ...) is read whole: the numbered neighbours of every path are included
//! unless `--no-rotated` says otherwise. The numbers are those of `opp_clips` (direction hit rate, hook, aim error), per tick of the window `k` and per
//! our lag `w`; see `ddai_oppnet::live::analyze` and `docs/formats.md` for the line format.

use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Args)]
pub struct OppnetLiveArgs {
    #[command(subcommand)]
    pub command: OppnetLiveCommand,
}

#[derive(Debug, Subcommand)]
pub enum OppnetLiveCommand {
    /// Accuracy of the model against hold: by tick of the window (k) and by our lag (w), the guard's changes of state.
    Report {
        /// The log file(s), e.g. `~/aiddnet/data/bot/oppnet-live.jsonl`.
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Also the table by lag and tick together.
        #[arg(long)]
        detail: bool,
        /// Read exactly the given files, not their rotated neighbours `<file>.1` ... `<file>.9`.
        #[arg(long)]
        no_rotated: bool,
    },
}

pub fn run(args: OppnetLiveArgs) -> ExitCode {
    match args.command {
        OppnetLiveCommand::Report {
            files,
            detail,
            no_rotated,
        } => report(&files, detail, !no_rotated),
    }
}

/// The files to read: each given path, and (when `rotated`) its numbered neighbours that exist, oldest first.
fn expand(files: &[PathBuf], rotated: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for f in files {
        if rotated {
            for n in (1..=9).rev() {
                let mut s = f.as_os_str().to_owned();
                s.push(format!(".{n}"));
                let p = PathBuf::from(s);
                if p.is_file() && !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        if !out.contains(f) {
            out.push(f.clone());
        }
    }
    out
}

fn report(files: &[PathBuf], detail: bool, rotated: bool) -> ExitCode {
    let mut r = ddai_oppnet::live::analyze::Report::new();
    for path in expand(files, rotated) {
        let file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = r.add_reader(std::io::BufReader::new(file)) {
            eprintln!("{}: {e}", display(&path));
            return ExitCode::FAILURE;
        }
    }
    if r.samples() == 0 {
        eprintln!("no samples in the given log(s): was the bot started with --window-model, and did it fight?");
        return ExitCode::FAILURE;
    }
    print!("{}", r.render(detail));
    ExitCode::SUCCESS
}

fn display(p: &Path) -> String {
    p.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_neighbours_are_read_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("oppnet-live.jsonl");
        for s in ["", ".1", ".2"] {
            std::fs::write(format!("{}{s}", base.display()), "").unwrap();
        }
        let got = expand(std::slice::from_ref(&base), true);
        let names: Vec<String> = got
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            ["oppnet-live.jsonl.2", "oppnet-live.jsonl.1", "oppnet-live.jsonl"]
        );
        assert_eq!(expand(std::slice::from_ref(&base), false), vec![base]);
    }

    #[test]
    fn a_report_over_a_written_log_prints_the_tables() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("oppnet-live.jsonl");
        std::fs::write(
            &base,
            "{\"ev\":\"open\",\"v\":1,\"model\":\"ab\"}\n{\"v\":1,\"t\":10,\"w\":3,\"u\":1,\"o\":\"c1-aa\",\"s\":[[1,1,1,0,1,1,1,1000,1000,2000,0,0]]}\n",
        )
        .unwrap();
        assert_eq!(report(std::slice::from_ref(&base), false, true), ExitCode::SUCCESS);
        let empty = dir.path().join("empty.jsonl");
        std::fs::write(&empty, "").unwrap();
        assert_eq!(report(&[empty], false, true), ExitCode::FAILURE);
        assert_eq!(report(&[dir.path().join("none.jsonl")], false, true), ExitCode::FAILURE);
    }
}
