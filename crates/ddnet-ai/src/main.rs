//! `ddnet-ai` will be the single entry point for the bot: playing live, running the offline
//! arena, evaluating brains, training, tracing, and serving the web UI (see `docs/PLAN.md`
//! §1.1). Most of that doesn't exist yet; the `trace` subcommand group (task 1.2) is the first
//! real one — synthetic maps, scenario generation, and trace comparison for the DDNet physics
//! parity work (see `docs/formats.md`).

mod trace_cmd;

use clap::{Parser, Subcommand};
use std::process::ExitCode;

/// DDNet-AI: a block-mode bot for DDNet, rewritten in Rust.
#[derive(Debug, Parser)]
#[command(name = "ddnet-ai", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Physics parity tooling: synthetic maps, scenario generation, trace comparison.
    Trace(trace_cmd::TraceArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        None => ExitCode::SUCCESS,
        Some(Command::Trace(args)) => trace_cmd::run(args),
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn no_args_parses_successfully() {
        let result = Cli::try_parse_from(["ddnet-ai"]);
        assert!(result.is_ok(), "expected empty args to parse, got {result:?}");
    }

    #[test]
    fn version_flag_is_handled_by_clap() {
        let err = Cli::try_parse_from(["ddnet-ai", "--version"]).expect_err("--version should short-circuit parsing");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(err.to_string().contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn help_flag_is_handled_by_clap() {
        let err = Cli::try_parse_from(["ddnet-ai", "--help"]).expect_err("--help should short-circuit parsing");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
    }

    #[test]
    fn unknown_flag_is_rejected() {
        let result = Cli::try_parse_from(["ddnet-ai", "--not-a-real-flag"]);
        assert!(result.is_err());
    }

    #[test]
    fn trace_export_map_parses() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "trace",
            "export-map",
            "--recipe",
            "arena",
            "--out",
            "map.rawmap",
        ])
        .expect("trace export-map should parse");
        match cli.command {
            Some(super::Command::Trace(args)) => match args.command {
                super::trace_cmd::TraceCommand::ExportMap { recipe, out } => {
                    assert_eq!(recipe, "arena");
                    assert_eq!(out, std::path::PathBuf::from("map.rawmap"));
                }
                other => panic!("expected ExportMap, got {other:?}"),
            },
            other => panic!("expected Trace command, got {other:?}"),
        }
    }

    #[test]
    fn trace_gen_scenario_parses() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "trace",
            "gen-scenario",
            "--recipe",
            "arena",
            "--seed",
            "7",
            "--ticks",
            "100",
            "--chars",
            "3",
            "--out",
            "s.scn",
        ])
        .expect("trace gen-scenario should parse");
        match cli.command {
            Some(super::Command::Trace(args)) => match args.command {
                super::trace_cmd::TraceCommand::GenScenario {
                    recipe,
                    seed,
                    ticks,
                    chars,
                    no_weak_hook,
                    tune,
                    out,
                } => {
                    assert_eq!(recipe, "arena");
                    assert_eq!(seed, 7);
                    assert_eq!(ticks, 100);
                    assert_eq!(chars, 3);
                    assert!(!no_weak_hook);
                    assert!(tune.is_empty());
                    assert_eq!(out, std::path::PathBuf::from("s.scn"));
                }
                other => panic!("expected GenScenario, got {other:?}"),
            },
            other => panic!("expected Trace command, got {other:?}"),
        }
    }

    #[test]
    fn trace_gen_scenario_no_weak_hook_and_tune_flags_parse() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "trace",
            "gen-scenario",
            "--recipe",
            "arena",
            "--seed",
            "7",
            "--ticks",
            "100",
            "--chars",
            "2",
            "--no-weak-hook",
            "--tune",
            "gravity=0",
            "--tune",
            "hook_length=38000",
            "--out",
            "s.scn",
        ])
        .expect("trace gen-scenario with --no-weak-hook/--tune should parse");
        match cli.command {
            Some(super::Command::Trace(args)) => match args.command {
                super::trace_cmd::TraceCommand::GenScenario { no_weak_hook, tune, .. } => {
                    assert!(no_weak_hook);
                    assert_eq!(tune, vec!["gravity=0".to_string(), "hook_length=38000".to_string()]);
                }
                other => panic!("expected GenScenario, got {other:?}"),
            },
            other => panic!("expected Trace command, got {other:?}"),
        }
    }

    #[test]
    fn trace_diff_parses() {
        let cli =
            Cli::try_parse_from(["ddnet-ai", "trace", "diff", "a.trace", "b.trace"]).expect("trace diff should parse");
        match cli.command {
            Some(super::Command::Trace(args)) => match args.command {
                super::trace_cmd::TraceCommand::Diff { a, b } => {
                    assert_eq!(a, std::path::PathBuf::from("a.trace"));
                    assert_eq!(b, std::path::PathBuf::from("b.trace"));
                }
                other => panic!("expected Diff, got {other:?}"),
            },
            other => panic!("expected Trace command, got {other:?}"),
        }
    }

    #[test]
    fn trace_hashes_parses() {
        let cli = Cli::try_parse_from(["ddnet-ai", "trace", "hashes", "t.trace", "--out", "h.json"])
            .expect("trace hashes should parse");
        match cli.command {
            Some(super::Command::Trace(args)) => match args.command {
                super::trace_cmd::TraceCommand::Hashes { trace, out } => {
                    assert_eq!(trace, std::path::PathBuf::from("t.trace"));
                    assert_eq!(out, std::path::PathBuf::from("h.json"));
                }
                other => panic!("expected Hashes, got {other:?}"),
            },
            other => panic!("expected Trace command, got {other:?}"),
        }
    }

    #[test]
    fn trace_gen_scenario_rejects_missing_required_flag() {
        let result = Cli::try_parse_from(["ddnet-ai", "trace", "gen-scenario", "--recipe", "arena"]);
        assert!(result.is_err());
    }
}
