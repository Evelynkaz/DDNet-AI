//! `ddnet-ai` will be the single entry point for the bot: playing live, running the offline
//! arena, evaluating brains, training, tracing, and serving the web UI (see `docs/PLAN.md`
//! §1.1). Most of that doesn't exist yet; the `trace` subcommand group (task 1.2) is the first
//! real one — synthetic maps, scenario generation, and trace comparison for the DDNet physics
//! parity work (see `docs/formats.md`).

mod fly_cmd;
mod map_cmd;
mod trace_cmd;
mod web_cmd;

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
    /// The fly's inference engine (phase 7.1): bench and info tooling for a compiled `.flyg`.
    Fly(fly_cmd::FlyArgs),
    /// Real DDNet `.map` file inspection (task 1.4, `ddai-map`).
    Map(map_cmd::MapArgs),
    /// Starts the bot's own web server (login + status page), listening on loopback only.
    Web(web_cmd::WebArgs),
    /// Generates (and stores the argon2id hash of) the web UI's owner password.
    WebPasswd(web_cmd::WebPasswdArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        None => ExitCode::SUCCESS,
        Some(Command::Trace(args)) => trace_cmd::run(args),
        Some(Command::Fly(args)) => fly_cmd::run(args),
        Some(Command::Map(args)) => map_cmd::run(args),
        Some(Command::Web(args)) => web_cmd::run_web(args),
        Some(Command::WebPasswd(args)) => web_cmd::run_web_passwd(args),
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
                super::trace_cmd::TraceCommand::ExportMap { recipe, map, out } => {
                    assert_eq!(recipe.as_deref(), Some("arena"));
                    assert_eq!(map, None);
                    assert_eq!(out, std::path::PathBuf::from("map.rawmap"));
                }
                other => panic!("expected ExportMap, got {other:?}"),
            },
            other => panic!("expected Trace command, got {other:?}"),
        }
    }

    #[test]
    fn trace_export_map_with_map_flag_parses() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "trace",
            "export-map",
            "--map",
            "input.map",
            "--out",
            "map.rawmap",
        ])
        .expect("trace export-map --map should parse");
        match cli.command {
            Some(super::Command::Trace(args)) => match args.command {
                super::trace_cmd::TraceCommand::ExportMap { recipe, map, out } => {
                    assert_eq!(recipe, None);
                    assert_eq!(map, Some(std::path::PathBuf::from("input.map")));
                    assert_eq!(out, std::path::PathBuf::from("map.rawmap"));
                }
                other => panic!("expected ExportMap, got {other:?}"),
            },
            other => panic!("expected Trace command, got {other:?}"),
        }
    }

    #[test]
    fn trace_export_map_rejects_both_recipe_and_map() {
        let result = Cli::try_parse_from([
            "ddnet-ai",
            "trace",
            "export-map",
            "--recipe",
            "arena",
            "--map",
            "input.map",
            "--out",
            "map.rawmap",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn map_info_parses() {
        let cli = Cli::try_parse_from(["ddnet-ai", "map", "info", "input.map"]).expect("map info should parse");
        match cli.command {
            Some(super::Command::Map(args)) => match args.command {
                super::map_cmd::MapCommand::Info { map } => {
                    assert_eq!(map, std::path::PathBuf::from("input.map"));
                }
            },
            other => panic!("expected Map command, got {other:?}"),
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

    #[test]
    fn fly_info_parses() {
        let cli =
            Cli::try_parse_from(["ddnet-ai", "fly", "info", "--flyg", "fly-S-v1.flyg"]).expect("fly info should parse");
        match cli.command {
            Some(super::Command::Fly(args)) => match args.command {
                super::fly_cmd::FlyCommand::Info { flyg } => {
                    assert_eq!(flyg, std::path::PathBuf::from("fly-S-v1.flyg"));
                }
                other => panic!("expected Info, got {other:?}"),
            },
            other => panic!("expected Fly command, got {other:?}"),
        }
    }

    #[test]
    fn fly_bench_parses_with_defaults() {
        let cli = Cli::try_parse_from(["ddnet-ai", "fly", "bench", "--flyg", "fly-M-v1.flyg"])
            .expect("fly bench should parse");
        match cli.command {
            Some(super::Command::Fly(args)) => match args.command {
                super::fly_cmd::FlyCommand::Bench {
                    flyg,
                    substeps,
                    decisions,
                    tick_ms,
                    seed,
                } => {
                    assert_eq!(flyg, std::path::PathBuf::from("fly-M-v1.flyg"));
                    assert_eq!(substeps, 4);
                    assert_eq!(decisions, 500);
                    assert_eq!(tick_ms, 40);
                    assert_eq!(seed, 42);
                }
                other => panic!("expected Bench, got {other:?}"),
            },
            other => panic!("expected Fly command, got {other:?}"),
        }
    }

    #[test]
    fn fly_bench_overrides_parse() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "fly",
            "bench",
            "--flyg",
            "fly-S-v1.flyg",
            "--substeps",
            "1",
            "--decisions",
            "10",
            "--tick-ms",
            "20",
            "--seed",
            "7",
        ])
        .expect("fly bench with overrides should parse");
        match cli.command {
            Some(super::Command::Fly(args)) => match args.command {
                super::fly_cmd::FlyCommand::Bench {
                    substeps,
                    decisions,
                    tick_ms,
                    seed,
                    ..
                } => {
                    assert_eq!(substeps, 1);
                    assert_eq!(decisions, 10);
                    assert_eq!(tick_ms, 20);
                    assert_eq!(seed, 7);
                }
                other => panic!("expected Bench, got {other:?}"),
            },
            other => panic!("expected Fly command, got {other:?}"),
        }
    }

    #[test]
    fn fly_bench_rejects_missing_required_flyg_flag() {
        let result = Cli::try_parse_from(["ddnet-ai", "fly", "bench"]);
        assert!(result.is_err());
    }

    #[test]
    fn web_parses_with_defaults() {
        let cli = Cli::try_parse_from(["ddnet-ai", "web"]).expect("web should parse with no flags");
        match cli.command {
            Some(super::Command::Web(args)) => {
                assert_eq!(args.listen, "127.0.0.1:7788".parse().unwrap());
                assert!(args.data_dir.is_none());
                assert!(!args.trust_proxy);
                assert!(!args.i_know_this_is_public);
                assert!(!args.cookie_secure);
            }
            other => panic!("expected Web command, got {other:?}"),
        }
    }

    #[test]
    fn web_parses_all_flags() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "web",
            "--listen",
            "127.0.0.1:9999",
            "--data-dir",
            "/tmp/ddai-data",
            "--trust-proxy",
            "--cookie-secure",
        ])
        .expect("web should parse with flags");
        match cli.command {
            Some(super::Command::Web(args)) => {
                assert_eq!(args.listen, "127.0.0.1:9999".parse().unwrap());
                assert_eq!(args.data_dir, Some(std::path::PathBuf::from("/tmp/ddai-data")));
                assert!(args.trust_proxy);
                assert!(!args.i_know_this_is_public);
                assert!(args.cookie_secure);
            }
            other => panic!("expected Web command, got {other:?}"),
        }
    }

    #[test]
    fn web_rejects_non_socket_addr_listen() {
        let result = Cli::try_parse_from(["ddnet-ai", "web", "--listen", "not-an-address"]);
        assert!(result.is_err());
    }

    #[test]
    fn web_passwd_parses_with_defaults() {
        let cli = Cli::try_parse_from(["ddnet-ai", "web-passwd"]).expect("web-passwd should parse");
        match cli.command {
            Some(super::Command::WebPasswd(args)) => {
                assert!(!args.show);
                assert!(args.data_dir.is_none());
            }
            other => panic!("expected WebPasswd command, got {other:?}"),
        }
    }

    #[test]
    fn web_passwd_parses_show_flag() {
        let cli = Cli::try_parse_from(["ddnet-ai", "web-passwd", "--show"]).expect("web-passwd --show should parse");
        match cli.command {
            Some(super::Command::WebPasswd(args)) => assert!(args.show),
            other => panic!("expected WebPasswd command, got {other:?}"),
        }
    }
}
