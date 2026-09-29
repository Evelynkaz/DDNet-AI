//! `ddnet-ai` will be the single entry point for the bot: playing live, running the offline
//! arena, evaluating brains, training, tracing, and serving the web UI (see `docs/PLAN.md`
//! §1.1). Most of that doesn't exist yet; the `trace` subcommand group (task 1.2) is the first
//! real one — synthetic maps, scenario generation, and trace comparison for the DDNet physics
//! parity work (see `docs/formats.md`).

mod arena_cmd;
mod dataset_cmd;
mod demo_cmd;
mod fly_cmd;
mod map_cmd;
mod play_cmd;
mod rec_cmd;
mod record_cmd;
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
    /// Offline evaluation arena (task 8.1): N-player block matches on the bit-exact world,
    /// W:L:D:T with Wilson CIs, JSONL + summary output, technique scenarios.
    Arena(arena_cmd::ArenaArgs),
    /// Real DDNet `.map` file inspection (task 1.4, `ddai-map`).
    Map(map_cmd::MapArgs),
    /// Real DDNet `.demo` file inspection (task 8.4b, `ddai-demo`): info/dump/stats.
    Demo(demo_cmd::DemoArgs),
    /// Human-play dataset (task 8.4c, `ddai-dataset`): `from-demos`, `info`, `show`, `locate`.
    Dataset(dataset_cmd::DatasetArgs),
    /// Starts the bot's own web server (login + status page), listening on loopback only.
    Web(web_cmd::WebArgs),
    /// Generates (and stores the argon2id hash of) the web UI's owner password.
    WebPasswd(web_cmd::WebPasswdArgs),
    /// Connects to a real DDNet 20.x server and plays with a trivial built-in brain (task 2.3;
    /// the real bot's brain is a later phase) — `--server 127.0.0.1:8303 --brain idle|circle`.
    Play(play_cmd::PlayArgs),
    /// Observer recorder (task 8.4a): connects as a pure spectator (never sends non-neutral input
    /// or chat) and records the session into rec v1 — `--server <addr> --name Muha --duration <s>
    /// --out ~/aiddnet/data/recordings/<date>/`.
    Record(record_cmd::RecordArgs),
    /// Offline rec v1 tooling (task 8.4a): `rec inspect|reconstruct|anonymize`.
    Rec(rec_cmd::RecArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        None => ExitCode::SUCCESS,
        Some(Command::Trace(args)) => trace_cmd::run(args),
        Some(Command::Fly(args)) => fly_cmd::run(args),
        Some(Command::Arena(args)) => arena_cmd::run(args),
        Some(Command::Map(args)) => map_cmd::run(args),
        Some(Command::Demo(args)) => demo_cmd::run(args),
        Some(Command::Dataset(args)) => dataset_cmd::run(args),
        Some(Command::Web(args)) => web_cmd::run_web(args),
        Some(Command::WebPasswd(args)) => web_cmd::run_web_passwd(args),
        Some(Command::Play(args)) => play_cmd::run(args),
        Some(Command::Record(args)) => record_cmd::run(args),
        Some(Command::Rec(args)) => rec_cmd::run(args),
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
    fn demo_info_parses() {
        let cli = Cli::try_parse_from(["ddnet-ai", "demo", "info", "game.demo"]).expect("demo info should parse");
        match cli.command {
            Some(super::Command::Demo(args)) => match args.command {
                super::demo_cmd::DemoCommand::Info { file } => {
                    assert_eq!(file, std::path::PathBuf::from("game.demo"));
                }
                other => panic!("expected Info, got {other:?}"),
            },
            other => panic!("expected Demo command, got {other:?}"),
        }
    }

    #[test]
    fn demo_dump_parses_flags() {
        // `--raw` and `--anonymize` conflict (review round 1 finding F3) — tested separately
        // below, so this only combines `--raw` with `--limit`.
        let cli = Cli::try_parse_from(["ddnet-ai", "demo", "dump", "game.demo", "--raw", "--limit", "10"])
            .expect("demo dump should parse");
        match cli.command {
            Some(super::Command::Demo(args)) => match args.command {
                super::demo_cmd::DemoCommand::Dump {
                    file,
                    anonymize,
                    raw,
                    limit,
                } => {
                    assert_eq!(file, std::path::PathBuf::from("game.demo"));
                    assert!(!anonymize);
                    assert!(raw);
                    assert_eq!(limit, Some(10));
                }
                other => panic!("expected Dump, got {other:?}"),
            },
            other => panic!("expected Demo command, got {other:?}"),
        }
    }

    #[test]
    fn demo_dump_accepts_anonymize_alone() {
        let cli = Cli::try_parse_from(["ddnet-ai", "demo", "dump", "game.demo", "--anonymize"]).expect("should parse");
        match cli.command {
            Some(super::Command::Demo(args)) => match args.command {
                super::demo_cmd::DemoCommand::Dump { anonymize, raw, .. } => {
                    assert!(anonymize);
                    assert!(!raw);
                }
                other => panic!("expected Dump, got {other:?}"),
            },
            other => panic!("expected Demo command, got {other:?}"),
        }
    }

    #[test]
    fn demo_dump_rejects_raw_and_anonymize_together() {
        // Review round 1 finding F3: `--raw` prints unredacted `ClientInfo` ints regardless of
        // `--anonymize`, so the two must be rejected together rather than one silently winning.
        let result = Cli::try_parse_from(["ddnet-ai", "demo", "dump", "game.demo", "--raw", "--anonymize"]);
        assert!(result.is_err(), "expected --raw and --anonymize to conflict");
    }

    #[test]
    fn demo_stats_parses() {
        let cli = Cli::try_parse_from(["ddnet-ai", "demo", "stats", "demos/", "--anonymize"])
            .expect("demo stats should parse");
        match cli.command {
            Some(super::Command::Demo(args)) => match args.command {
                super::demo_cmd::DemoCommand::Stats { path, anonymize } => {
                    assert_eq!(path, std::path::PathBuf::from("demos/"));
                    assert!(anonymize);
                }
                other => panic!("expected Stats, got {other:?}"),
            },
            other => panic!("expected Demo command, got {other:?}"),
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
                assert!(args.replay.is_none());
                assert!(args.maps_dir.is_empty());
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

    /// Task 5.2a: `--replay` (single value) and repeatable `--maps-dir`.
    #[test]
    fn web_parses_replay_and_maps_dir_flags() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "web",
            "--replay",
            "/tmp/traces",
            "--maps-dir",
            "/tmp/maps-a",
            "--maps-dir",
            "/tmp/maps-b",
        ])
        .expect("web should parse with --replay/--maps-dir");
        match cli.command {
            Some(super::Command::Web(args)) => {
                assert_eq!(args.replay, Some(std::path::PathBuf::from("/tmp/traces")));
                assert_eq!(
                    args.maps_dir,
                    vec![
                        std::path::PathBuf::from("/tmp/maps-a"),
                        std::path::PathBuf::from("/tmp/maps-b"),
                    ]
                );
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

    #[test]
    fn fly_train_demo_parses_with_defaults() {
        let cli = Cli::try_parse_from(["ddnet-ai", "fly", "train-demo", "--flyg", "fly-S-v1.flyg"])
            .expect("fly train-demo should parse");
        match cli.command {
            Some(super::Command::Fly(args)) => match args.command {
                super::fly_cmd::FlyCommand::TrainDemo {
                    flyg,
                    steps,
                    batch_size,
                    t_decisions,
                    readout_decisions,
                    left_action,
                    right_action,
                    out,
                    ..
                } => {
                    assert_eq!(flyg, std::path::PathBuf::from("fly-S-v1.flyg"));
                    assert_eq!(steps, 300);
                    assert_eq!(batch_size, 24);
                    assert_eq!(t_decisions, 6);
                    assert_eq!(readout_decisions, 2);
                    assert_eq!(left_action, "direction_left");
                    assert_eq!(right_action, "direction_right");
                    assert_eq!(out, None);
                }
                other => panic!("expected TrainDemo, got {other:?}"),
            },
            other => panic!("expected Fly command, got {other:?}"),
        }
    }

    #[test]
    fn fly_train_demo_overrides_parse() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "fly",
            "train-demo",
            "--flyg",
            "fly-S-v1.flyg",
            "--steps",
            "10",
            "--batch-size",
            "4",
            "--out",
            "out.csv",
        ])
        .expect("fly train-demo with overrides should parse");
        match cli.command {
            Some(super::Command::Fly(args)) => match args.command {
                super::fly_cmd::FlyCommand::TrainDemo {
                    steps, batch_size, out, ..
                } => {
                    assert_eq!(steps, 10);
                    assert_eq!(batch_size, 4);
                    assert_eq!(out, Some(std::path::PathBuf::from("out.csv")));
                }
                other => panic!("expected TrainDemo, got {other:?}"),
            },
            other => panic!("expected Fly command, got {other:?}"),
        }
    }

    #[test]
    fn play_parses_with_defaults() {
        let cli = Cli::try_parse_from(["ddnet-ai", "play", "--server", "127.0.0.1:8303"]).expect("play should parse");
        match cli.command {
            Some(super::Command::Play(args)) => {
                assert_eq!(args.server, "127.0.0.1:8303".parse().unwrap());
                assert_eq!(args.name, "ddai-bot");
                assert!(matches!(args.brain, super::play_cmd::Brain::Idle));
                assert_eq!(args.duration, 30);
                assert!(args.data_dir.is_none());
            }
            other => panic!("expected Play command, got {other:?}"),
        }
    }

    #[test]
    fn play_parses_all_flags() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "play",
            "--server",
            "127.0.0.1:8303",
            "--name",
            "test-bot",
            "--brain",
            "circle",
            "--duration",
            "60",
            "--data-dir",
            "/tmp/ddai-data",
        ])
        .expect("play should parse with flags");
        match cli.command {
            Some(super::Command::Play(args)) => {
                assert_eq!(args.name, "test-bot");
                assert!(matches!(args.brain, super::play_cmd::Brain::Circle));
                assert_eq!(args.duration, 60);
                assert_eq!(args.data_dir, Some(std::path::PathBuf::from("/tmp/ddai-data")));
            }
            other => panic!("expected Play command, got {other:?}"),
        }
    }

    #[test]
    fn play_rejects_missing_server() {
        let result = Cli::try_parse_from(["ddnet-ai", "play"]);
        assert!(result.is_err());
    }

    #[test]
    fn play_rejects_invalid_brain() {
        let result = Cli::try_parse_from(["ddnet-ai", "play", "--server", "127.0.0.1:8303", "--brain", "planner"]);
        assert!(result.is_err());
    }

    /// Task 8.4a: `--brain random-scripted`, `--seed`, `--input-log`.
    #[test]
    fn play_parses_random_scripted_brain_with_seed_and_input_log() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "play",
            "--server",
            "127.0.0.1:8303",
            "--brain",
            "random-scripted",
            "--seed",
            "7",
            "--input-log",
            "/tmp/inputs.jsonl",
        ])
        .expect("play --brain random-scripted should parse");
        match cli.command {
            Some(super::Command::Play(args)) => {
                assert!(matches!(args.brain, super::play_cmd::Brain::RandomScripted));
                assert_eq!(args.seed, 7);
                assert_eq!(args.input_log, Some(std::path::PathBuf::from("/tmp/inputs.jsonl")));
            }
            other => panic!("expected Play command, got {other:?}"),
        }
    }

    #[test]
    fn play_seed_defaults_when_omitted() {
        let cli = Cli::try_parse_from(["ddnet-ai", "play", "--server", "127.0.0.1:8303"]).expect("play should parse");
        match cli.command {
            Some(super::Command::Play(args)) => {
                assert_eq!(args.seed, 42);
                assert_eq!(args.input_log, None);
            }
            other => panic!("expected Play command, got {other:?}"),
        }
    }

    #[test]
    fn record_parses_with_defaults() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "record",
            "--server",
            "127.0.0.1:8303",
            "--out",
            "/tmp/recordings",
        ])
        .expect("record should parse");
        match cli.command {
            Some(super::Command::Record(args)) => {
                assert_eq!(args.server, "127.0.0.1:8303".parse().unwrap());
                assert_eq!(args.name, "Muha");
                assert_eq!(args.duration, 30);
                assert_eq!(args.out, std::path::PathBuf::from("/tmp/recordings"));
                assert_eq!(args.show_distance, 2_000_000);
                assert_eq!(args.input_log, None);
                assert_eq!(args.live_servers, None);
            }
            other => panic!("expected Record command, got {other:?}"),
        }
    }

    #[test]
    fn record_rejects_missing_out() {
        let result = Cli::try_parse_from(["ddnet-ai", "record", "--server", "127.0.0.1:8303"]);
        assert!(result.is_err());
    }

    #[test]
    fn record_parses_all_flags() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "record",
            "--server",
            "45.141.57.35:8308",
            "--name",
            "TestObserver",
            "--duration",
            "60",
            "--out",
            "/tmp/recordings",
            "--live-servers",
            "/tmp/live-servers.toml",
            "--input-log",
            "/tmp/inputs.jsonl",
            "--show-distance",
            "5000",
        ])
        .expect("record should parse with all flags");
        match cli.command {
            Some(super::Command::Record(args)) => {
                assert_eq!(args.name, "TestObserver");
                assert_eq!(args.duration, 60);
                assert_eq!(
                    args.live_servers,
                    Some(std::path::PathBuf::from("/tmp/live-servers.toml"))
                );
                assert_eq!(args.input_log, Some(std::path::PathBuf::from("/tmp/inputs.jsonl")));
                assert_eq!(args.show_distance, 5000);
            }
            other => panic!("expected Record command, got {other:?}"),
        }
    }

    #[test]
    fn rec_inspect_parses() {
        let cli = Cli::try_parse_from(["ddnet-ai", "rec", "inspect", "in.rec", "--verify"])
            .expect("rec inspect should parse");
        match cli.command {
            Some(super::Command::Rec(args)) => match args.command {
                super::rec_cmd::RecCommand::Inspect { recording, verify } => {
                    assert_eq!(recording, std::path::PathBuf::from("in.rec"));
                    assert!(verify);
                }
                other => panic!("expected Inspect, got {other:?}"),
            },
            other => panic!("expected Rec command, got {other:?}"),
        }
    }

    #[test]
    fn rec_reconstruct_parses_with_and_without_validate() {
        let cli = Cli::try_parse_from(["ddnet-ai", "rec", "reconstruct", "in.rec", "--out", "out.json"])
            .expect("rec reconstruct should parse");
        match cli.command {
            Some(super::Command::Rec(args)) => match args.command {
                super::rec_cmd::RecCommand::Reconstruct {
                    recording,
                    out,
                    validate,
                    client_id,
                    player_name,
                } => {
                    assert_eq!(recording, std::path::PathBuf::from("in.rec"));
                    assert_eq!(out, Some(std::path::PathBuf::from("out.json")));
                    assert_eq!(validate, None);
                    assert_eq!(client_id, None);
                    assert_eq!(player_name, None);
                }
                other => panic!("expected Reconstruct, got {other:?}"),
            },
            other => panic!("expected Rec command, got {other:?}"),
        }

        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "rec",
            "reconstruct",
            "in.rec",
            "--validate",
            "truth.jsonl",
            "--client-id",
            "2",
        ])
        .expect("rec reconstruct --validate should parse");
        match cli.command {
            Some(super::Command::Rec(args)) => match args.command {
                super::rec_cmd::RecCommand::Reconstruct {
                    validate, client_id, ..
                } => {
                    assert_eq!(validate, Some(std::path::PathBuf::from("truth.jsonl")));
                    assert_eq!(client_id, Some(2));
                }
                other => panic!("expected Reconstruct, got {other:?}"),
            },
            other => panic!("expected Rec command, got {other:?}"),
        }
    }

    #[test]
    fn rec_reconstruct_parses_player_name() {
        let cli = Cli::try_parse_from([
            "ddnet-ai",
            "rec",
            "reconstruct",
            "in.rec",
            "--validate",
            "truth.jsonl",
            "--player-name",
            "RSValid",
        ])
        .expect("rec reconstruct --player-name should parse");
        match cli.command {
            Some(super::Command::Rec(args)) => match args.command {
                super::rec_cmd::RecCommand::Reconstruct {
                    client_id, player_name, ..
                } => {
                    assert_eq!(client_id, None);
                    assert_eq!(player_name, Some("RSValid".to_string()));
                }
                other => panic!("expected Reconstruct, got {other:?}"),
            },
            other => panic!("expected Rec command, got {other:?}"),
        }
    }

    #[test]
    fn rec_anonymize_parses() {
        let cli = Cli::try_parse_from(["ddnet-ai", "rec", "anonymize", "in.rec", "--out", "out.rec"])
            .expect("rec anonymize should parse");
        match cli.command {
            Some(super::Command::Rec(args)) => match args.command {
                super::rec_cmd::RecCommand::Anonymize { recording, out } => {
                    assert_eq!(recording, std::path::PathBuf::from("in.rec"));
                    assert_eq!(out, std::path::PathBuf::from("out.rec"));
                }
                other => panic!("expected Anonymize, got {other:?}"),
            },
            other => panic!("expected Rec command, got {other:?}"),
        }
    }

    #[test]
    fn rec_requires_a_subcommand() {
        let result = Cli::try_parse_from(["ddnet-ai", "rec"]);
        assert!(result.is_err());
    }
}
