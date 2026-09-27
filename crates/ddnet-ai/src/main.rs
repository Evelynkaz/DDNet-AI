//! `ddnet-ai` will be the single entry point for the bot: playing live, running the offline
//! arena, evaluating brains, training, tracing, and serving the web UI (see `docs/PLAN.md`
//! §1.1). None of that exists yet — this is just the command-line skeleton so the binary,
//! `--version` and `--help` are in place before real subcommands are added.

use clap::Parser;

/// DDNet-AI: a block-mode bot for DDNet, rewritten in Rust.
#[derive(Debug, Parser)]
#[command(name = "ddnet-ai", version, about, long_about = None)]
struct Cli {}

fn main() {
    let Cli {} = Cli::parse();
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
}
