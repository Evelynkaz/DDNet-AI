//! `ddai-connectome`: fetch, verify, inspect and compact-table-ify the MaleCNS connectome. See
//! `crates/ddai-connectome/README.md` for usage. Deliberately not a dependency of `ddnet-ai` —
//! this is the only place the heavy `arrow` dependency links into.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use ddai_connectome::fetch::{self, FetchAction, FetchOptions};
use ddai_connectome::inspect;
use ddai_connectome::stats;
use ddai_connectome::subgraph;
use ddai_connectome::tables;

/// Fetch, verify and read the MaleCNS connectome (Arrow Feather) into compact internal tables.
#[derive(Debug, Parser)]
#[command(name = "ddai-connectome", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Download and verify the files listed in a manifest (network access).
    Fetch {
        /// Path to `manifests/connectome.toml`.
        #[arg(long)]
        manifest: PathBuf,
        /// Directory to download into (created if missing).
        #[arg(long)]
        dest: PathBuf,
        /// Write computed sha256 into the manifest for any entry that doesn't have one yet.
        #[arg(long)]
        update_sha256: bool,
    },
    /// Print a Feather (Arrow IPC) file's schema, row count and compression codec.
    Inspect {
        /// Path to the `.feather` file.
        file: PathBuf,
    },
    /// Stream the raw Feather files into compact postcard+zstd tables.
    BuildTables {
        /// Directory containing the raw `.feather` files (as `fetch` downloaded them).
        #[arg(long)]
        raw: PathBuf,
        /// Directory to write `connectome.tables` into (created if missing).
        #[arg(long)]
        out: PathBuf,
    },
    /// Read the compact tables and write a statistics report (in Russian).
    Stats {
        /// Directory containing `connectome.tables` (as `build-tables` wrote it).
        #[arg(long)]
        tables: PathBuf,
        /// Report path to write (Markdown).
        #[arg(long)]
        out: PathBuf,
    },
    /// Select the fly's connectome subgraph and compile it into a `.flyg` file (task 6.3).
    BuildSubgraph {
        /// Path to `connectome.tables` (or its containing directory).
        #[arg(long)]
        tables: PathBuf,
        /// Path to a selection config (e.g. `configs/fly/S.toml`).
        #[arg(long)]
        config: PathBuf,
        /// `.flyg` file to write.
        #[arg(long)]
        out: PathBuf,
        /// Report path to write (Markdown).
        #[arg(long)]
        report: PathBuf,
    },
    /// Print a summary of a `.flyg` file (offline; validates it first).
    FlygInfo {
        /// Path to the `.flyg` file.
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Fetch {
            manifest,
            dest,
            update_sha256,
        } => cmd_fetch(manifest, dest, update_sha256),
        Command::Inspect { file } => cmd_inspect(&file),
        Command::BuildTables { raw, out } => cmd_build_tables(&raw, &out),
        Command::Stats { tables, out } => cmd_stats(&tables, &out),
        Command::BuildSubgraph {
            tables,
            config,
            out,
            report,
        } => cmd_build_subgraph(&tables, &config, &out, &report),
        Command::FlygInfo { file } => cmd_flyg_info(&file),
    }
}

fn cmd_fetch(manifest: PathBuf, dest: PathBuf, update_sha256: bool) -> Result<()> {
    let outcomes = fetch::run_fetch(&FetchOptions {
        manifest_path: manifest,
        dest,
        update_sha256,
    })?;
    for outcome in &outcomes {
        match &outcome.action {
            FetchAction::AlreadyPresent => {
                println!("{}: already downloaded and verified, nothing to do", outcome.name);
                if outcome.sha256_written_to_manifest {
                    println!("  wrote sha256 into the manifest (--update-sha256)");
                }
            }
            FetchAction::Downloaded {
                resumed_from,
                performed_get,
                hashes,
            } => {
                if !performed_get {
                    println!(
                        "{}: found a complete .part from a previous run ({} bytes), verified and finalized (no download this run)",
                        outcome.name, hashes.size
                    );
                } else if *resumed_from > 0 {
                    println!(
                        "{}: resumed from byte {resumed_from}, downloaded to {} bytes total, verified",
                        outcome.name, hashes.size
                    );
                } else {
                    println!("{}: downloaded {} bytes, verified", outcome.name, hashes.size);
                }
                println!("  md5={} sha256={}", hashes.md5_hex, hashes.sha256_hex);
                if outcome.sha256_written_to_manifest {
                    println!("  wrote sha256 into the manifest (--update-sha256)");
                }
            }
        }
    }
    Ok(())
}

fn cmd_inspect(file: &std::path::Path) -> Result<()> {
    let report = inspect::inspect_file(file)?;
    println!("{}", file.display());
    println!("schema ({} columns):", report.fields.len());
    for f in &report.fields {
        println!("  {}: {}", f.name, f.data_type);
    }
    println!("row count: {}", report.row_count);
    println!("batches: {}", report.num_batches);
    println!("compression: {}", report.compression.as_deref().unwrap_or("(none)"));
    Ok(())
}

fn cmd_build_tables(raw: &std::path::Path, out: &std::path::Path) -> Result<()> {
    let summary = tables::run_build_tables(raw, out)?;
    println!("neurons: {} (Traced: {})", summary.neurons, summary.traced_neurons);
    println!("types: {}", summary.types);
    println!(
        "edges: {} (autapses dropped: {})",
        summary.edges, summary.autapses_dropped
    );
    println!("wall time: {:.1}s", summary.wall_time.as_secs_f64());
    println!("wrote {}", summary.output_path.display());
    Ok(())
}

fn cmd_stats(tables_dir: &std::path::Path, out: &std::path::Path) -> Result<()> {
    let summary = stats::run_stats(tables_dir, out)?;
    println!(
        "neurons: {} (Traced: {})",
        summary.total_neurons, summary.traced_neurons
    );
    println!("DN: {} neurons / {} types", summary.dn.neurons, summary.dn.types);
    println!("VPN: {} neurons / {} types", summary.vpn.neurons, summary.vpn.types);
    println!("AN: {} neurons / {} types", summary.an.neurons, summary.an.types);
    println!(
        "edges: {} (>=3: {}, >=5: {}, >=10: {}), autapses dropped: {}",
        summary.edges_total,
        summary.edges_weight_ge3,
        summary.edges_weight_ge5,
        summary.edges_weight_ge10,
        summary.autapses_dropped
    );
    println!("report written to {}", out.display());
    Ok(())
}

fn cmd_build_subgraph(
    tables: &std::path::Path,
    config: &std::path::Path,
    out: &std::path::Path,
    report: &std::path::Path,
) -> Result<()> {
    let summary = subgraph::run_build_subgraph(tables, config, out, report)?;
    let rc = &summary.role_counts;
    println!(
        "neurons: {} (input_visual: {}, input_ascending: {}, hidden: {}, output: {})",
        summary.neurons_total, rc.input_visual, rc.input_ascending, rc.hidden, rc.output
    );
    println!("edges: {}", summary.num_edges);
    println!("wall time: {:.1}s", summary.wall_time.as_secs_f64());
    match summary.peak_rss_kb {
        Some(kb) => println!("peak RSS: {:.0} MiB", kb as f64 / 1024.0),
        None => println!("peak RSS: n/a (could not read /proc/self/status)"),
    }
    println!("wrote {}", summary.flyg_path.display());
    println!("  sha256={}", summary.flyg_sha256);
    println!("wrote {}", summary.report_path.display());
    Ok(())
}

fn cmd_flyg_info(file: &std::path::Path) -> Result<()> {
    let flyg = ddai_flyg::load(file)?;
    println!("{}", file.display());
    println!(
        "format_version={} generator={}",
        flyg.header.format_version, flyg.header.generator_version
    );
    println!(
        "source_tables_sha256={} config_sha256={}",
        flyg.header.source_tables_sha256, flyg.header.config_sha256
    );
    let rc = &flyg.summary.neurons_by_role;
    let total = rc.input_visual + rc.input_ascending + rc.hidden + rc.output;
    println!(
        "neurons: {total} (input_visual: {}, input_ascending: {}, hidden: {}, output: {})",
        rc.input_visual, rc.input_ascending, rc.hidden, rc.output
    );
    println!("types: {}", flyg.summary.num_types);
    println!("edges: {}", flyg.summary.num_edges);
    println!(
        "type_pairs: {} (shared parameters: {})",
        flyg.summary.num_type_pairs, flyg.summary.shared_param_count
    );
    let sc = &flyg.summary.sign_counts;
    println!(
        "signs: +1={} -1={} 0={} (uncertain types: {})",
        sc.excitatory, sc.inhibitory, sc.neutral, flyg.summary.uncertain_types
    );
    println!("output groups: {}", flyg.output_groups.len());
    for g in &flyg.output_groups {
        println!("  {}: {} neurons", g.action, g.members.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_help_parses() {
        let err = Cli::try_parse_from(["ddai-connectome", "--help"]).expect_err("--help short-circuits");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
    }

    #[test]
    fn fetch_requires_manifest_and_dest() {
        assert!(Cli::try_parse_from(["ddai-connectome", "fetch"]).is_err());
        let ok = Cli::try_parse_from(["ddai-connectome", "fetch", "--manifest", "m.toml", "--dest", "d"]);
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn inspect_takes_a_positional_file() {
        let ok = Cli::try_parse_from(["ddai-connectome", "inspect", "a.feather"]);
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn build_tables_requires_raw_and_out() {
        assert!(Cli::try_parse_from(["ddai-connectome", "build-tables"]).is_err());
        let ok = Cli::try_parse_from(["ddai-connectome", "build-tables", "--raw", "r", "--out", "o"]);
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn stats_requires_tables_and_out() {
        assert!(Cli::try_parse_from(["ddai-connectome", "stats"]).is_err());
        let ok = Cli::try_parse_from(["ddai-connectome", "stats", "--tables", "t", "--out", "report.md"]);
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn build_subgraph_requires_tables_config_out_and_report() {
        assert!(Cli::try_parse_from(["ddai-connectome", "build-subgraph"]).is_err());
        assert!(
            Cli::try_parse_from([
                "ddai-connectome",
                "build-subgraph",
                "--tables",
                "t",
                "--config",
                "c.toml"
            ])
            .is_err()
        );
        let ok = Cli::try_parse_from([
            "ddai-connectome",
            "build-subgraph",
            "--tables",
            "t",
            "--config",
            "c.toml",
            "--out",
            "out.flyg",
            "--report",
            "report.md",
        ]);
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn flyg_info_takes_a_positional_file() {
        let ok = Cli::try_parse_from(["ddai-connectome", "flyg-info", "a.flyg"]);
        assert!(ok.is_ok(), "{ok:?}");
    }
}
