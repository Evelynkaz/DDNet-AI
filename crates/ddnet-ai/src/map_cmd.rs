//! Implementation of the `ddnet-ai map` subcommand group: thin CLI glue around `ddai-map` (task
//! 1.4) for inspecting a real DDNet `.map` file.

use clap::{Args, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Args)]
pub struct MapArgs {
    #[command(subcommand)]
    pub command: MapCommand,
}

#[derive(Debug, Subcommand)]
pub enum MapCommand {
    /// Prints size, sha256, crc32, dimensions, present physics layers, and Settings for a real
    /// DDNet `.map` file.
    Info { map: PathBuf },
}

pub fn run(args: MapArgs) -> ExitCode {
    match args.command {
        MapCommand::Info { map } => info(&map),
    }
}

fn info(path: &PathBuf) -> ExitCode {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let loaded = match ddai_map::load_map(&bytes) {
        Ok(loaded) => loaded,
        Err(e) => {
            eprintln!("failed to load {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let mut layers = vec!["game"];
    if loaded.data.front.is_some() {
        layers.push("front");
    }
    if loaded.data.tele.is_some() {
        layers.push("tele");
    }
    if loaded.data.speedup.is_some() {
        layers.push("speedup");
    }
    if loaded.data.switch.is_some() {
        layers.push("switch");
    }
    if loaded.data.tune.is_some() {
        layers.push("tune");
    }

    println!("size: {} bytes", loaded.size);
    println!("sha256: {}", hex(&loaded.sha256));
    println!("crc32: {:08x}", loaded.crc32);
    println!("dimensions: {}x{}", loaded.data.width, loaded.data.height);
    println!("layers: {}", layers.join(", "));
    if let Some(author) = &loaded.info.author {
        println!("author: {author}");
    }
    if let Some(version) = &loaded.info.version {
        println!("map version: {version}");
    }
    if let Some(credits) = &loaded.info.credits {
        println!("credits: {credits}");
    }
    if let Some(license) = &loaded.info.license {
        println!("license: {license}");
    }
    println!("settings ({}):", loaded.settings.len());
    for s in &loaded.settings {
        println!("  {s}");
    }
    ExitCode::SUCCESS
}

fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(64);
    for b in bytes {
        write!(out, "{b:02x}").expect("writing to a String cannot fail");
    }
    out
}
