//! Records the git commit of the source tree the binary is built from (`DDAI_BUILD_GIT_COMMIT`),
//! so run records (`ddnet-ai arena`) name the code that produced them, not the directory the CLI
//! happens to be started from.

use std::path::PathBuf;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Windows gives the main thread of a binary 1 MiB of stack by default, Linux 8 MiB. The commands run their work on the main thread and
/// were only ever exercised with the Linux size, so the Windows binary asks the linker for the same 8 MiB (task 5.5a, D-127).
fn main_thread_stack() {
    const EIGHT_MIB: u32 = 8 * 1024 * 1024;
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
        Ok("msvc") => println!("cargo:rustc-link-arg-bins=/STACK:{EIGHT_MIB}"),
        Ok("gnu") => println!("cargo:rustc-link-arg-bins=-Wl,--stack,{EIGHT_MIB}"),
        _ => {}
    }
}

fn main() {
    main_thread_stack();
    let commit = git(&["rev-parse", "HEAD"])
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=DDAI_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rerun-if-changed=build.rs");
    // Rebuild when HEAD moves (checkout, or a commit: the reflog is appended on both).
    for name in ["HEAD", "logs/HEAD"] {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            let path = PathBuf::from(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
