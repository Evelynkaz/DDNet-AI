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

fn main() {
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
