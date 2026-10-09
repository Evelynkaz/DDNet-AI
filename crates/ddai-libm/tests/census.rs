//! Census of direct `std` libm calls in the crates whose behaviour must not depend on the platform (task 5.5a, D-127).
//!
//! On `linux-gnu` `f32::sin`, `f64::hypot`, ... reach glibc; on Windows they reach the UCRT, whose last bits differ. The crates below
//! therefore call `ddai_libm` for every transcendental function that has a port there (`sinf`, `cosf`, `atanf`, `atan2f`, `powf`,
//! `hypotf`, `log`, `atan2`, `pow`, `hypot`), and this test fails if a new direct call appears in their non-test code. A call that must
//! stay (a function with no port, e.g. `f64::sin`, in code that is not parity-relevant) carries `// libm-census: <why>` on its line or
//! the line above (a comment covers the next four lines).
//!
//! Scope: files under `src/` of the listed crates; everything from the first `#[cfg(test)]` line on is test code and is skipped (the
//! workspace convention puts the test module last). `exp`, `tanh` and friends are not listed: the neural-network crates use them and
//! are outside the scope (docs/DECISIONS.md D-127 lists them).

use std::path::{Path, PathBuf};

/// Crates whose non-test code is checked.
const CRATES: &[&str] = &[
    "ddai-physics",
    "ddai-world",
    "ddai-bot",
    "ddai-nav",
    "ddai-clip",
    "ddai-planner",
    "ddai-env",
    "ddai-trace",
    "ddai-recorder",
    "ddai-dataset",
    "ddnet-ai",
];

/// Method names that reach the C library's math through `std`.
const FORBIDDEN: &[&str] = &[
    "sin", "cos", "tan", "atan", "atan2", "asin", "acos", "powf", "hypot", "ln",
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The method calls `.name(` in `line` that are on the forbidden list, with the receiver text before the dot (to tell `x.ln()` from
/// `self.log(...)`).
fn hits(line: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    for name in FORBIDDEN {
        let needle = format!(".{name}(");
        let mut from = 0;
        while let Some(i) = line[from..].find(&needle) {
            let at = from + i;
            // `x.ln()` and `x.ln_1p()`: the name must end exactly at `(` (the needle includes it). A method on `self` is the type's own.
            let on_self = line[..at].ends_with("self");
            if !on_self {
                found.push(*name);
            }
            from = at + needle.len();
        }
    }
    found
}

#[test]
fn no_direct_std_libm_call_in_the_parity_relevant_crates() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut bad = Vec::new();
    let mut checked = 0usize;
    for krate in CRATES {
        let mut files = Vec::new();
        rust_files(&root.join(krate).join("src"), &mut files);
        assert!(!files.is_empty(), "{krate}: no sources found under {}", root.display());
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            checked += 1;
            // A `libm-census:` comment covers its own line and the next four (a reason may take a few lines).
            let mut covered = 0u32;
            for (n, line) in text.lines().enumerate() {
                if line.trim_start().starts_with("#[cfg(test)]") {
                    break;
                }
                if line.contains("libm-census:") {
                    covered = 5;
                }
                let marked = covered > 0;
                covered = covered.saturating_sub(1);
                if line.trim_start().starts_with("//") {
                    continue;
                }
                let code = line.split("//").next().unwrap_or("");
                if !hits(code).is_empty() && !marked {
                    bad.push(format!(
                        "{}:{}: {}",
                        file.strip_prefix(&root).unwrap().display(),
                        n + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
    assert!(checked > 100, "the census looked at only {checked} files");
    assert!(
        bad.is_empty(),
        "direct std libm calls in code that must give the same bits on every platform (use ddai_libm, or mark the line `// libm-census: <why>`):\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_census_pattern_sees_what_it_should() {
    assert_eq!(hits("let y = x.sin() + z.hypot(w);"), ["sin", "hypot"]);
    assert_eq!(hits("let y = (a.atan2(b)).ln();"), ["atan2", "ln"]);
    assert!(hits("self.sin(x);").is_empty());
    assert!(hits("let y = x.ln_1p();").is_empty());
    assert!(hits("let y = ddai_libm::sinf(x);").is_empty());
}
