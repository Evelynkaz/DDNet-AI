//! Census of direct `std` libm calls in the crates whose behaviour must not depend on the platform (task 5.5a, D-127).
//!
//! On `linux-gnu` `f32::sin`, `f64::hypot`, ... reach glibc; on Windows they reach the UCRT, whose last bits differ. The crates below
//! therefore call `ddai_libm` for every transcendental function that has a port there (`sinf`, `cosf`, `atanf`, `atan2f`, `powf`,
//! `hypotf`, `log`, `atan2`, `pow`, `hypot`), and this test fails if a new direct call appears in their non-test code. A call that must
//! stay (a function with no port, e.g. `f64::sin`, in code that is not parity-relevant) carries `// libm-census: <why>` on its line or
//! the line above (a comment covers the next four lines).
//!
//! Scope: files under `src/` of the listed crates. The test module (a `#[cfg(test)]` followed, after any further attributes and comments,
//! by `mod ...`) ends the production code; a `#[cfg(test)]` on anything else (a helper function, a `use`) does not, and a test-only
//! function's body is skipped. Checked: method calls `.name(` (not on `self`), and `f32::name` / `f64::name` calls or function pointers,
//! for sin, cos, tan, their inverses and hyperbolic forms, `sin_cos`, exp, exp2, exp_m1, ln, ln_1p, log, log2, log10, powf, hypot, cbrt.
//! The neural-network crates (`ddai-fly`, `ddai-train`, `ddai-controls`, `ddai-oppnet`) are outside the scope: docs/DECISIONS.md D-127
//! lists them as still on the host's libm.

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

/// Names of `f32`/`f64` methods that reach the C library's math through `std` (or that `std` implements
/// with it): called as `x.name(..)`, as `f32::name(..)`, or passed as `f32::name`.
const FORBIDDEN: &[&str] = &[
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sin_cos", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10", "powf", "hypot", "cbrt",
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

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The forbidden calls in `line`: `.name(` (method call; a method on `self` is the type's own), and `f32::name` / `f64::name`
/// (UFCS call or function pointer; the name must end at a non-identifier byte, so `f32::sinh_x` is not `sin`).
fn hits(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let bytes = line.as_bytes();
    for name in FORBIDDEN {
        let needle = format!(".{name}(");
        let mut from = 0;
        while let Some(i) = line[from..].find(&needle) {
            let at = from + i;
            if !line[..at].ends_with("self") {
                found.push(format!(".{name}("));
            }
            from = at + needle.len();
        }
        for ty in ["f32", "f64"] {
            let needle = format!("{ty}::{name}");
            let mut from = 0;
            while let Some(i) = line[from..].find(&needle) {
                let at = from + i;
                let end = at + needle.len();
                let starts_ok = at == 0 || !is_ident_byte(bytes[at - 1]);
                let ends_ok = end >= bytes.len() || !is_ident_byte(bytes[end]);
                if starts_ok && ends_ok {
                    found.push(needle.clone());
                }
                from = end;
            }
        }
    }
    found
}

/// Does the `#[cfg(test)]` at `lines[i]` open a test module (the next line that is not an attribute, a comment or empty starts a
/// `mod`)? Anything else under `#[cfg(test)]` (a helper `fn`, an `impl`, a `use`) does not end the production code.
fn opens_test_module(lines: &[&str], i: usize) -> bool {
    lines[i + 1..]
        .iter()
        .map(|l| l.trim_start())
        .find(|l| !l.is_empty() && !l.starts_with("#[") && !l.starts_with("//"))
        .is_some_and(|l| l.starts_with("mod ") || l.starts_with("pub mod ") || l.starts_with("pub(crate) mod "))
}

/// The offending lines of one source text (`name:line: text`), honouring `// libm-census: why` markers.
fn violations_in(text: &str, name: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut bad = Vec::new();
    // A `libm-census:` comment covers its own line and the next four (a reason may take a few lines).
    let mut covered = 0u32;
    let mut skip_item_depth: Option<i64> = None;
    for (n, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            if opens_test_module(&lines, n) {
                break;
            }
            // A test-only helper: its body is test code; skip it by brace counting.
            skip_item_depth = Some(0);
            continue;
        }
        if let Some(depth) = skip_item_depth.as_mut() {
            let code = line.split("//").next().unwrap_or("");
            *depth += code.matches('{').count() as i64 - code.matches('}').count() as i64;
            if *depth <= 0 && (code.contains('{') || code.trim_end().ends_with(';')) && *depth == 0 {
                skip_item_depth = None;
            }
            continue;
        }
        if line.contains("libm-census:") {
            covered = 5;
        }
        let marked = covered > 0;
        covered = covered.saturating_sub(1);
        if trimmed.starts_with("//") {
            continue;
        }
        let code = line.split("//").next().unwrap_or("");
        if !hits(code).is_empty() && !marked {
            bad.push(format!("{name}:{}: {}", n + 1, line.trim()));
        }
    }
    bad
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
            bad.extend(violations_in(
                &text,
                &file.strip_prefix(&root).unwrap().display().to_string(),
            ));
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
    assert_eq!(hits("let y = x.sin() + z.hypot(w);"), [".sin(", ".hypot("]);
    assert_eq!(hits("let y = (a.atan2(b)).ln();"), [".atan2(", ".ln("]);
    assert!(hits("self.sin(x);").is_empty());
    assert!(hits("let y = x.ln_1p_not();").is_empty());
    assert!(hits("let y = ddai_libm::sinf(x);").is_empty());
    assert!(hits("let y = ddai_libm::log(x) + ddai_libm::pow(x, y);").is_empty());
    // UFCS, and a function pointer.
    assert_eq!(hits("let y = f32::sin(x);"), ["f32::sin"]);
    assert_eq!(hits("let y = f64::atan2(a, b);"), ["f64::atan2"]);
    assert_eq!(hits("xs.iter().map(f32::cos)"), ["f32::cos"]);
    assert_eq!(hits("let (s, c) = x.sin_cos();"), [".sin_cos("]);
    // The extended list.
    for call in [
        "x.exp()",
        "x.exp2()",
        "x.log2()",
        "x.log10()",
        "x.tanh()",
        "x.cbrt()",
        "x.exp_m1()",
        "x.ln_1p()",
        "x.sinh()",
        "x.log(2.0)",
    ] {
        assert_eq!(hits(call).len(), 1, "{call}");
    }
    // A longer name that merely starts with a forbidden one, and an identifier that ends in a type name.
    assert!(hits("let y = f32::sinful(x);").is_empty());
    assert!(hits("let y = my_f32::sin(x);").is_empty());
    assert!(hits("let y = f32::MAX.min(x);").is_empty());
}

#[test]
fn only_a_test_module_ends_the_production_code() {
    // The usual layout: the test module is last.
    let usual = "fn f(x: f32) -> f32 { ddai_libm::sinf(x) }\n#[cfg(test)]\nmod tests {\n    fn g(x: f32) -> f32 { x.sin() }\n}\n";
    assert!(violations_in(usual, "usual.rs").is_empty());
    // A test-only helper does not end it: the production code after it is still checked.
    let helper = "#[cfg(test)]\nfn helper() { 1.0f32.sin(); }\nfn f(x: f32) -> f32 { x.cos() }\n";
    let v = violations_in(helper, "helper.rs");
    assert_eq!(v.len(), 1, "{v:?}");
    assert!(v[0].starts_with("helper.rs:3:"));
    // `#[cfg(test)]` on a `use` line does not end it either.
    let use_line = "#[cfg(test)]\nuse std::f32::consts::PI;\nfn f(x: f32) -> f32 { f32::sin(x) }\n";
    assert_eq!(violations_in(use_line, "use.rs").len(), 1);
    // A test module behind a second attribute and a comment still counts as the test module.
    let attrs = "fn f() {}\n#[cfg(test)]\n#[cfg(unix)]\n// why\nmod tests {\n    fn g(x: f32) -> f32 { x.sin() }\n}\n";
    assert!(violations_in(attrs, "attrs.rs").is_empty());
    // The marker covers its line and the next four.
    let marked = "// libm-census: statistics only\nlet a = x.exp();\nlet b = x.sin();\n";
    assert!(violations_in(marked, "marked.rs").is_empty());
    let late = "// libm-census: why\n\n\n\n\nlet a = x.exp();\n";
    assert_eq!(violations_in(late, "late.rs").len(), 1);
}
