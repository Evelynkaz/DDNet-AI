//! The call-site census of the owner chat (task 4.9, D-094): "only the website can make the bot speak" is a type guarantee (an
//! `OwnerText` needs the process's one `OwnerChannel`, which only the control dispatcher holds) **and** this test, which keeps the
//! places that can build or send an owner line a short, reviewed list.
//!
//! It scans every `.rs` file under `crates/` and fails when one of the patterns below appears in code (comments are skipped) outside
//! the files allowed for it. What does **not** count is test code: integration `tests/` directories, and items marked `#[cfg(test)]`
//! (a `mod`, `fn`, `use`, `impl` ... with its whole brace-matched body, or a single item ending in `;`): only those exact items are
//! skipped, never "the rest of the file"; braces are counted with strings, chars, raw strings and comments understood, and a file whose
//! braces do not balance, or a `#[cfg(test)]` in a place this scan does not understand, **fails the census** (fail closed). To add a
//! new route to the chat, add its file to the list here, in the open: the diff of this file is what a reviewer has to look at.
//!
//! The identifier `OwnerChannel` itself (a `use`, an alias, a rename) is allowed only in `control.rs` and `owner_chat.rs`, and the
//! protocol-side `ControlCommand::Say {` / `SayText::new(` only where the web, the protocol and the dispatcher meet. `OwnerChannel::
//! mint_for_tests` (the `test-util` feature of `ddai-net`) is held to the same rule, and the feature may only be switched on from a
//! `[dev-dependencies]` table.

use std::path::{Path, PathBuf};

const CONTROL: &str = "crates/ddai-bot/src/control.rs";
const OWNER_CHAT: &str = "crates/ddai-net/src/owner_chat.rs";
const SAY_ROUTE: &str = "crates/ddai-web/src/http/say.rs";

/// (pattern, the production files allowed to contain it).
const RULES: &[(&str, &[&str])] = &[
    // The capability itself, under any name: only the module that defines it and the dispatcher that holds it.
    ("OwnerChannel", &[OWNER_CHAT, CONTROL]),
    // Making the text: only the control dispatcher, which holds the channel.
    ("OwnerText::new(", &[CONTROL]),
    ("mint_for_tests(", &[OWNER_CHAT]),
    // Renaming or aliasing the types would hide the patterns below: not at all in production code.
    ("OwnerText as ", &[]),
    ("OwnerSay as ", &[]),
    ("= OwnerText", &[OWNER_CHAT]),
    ("= OwnerSay", &[OWNER_CHAT]),
    // Making an `OwnerSay` out of a typed text, and handing it to the client: only the runner.
    ("OwnerSay::new(", &["crates/ddai-bot/src/runner.rs"]),
    ("OwnerSay {", &[OWNER_CHAT]),
    (".owner_say(", &["crates/ddai-bot/src/runner.rs"]),
    // The session's sending path, and the driver that calls it.
    (
        "request_owner_say(",
        &["crates/ddai-client/src/session.rs", "crates/ddai-client/src/driver.rs"],
    ),
    ("owner_auth.grant(", &["crates/ddai-client/src/session.rs"]),
    // The command that carries a line inside the bot.
    (
        "BotCommand::Say {",
        &[
            CONTROL,
            "crates/ddai-bot/src/runner.rs",
            "crates/ddai-bot/src/bot/apply.rs",
        ],
    ),
    // The typed in-process route to the control socket: made by the web's route, read by the protocol and the dispatcher.
    (
        "ControlCommand::Say {",
        &[
            SAY_ROUTE,
            "crates/ddai-web/src/http/bot.rs", // refuses a `say` on the generic command route
            CONTROL,
            "crates/ddai-botctl/src/proto.rs",
        ],
    ),
    ("SayText::new(", &[SAY_ROUTE]),
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if name != "target" && name != ".git" {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Why a file could not be scanned with confidence: the census then fails (fail closed).
type ScanError = String;

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Index just past the end of the item that starts at `from` (after its attributes): the matching `}` of its first brace block, or the
/// first `;` met before any brace. Understands line and block comments, strings, raw strings and char literals.
fn item_end(chars: &[char], from: usize) -> Result<usize, ScanError> {
    let (mut i, mut depth, mut opened) = (from, 0usize, false);
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            '/' if next == Some('/') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            '/' if next == Some('*') => {
                let mut nest = 1;
                i += 2;
                while i < chars.len() && nest > 0 {
                    if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                        nest += 1;
                        i += 1;
                    } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                        nest -= 1;
                        i += 1;
                    }
                    i += 1;
                }
                if nest > 0 {
                    return Err("an unterminated block comment".into());
                }
                continue;
            }
            'r' if (i == 0 || !is_ident(chars[i - 1])) && matches!(next, Some('"' | '#')) => {
                let mut j = i + 1;
                let mut hashes = 0;
                while chars.get(j) == Some(&'#') {
                    hashes += 1;
                    j += 1;
                }
                if chars.get(j) == Some(&'"') {
                    j += 1;
                    loop {
                        match chars.get(j) {
                            None => return Err("an unterminated raw string".into()),
                            Some('"') if (0..hashes).all(|k| chars.get(j + 1 + k) == Some(&'#')) => {
                                j += 1 + hashes;
                                break;
                            }
                            _ => j += 1,
                        }
                    }
                    i = j;
                    continue;
                }
            }
            '"' => {
                i += 1;
                loop {
                    match chars.get(i) {
                        None => return Err("an unterminated string".into()),
                        Some('\\') => i += 2,
                        Some('"') => {
                            i += 1;
                            break;
                        }
                        _ => i += 1,
                    }
                }
                continue;
            }
            '\'' => {
                // a char literal ('x', '\n', '\'') or a lifetime ('a): a literal closes within a few chars
                if next == Some('\\') {
                    let mut j = i + 2;
                    while j < chars.len() && chars[j] != '\'' && j < i + 12 {
                        j += 1;
                    }
                    i = j + 1;
                    continue;
                }
                if chars.get(i + 2) == Some(&'\'') {
                    i += 3;
                    continue;
                }
            }
            '{' => {
                depth += 1;
                opened = true;
            }
            '}' => {
                if depth == 0 {
                    return Err("a closing brace with no opening one".into());
                }
                depth -= 1;
                if opened && depth == 0 {
                    return Ok(i + 1);
                }
            }
            ';' if !opened && depth == 0 => return Ok(i + 1),
            _ => {}
        }
        i += 1;
    }
    Err("the braces do not balance (end of file inside an item)".into())
}

/// `text` with every `#[cfg(test)]` item blanked out (newlines kept, so line numbers stay), or why that could not be done safely.
fn without_cfg_test_items(text: &str) -> Result<String, ScanError> {
    const ITEM_KEYWORDS: &[&str] = &[
        "mod", "fn", "use", "impl", "struct", "enum", "const", "static", "trait", "type", "pub", "async", "unsafe",
        "extern",
    ];
    let chars: Vec<char> = text.chars().collect();
    let mut out = chars.clone();
    let mut starts = vec![0usize];
    starts.extend(
        chars
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == '\n')
            .map(|(i, _)| i + 1),
    );
    let mut skip_until = 0;
    for &ls in &starts {
        if ls < skip_until || ls >= chars.len() {
            continue;
        }
        let line_end = chars[ls..]
            .iter()
            .position(|&c| c == '\n')
            .map_or(chars.len(), |n| ls + n);
        let line: String = chars[ls..line_end].iter().collect();
        if line.trim() != "#[cfg(test)]" {
            continue;
        }
        // skip further attribute lines (`#[allow(..)]` ...), then find the item's keyword
        let mut p = line_end;
        loop {
            while p < chars.len() && chars[p].is_whitespace() {
                p += 1;
            }
            if chars.get(p) == Some(&'/') && chars.get(p + 1) == Some(&'/') {
                while p < chars.len() && chars[p] != '\n' {
                    p += 1;
                }
                continue;
            }
            if chars.get(p) == Some(&'#') && chars.get(p + 1) == Some(&'[') {
                let mut depth = 0;
                while p < chars.len() {
                    match chars[p] {
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                p += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    p += 1;
                }
            } else {
                break;
            }
        }
        let word: String = chars[p..].iter().take_while(|c| is_ident(**c)).collect();
        if !ITEM_KEYWORDS.contains(&word.as_str()) {
            // A `#[cfg(test)]` on a field, a statement or an expression: nothing is blanked, the code is scanned as production code
            // (the safe side: more is scanned, never less).
            continue;
        }
        let end = item_end(&chars, p)?;
        for c in out.iter_mut().take(end).skip(ls) {
            if *c != '\n' {
                *c = ' ';
            }
        }
        skip_until = end;
    }
    Ok(out.into_iter().collect())
}

/// The part of a source file that is compiled into the program, comments aside: files under `tests/` or `benches/` are test code
/// altogether, and `#[cfg(test)]` items are blanked.
fn production_part(rel: &str, text: &str) -> Result<String, ScanError> {
    if rel.contains("/tests/") || rel.contains("/benches/") {
        return Ok(String::new());
    }
    without_cfg_test_items(text)
}

/// Every `(file, pattern)` pair that appears in production code outside its allowed files, and every file that could not be scanned.
/// `files` are `(path relative to the repo root with `/`, contents)`.
fn violations(files: &[(String, String)]) -> Vec<String> {
    let mut found = Vec::new();
    for (rel, text) in files {
        let prod = match production_part(rel, text) {
            Ok(p) => p,
            Err(why) => {
                found.push(format!("{rel}: cannot be scanned ({why}): the census fails closed"));
                continue;
            }
        };
        for (n, line) in prod.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            for (pattern, allowed) in RULES {
                if line.contains(pattern) && !allowed.contains(&rel.as_str()) {
                    found.push(format!("{rel}:{}: `{pattern}`", n + 1));
                }
            }
        }
    }
    found
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/ddai-bot is two levels under the repo root")
        .to_path_buf()
}

fn load_sources() -> Vec<(String, String)> {
    let root = repo_root();
    let mut paths = Vec::new();
    rust_files(&root.join("crates"), &mut paths);
    paths
        .into_iter()
        .filter_map(|p| {
            let rel = p.strip_prefix(&root).ok()?.to_string_lossy().replace('\\', "/");
            Some((rel, std::fs::read_to_string(&p).ok()?))
        })
        .collect()
}

#[test]
fn only_the_reviewed_files_can_build_or_send_an_owner_line() {
    let files = load_sources();
    assert!(
        files.len() > 100,
        "the census found only {} source files: is the scan broken?",
        files.len()
    );
    let found = violations(&files);
    assert!(
        found.is_empty(),
        "owner-chat call sites outside the reviewed files (add the file to RULES in this test, in review, or remove the call):\n{}",
        found.join("\n")
    );
}

/// The census is not vacuous: every pattern is really present in the production code of the files it allows, so a rename that
/// slips past a pattern is noticed.
#[test]
fn every_pattern_is_present_where_it_is_allowed() {
    let files = load_sources();
    for (pattern, allowed) in RULES {
        if allowed.is_empty() {
            continue; // forbidden everywhere: nothing has to be present
        }
        let present = files.iter().any(|(rel, text)| {
            allowed.contains(&rel.as_str())
                && production_part(rel, text)
                    .unwrap_or_default()
                    .lines()
                    .any(|l| !l.trim_start().starts_with("//") && l.contains(pattern))
        });
        assert!(
            present,
            "`{pattern}` is nowhere in production code of {allowed:?}: renamed? update the census"
        );
    }
}

#[test]
fn the_census_catches_a_new_route_to_the_chat() {
    let src = |rel: &str, body: &str| (rel.to_string(), body.to_string());
    // an echo of the game chat written into the runner's `SvChat` arm, and one in a new module
    let echo = "    client.owner_say(OwnerSay::new(false, OwnerText::new(&channel, &c.message)?));\n";
    let files = vec![
        src("crates/ddai-bot/src/runner.rs", "fn ok() { client.owner_say(say); }\n"),
        src("crates/ddai-bot/src/echo.rs", echo),
        src(
            "crates/ddai-bot/src/other.rs",
            "let c = BotCommand::Say { team: false, text };\n",
        ),
        src("crates/ddai-bot/src/third.rs", "let ch = OwnerChannel::claim();\n"),
        src("crates/ddai-client/src/x.rs", "session.request_owner_say(&say, now);\n"),
        src(
            "crates/ddai-bot/src/leak.rs",
            "let ch = OwnerChannel::mint_for_tests();\n",
        ),
        src(
            "crates/ddai-bot/src/web.rs",
            "let c = ControlCommand::Say { team: false, text: SayText::new(s) };\n",
        ),
    ];
    let found = violations(&files);
    for needle in [
        "echo.rs:1: `.owner_say(`",
        "echo.rs:1: `OwnerSay::new(`",
        "echo.rs:1: `OwnerText::new(`",
        "other.rs:1: `BotCommand::Say {`",
        "third.rs:1: `OwnerChannel`",
        "x.rs:1: `request_owner_say(`",
        "leak.rs:1: `mint_for_tests(`",
        "web.rs:1: `ControlCommand::Say {`",
        "web.rs:1: `SayText::new(`",
    ] {
        assert!(
            found.iter().any(|f| f.ends_with(needle)),
            "{needle} not caught: {found:?}"
        );
    }
    assert!(
        !found.iter().any(|f| f.contains("runner.rs")),
        "the runner is allowed: {found:?}"
    );
    // comments, `#[cfg(test)]` items and integration tests do not count
    let quiet = vec![
        src(
            "crates/ddai-bot/src/doc.rs",
            "// client.owner_say(x) is how the runner does it\n/// `OwnerText::new(` needs a channel\n",
        ),
        src(
            "crates/ddai-bot/src/t.rs",
            "fn p() {}\n#[cfg(test)]\nmod tests { fn t() { OwnerText::new(&c, \"x\"); } }\n",
        ),
        src(
            "crates/ddai-bot/tests/some.rs",
            "fn t() { let c = OwnerChannel::mint_for_tests(); }\n",
        ),
        src(
            "crates/ddai-bot/src/u.rs",
            "#[cfg(test)]\nuse x::OwnerChannel;\n#[cfg(test)]\n#[allow(dead_code)]\nfn helper() { OwnerChannel::claim(); }\n",
        ),
    ];
    assert_eq!(violations(&quiet), Vec::<String>::new());
}

/// F5: an early `#[cfg(test)]` item hides only itself, not the rest of the file (the old census skipped everything after the first one).
#[test]
fn an_early_cfg_test_item_hides_only_itself() {
    let src = |rel: &str, body: &str| (rel.to_string(), body.to_string());
    let early_mod = "#[cfg(test)]\nmod reconnect_tests {\n    fn t() { let s = \"}\"; let c = '{'; }\n}\n\nfn client_loop() {\n    client.owner_say(x);\n}\n";
    let early_helper = "#[cfg(test)]\nfn helper() {}\nfn prod() { session.request_owner_say(&s, now); }\n";
    let early_use = "#[cfg(test)]\nuse std::fmt;\nfn prod() { OwnerText::new(&c, s); }\n";
    let found = violations(&[
        src("crates/ddai-client/src/early_mod.rs", early_mod),
        src("crates/ddai-client/src/early_helper.rs", early_helper),
        src("crates/ddai-client/src/early_use.rs", early_use),
    ]);
    for needle in [
        "early_mod.rs:7: `.owner_say(`",
        "early_helper.rs:3: `request_owner_say(`",
        "early_use.rs:3: `OwnerText::new(`",
    ] {
        assert!(
            found.iter().any(|f| f.ends_with(needle)),
            "{needle} not caught: {found:?}"
        );
    }
    assert_eq!(found.len(), 3, "{found:?}");
}

/// F5: a renamed import or an alias does not get past the census.
#[test]
fn a_renamed_import_or_an_alias_is_caught() {
    let src = |rel: &str, body: &str| (rel.to_string(), body.to_string());
    let found = violations(&[
        src(
            "crates/ddai-bot/src/runner.rs",
            "use ddai_net::owner_chat::{OwnerChannel as Ch, OwnerText as T};\nlet c = Ch::claim();\n",
        ),
        src("crates/ddai-bot/src/alias.rs", "type Say = OwnerSay;\n"),
        src("crates/ddai-bot/src/fnptr.rs", "let f = OwnerChannel::claim;\n"),
    ]);
    for needle in [
        "runner.rs:1: `OwnerChannel`",
        "runner.rs:1: `OwnerText as `",
        "alias.rs:1: `= OwnerSay`",
        "fnptr.rs:1: `OwnerChannel`",
    ] {
        assert!(
            found.iter().any(|f| f.ends_with(needle)),
            "{needle} not caught: {found:?}"
        );
    }
}

/// A `#[cfg(test)]` on a field or a statement hides nothing: what follows is scanned as production code.
#[test]
fn a_cfg_test_on_a_field_or_statement_hides_nothing() {
    let body = "struct S {\n    #[cfg(test)]\n    a: u32,\n    b: OwnerChannel,\n}\n#[cfg(test)]\n// a comment between\nfn helper() { let ch = OwnerChannel::claim(); }\n";
    let found = violations(&[("crates/ddai-bot/src/field.rs".to_string(), body.to_string())]);
    assert!(
        found.iter().any(|f| f.ends_with("field.rs:4: `OwnerChannel`")),
        "{found:?}"
    );
    assert_eq!(
        found.len(),
        1,
        "the helper behind a comment is a skipped item: {found:?}"
    );
}

/// F5: fail closed: unbalanced braces, an unterminated string.
#[test]
fn a_file_the_scan_cannot_follow_fails_the_census() {
    let src = |rel: &str, body: &str| (rel.to_string(), body.to_string());
    for (name, body) in [
        ("unbalanced.rs", "#[cfg(test)]\nmod t {\n fn a() {\n}\nfn prod() {}\n"),
        ("string.rs", "#[cfg(test)]\nmod t { let s = \"open; }\n"),
        ("closing.rs", "#[cfg(test)]\nfn x() }\n"),
    ] {
        let found = violations(&[src(&format!("crates/ddai-bot/src/{name}"), body)]);
        assert!(
            found
                .iter()
                .any(|f| f.contains(name) && f.contains("cannot be scanned")),
            "{name} should fail the census: {found:?}"
        );
    }
}

/// The `test-util` feature of `ddai-net` (which unlocks `mint_for_tests`) may be switched on only from `[dev-dependencies]`.
#[test]
fn the_test_util_feature_of_ddai_net_is_only_enabled_by_dev_dependencies() {
    let root = repo_root();
    let mut manifests = Vec::new();
    for entry in std::fs::read_dir(root.join("crates")).unwrap().flatten() {
        let m = entry.path().join("Cargo.toml");
        if m.is_file() {
            manifests.push(m);
        }
    }
    assert!(manifests.len() > 10);
    let mut dev_uses = 0;
    for m in manifests {
        let text = std::fs::read_to_string(&m).unwrap();
        let mut section = String::new();
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                section = t.to_string();
                continue;
            }
            let mentions_feature = t.contains("test-util")
                && (t.starts_with("ddai-net") || t.contains("ddai-net/") || section.contains("ddai-net"));
            if mentions_feature {
                if section == "[dev-dependencies]" || section == "[dev-dependencies.ddai-net]" {
                    dev_uses += 1;
                } else {
                    panic!(
                        "{}: `ddai-net` with `test-util` in {section}: only [dev-dependencies] may enable it",
                        m.display()
                    );
                }
            }
        }
    }
    assert!(
        dev_uses >= 2,
        "the tests of ddai-client and ddai-bot mint their own channels"
    );
}
