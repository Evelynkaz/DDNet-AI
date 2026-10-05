//! Task 4.9 e2e (D-094): the owner types on the website and the bot says it in the game chat, against a **private** DDNet 20.1 server
//! that this test starts itself (UDP 127.0.0.1:8413, econ 127.0.0.1:8414, `sv_register 0`, its own scratch directory, its own random
//! econ password, stopped afterwards). `#[ignore]`d and guarded by `DDAI_E2E=1`:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddnet-ai --test e2e_owner_chat -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only. It never touches the shared server (8303), the production units, `~/aiddnet/data/bot`, the allow-list or the secrets:
//! the bot, the control socket, the web unit (an ephemeral port, its own password and data directory) and the map cache all live in the
//! scratch directory (the server binary and the map are only read). The rig is shared with the 4.9b test (`owner_chat_rig`).
//!
//! What it proves, with the server's own log as the witness of what the game server received:
//! 1. through the real web route (session, CSRF, Origin) two chat lines are said in the game chat, the second one 3 s or more behind the
//!    first, in order, with exactly their text; a team line is said as team chat;
//! 2. a third line right behind the first two is refused by the web's own rate limit (429) and is never said;
//! 3. through the control socket (bypassing the web limit) the bot's own limits hold: one line goes at once, a queue of three waits for
//!    its turns, the fifth is refused with the reason `queue_full`; the four that were taken are said, in order, 3 s apart;
//! 4. a line with an invisible character hiding a `/kill` and an empty line are refused (400) and never said; the server's log has no
//!    other chat from the bot, and no `/kill` (task 4.9b: a plain `/`-command is taken now, see `e2e_owner_commands.rs`);
//! 5. the bot's outgoing audit counts `Cl_Say(owner)` apart (accepted, none refused) and no other chat label; its log has "owner chat
//!    sent (len N)" lines and never the text.

mod owner_chat_rig;

use std::time::Duration;

use ddai_client::session::OWNER_SAY_LABEL;
use owner_chat_rig::{Rig, RigConfig};

const GAME_PORT: u16 = 8413;
const ECON_PORT: u16 = 8414;
const BOT_NAME: &str = "E2eSay";

#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8413/8414; DDAI_E2E=1 and --ignored"]
fn the_owner_types_on_the_website_and_the_bot_says_it_in_the_game_chat() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    let mut rig = Rig::start(&RigConfig {
        game_port: GAME_PORT,
        econ_port: ECON_PORT,
        bot_name: BOT_NAME,
        seed: 49,
        sv_name: "aiddnet e2e 4.9 (private, 127.0.0.1 only)",
        scratch_tag: "ddai-e2e-ownerchat",
        tap_chat: false,
        server_cfg: "",
    });

    // ---- 1 + 2: two lines through the web route, a third refused by the web's own limit ----------------
    let (s1, b1) = rig.site.say(false, "  hello from the owner  ");
    eprintln!("[web] say 1 -> {s1} {b1}");
    assert_eq!(s1, 200, "{b1}");
    let (s2, b2) = rig.site.say(false, "second line, привет");
    eprintln!("[web] say 2 -> {s2} {b2}");
    assert_eq!(s2, 200, "{b2}");
    assert!(
        b2["text"].as_str().unwrap_or("").contains("will be said in about"),
        "the second line is told it waits for its turn: {b2}"
    );
    let (s3, b3) = rig.site.say(false, "third line right away");
    eprintln!("[web] say 3 -> {s3} {b3}");
    assert_eq!(
        (s3, b3["error"].as_str()),
        (429, Some("rate_limited")),
        "the web's own rate limit"
    );
    // refused here, never said
    let (sc, bc) = rig.site.say(false, "\u{200B}/kill");
    assert_eq!(
        (sc, bc["error"].as_str(), bc["detail"].as_str()),
        (400, Some("invalid_text"), Some("control")),
        "an invisible character in front of a command is refused by the web"
    );
    let (se, _) = rig.site.say(false, "   ");
    assert_eq!(se, 400);

    let lines = rig.wait_for_chat(2, Duration::from_secs(15));
    eprintln!("[server] chat: {lines:?}");
    assert_eq!(lines.len(), 2, "the server received exactly two lines: {lines:?}");
    assert_eq!(
        (lines[0].category.as_str(), lines[0].text.as_str()),
        ("chat", "hello from the owner"),
        "trimmed, in all chat"
    );
    assert_eq!(
        (lines[1].category.as_str(), lines[1].text.as_str()),
        ("chat", "second line, привет")
    );
    assert!(
        lines[1].at >= lines[0].at + 2,
        "the second line came 3 s behind the first (log seconds {} and {})",
        lines[0].at,
        lines[1].at
    );
    // nothing else follows (the third line was never accepted)
    std::thread::sleep(Duration::from_secs(4));
    let lines = rig.chat();
    assert_eq!(lines.len(), 2, "still exactly two lines after waiting: {lines:?}");

    // ---- 3: the bot's own limits through the control socket (the web's limit is not in the way) ---------
    std::thread::sleep(Duration::from_secs(2)); // the pacing gap since the last line is over
    // The first goes out at once; the next three wait for their turn (the queue holds three); the fifth has no room.
    let r_a = rig.control_say(false, "queue one");
    let r_b = rig.control_say(true, "queue two (team)");
    let r_c = rig.control_say(false, "queue three");
    let r_d = rig.control_say(false, "queue four");
    let r_e = rig.control_say(false, "queue five");
    eprintln!("[bot] {r_a}\n[bot] {r_b}\n[bot] {r_c}\n[bot] {r_d}\n[bot] {r_e}");
    assert!(
        [&r_a, &r_b, &r_c, &r_d].iter().all(|r| r["ok"] == true),
        "four lines are taken (one said now, three waiting): {r_a} {r_b} {r_c} {r_d}"
    );
    assert_eq!(r_e["ok"], false, "the fifth is refused: {r_e}");
    assert_eq!(r_e["data"]["reason"], "queue_full");
    assert!(r_e["text"].as_str().unwrap().contains("already waiting"), "{r_e}");
    assert!(
        !r_e.to_string().contains("queue five"),
        "the refusal never repeats the line"
    );
    let lines = rig.wait_for_chat(6, Duration::from_secs(30));
    eprintln!("[server] chat: {lines:?}");
    assert_eq!(lines.len(), 6, "exactly six lines in all: {lines:?}");
    let got: Vec<(&str, &str)> = lines[2..]
        .iter()
        .map(|l| (l.category.as_str(), l.text.as_str()))
        .collect();
    assert_eq!(
        got,
        [
            ("chat", "queue one"),
            ("teamchat", "queue two (team)"),
            ("chat", "queue three"),
            ("chat", "queue four")
        ],
        "in order, the team line as team chat"
    );
    for w in lines[1..].windows(2) {
        assert!(w[1].at >= w[0].at + 2, "paced 3 s apart: {lines:?}");
    }
    std::thread::sleep(Duration::from_secs(4));
    let lines = rig.chat();
    assert_eq!(lines.len(), 6, "and nothing more (the fifth was refused): {lines:?}");
    let log = rig.server_log();
    assert!(
        !log.contains("queue five") && !log.contains("third line right away") && !log.contains("/kill"),
        "refused lines never reached the server"
    );

    // ---- stop, then the audit -----------------------------------------------------------------------
    let report = rig.quit();
    eprintln!(
        "[audit] outgoing {:?}; owner chat {:?}",
        report.outgoing, report.owner_chat
    );
    assert_eq!(report.exit_code, 0, "gave up: {:?}", report.gave_up);
    for (label, (accepted, refused)) in &report.outgoing {
        assert_eq!(*refused, 0, "the allow-list refused a {label}");
        assert!(*accepted > 0);
    }
    assert_eq!(
        report.outgoing.get(OWNER_SAY_LABEL).copied(),
        Some((6, 0)),
        "six `Cl_Say(owner)`, counted apart: {:?}",
        report.outgoing
    );
    let chat_labels: Vec<_> = report
        .outgoing
        .keys()
        .filter(|k| k.contains("Say") || k.contains("Chat"))
        .collect();
    assert_eq!(
        chat_labels,
        [&OWNER_SAY_LABEL.to_string()],
        "no other chat label (no /kill): {chat_labels:?}"
    );
    assert_eq!(report.owner_chat.sent, 6);
    assert_eq!(report.owner_chat.accepted, 6);
    assert_eq!(report.owner_chat.refused, 1, "the queue_full one");
    assert_eq!(report.owner_chat.dropped, 0);
    assert!(report.kill_command_ticks.is_empty());

    // the bot's log: one "owner chat sent (len N)" per line, and never a line's text
    let log = rig.logbuf.text();
    let sent_logs = log.lines().filter(|l| l.contains("owner chat sent (len ")).count();
    assert_eq!(sent_logs, 6, "six 'owner chat sent' lines in the bot's log");
    for secret in [
        "hello from the owner",
        "second line",
        "привет",
        "queue one",
        "queue two",
        "queue three",
        "queue four",
        "queue five",
        "third line",
    ] {
        assert!(!log.contains(secret), "the bot's log has a line's text: {secret:?}");
    }
    assert!(
        log.contains("owner chat sent (len 20)"),
        "the first line is 20 bytes: {log}"
    );
    // the control audit: tags only
    let entries = rig.audit.0.lock().unwrap().clone();
    assert_eq!(
        entries.iter().filter(|e| e.cmd == "say:all").count(),
        2 + 4,
        "two from the web, and queue one, three, four and five from the socket"
    );
    assert_eq!(entries.iter().filter(|e| e.cmd == "say:team").count(), 1);
    for e in &entries {
        let line = e.to_line();
        assert!(!line.contains("queue") && !line.contains("hello"), "{line}");
    }
    eprintln!("[e2e] done");
}
