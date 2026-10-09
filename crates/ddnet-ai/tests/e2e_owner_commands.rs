//! Task 4.9b e2e (D-094, amended 2026-10-05): the owner types a server command (a line starting with `/`) on the website. Against a
//! **private** DDNet 20.1 server that this test starts itself (UDP 127.0.0.1:8473, econ 127.0.0.1:8474, `sv_register 0`, its own scratch
//! directory, its own random econ password, stopped afterwards). `#[ignore]`d and guarded by `DDAI_E2E=1`:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddnet-ai --test e2e_owner_commands -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only; the rig (`owner_chat_rig`) is the one of the 4.9 test. It never touches the shared server (8303), the production units,
//! `~/aiddnet/data/bot`, the allow-list or the secrets.
//!
//! What it proves, with the server's own log and the system chat lines it sent the bot as the witnesses:
//! 1. through the real web route `/emote happy`, `/emote` and `/info` are taken (200) and the server **executed** them (it answered the
//!    bot with its own system chat lines, client id -1: the answers reach only the sender, so the rig reads them off the bot's live
//!    bridge) and **did not broadcast** them (the server's log has no chat line from the bot at all, and no player said a `/` line);
//! 2. the owner's own `/kill` is an owner line: the bot plays on, the audit counts it under `Cl_Say(owner)` and **never** under the
//!    fallback's `Cl_Say(/kill)`, and no `/kill` fallback was decided (`kill_command_ticks` is empty). (In a fresh life the server does
//!    nothing with `/kill`: `ConProtectedKill` acts only past `sv_kill_protection`.);
//! 3. `/spec` and `/pause` pause the bot on the server (the server says "speced" and, on the repeat, "resumed"). **They do not stop the
//!    bot, and that is what 20.1 does, not what the task text assumed:** to a modern client `CPlayer::Snap` keeps the player's team and
//!    only sets the `DDNetPlayer` flags (`SPEC`/`PAUSED`) and the paused character stays in the snapshot, so the bot does not see the
//!    "moved to the spectators" that D-058 stops on (that needs team -1: a moderator's `set_team`). The bot sees its own flag: its
//!    `STATUS` says `paused: true` (the site's «на паузе») for as long as the server's pause lasts, it idles (no kill, no team change,
//!    its tick keeps running, one log line per pause and per resume) and plays again when the owner types the command again;
//! 4. the bot's own log has "owner chat sent (len N)" and never a command's text.

// Task 5.5a: an ignored end-to-end test against a local DDNet server driven by python3/bash/POSIX tools: Linux only.
#![cfg(unix)]

mod owner_chat_rig;

use std::time::Duration;

use ddai_client::session::{OWNER_SAY_LABEL, SERVER_COMMAND_KILL_LABEL};
use owner_chat_rig::{Rig, RigConfig};

const GAME_PORT: u16 = 8473;
const ECON_PORT: u16 = 8474;
const BOT_NAME: &str = "E2eCmd";
/// The web's own limit is two lines in 3 s: lines go one at a time with a gap.
const GAP: Duration = Duration::from_millis(3500);

fn start() -> Rig {
    Rig::start(&RigConfig {
        game_port: GAME_PORT,
        econ_port: ECON_PORT,
        bot_name: BOT_NAME,
        seed: 94,
        sv_name: "aiddnet e2e 4.9b (private, 127.0.0.1 only)",
        scratch_tag: "ddai-e2e-ownercmd",
        tap_chat: true,
        server_cfg: "",
    })
}

/// One line through the real route; the pacing gap follows.
fn type_line(rig: &Rig, text: &str) {
    let (status, body) = rig.site.say(false, text);
    eprintln!("[web] say ({} bytes) -> {status} {body}", text.len());
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["ok"], true, "{body}");
    assert!(!body.to_string().contains(text), "the answer never repeats the line");
    std::thread::sleep(GAP);
}

#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8473/8474; DDAI_E2E=1 and --ignored"]
fn owner_commands_are_executed_not_broadcast_and_spec_pauses_the_bot() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    let mut rig = start();

    // ---- 1: commands are executed by the server and not said in the chat --------------------------------
    type_line(&rig, "/emote happy");
    type_line(&rig, "  /emote  ");
    type_line(&rig, "/info");
    // The server's answers to a command reach only the sender, as system chat lines (client id -1); the bot passes every chat line to
    // its live bridge, where the rig reads them. (The server's own log does not show them.)
    assert!(
        rig.wait_for_tapped(Duration::from_secs(10), |cid, text| cid == -1
            && text.starts_with("Emote commands are")),
        "the server never answered `/emote`: it was not run as a command: {:?}",
        rig.tapped.lock().unwrap()
    );
    assert!(
        rig.wait_for_tapped(Duration::from_secs(10), |cid, text| cid == -1
            && text.starts_with("DDraceNetwork Mod. Version")),
        "the server never answered `/info`: it was not run as a command: {:?}",
        rig.tapped.lock().unwrap()
    );
    // `/emote happy` is silent on success; its effect is not visible to a client but the line must not be chat either
    let log = rig.server_log();
    assert!(
        rig.chat().is_empty(),
        "a command is never broadcast as chat: {:?}",
        rig.chat()
    );
    assert!(
        !log.contains("/emote") && !log.contains("/info"),
        "the commands' text is nowhere in the server log"
    );
    assert!(
        !rig.tapped
            .lock()
            .unwrap()
            .iter()
            .any(|(cid, text)| *cid >= 0 && text.starts_with('/')),
        "no player (the bot) said a command as chat: {:?}",
        rig.tapped.lock().unwrap()
    );
    assert!(!rig.bot_finished(), "the server's replies did not stop the bot");

    // ---- 2: the owner's own /kill is an owner line, not the fallback ------------------------------------
    // (`/kill` is `ConProtectedKill`: it only acts on a life older than `sv_kill_protection` minutes with the race started, so on this
    // fresh life the server does nothing with it. What is checked is the bot's side: which path sent it, and how it was counted.)
    type_line(&rig, "/kill");
    // the bot plays on (it respawns): `!where` answers with a tile again
    let mut alive = false;
    for _ in 0..40 {
        if let Ok(reply) = rig.sender.send_line("!where", Duration::from_secs(2))
            && reply.text.contains("tile (")
        {
            alive = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    assert!(alive, "the bot did not come back after the owner's /kill");

    // ---- 3: /spec and /pause pause the bot (the server's flag), the bot idles; the repeat resumes it ------------------
    for command in ["/spec", "/pause"] {
        let said = |word: &'static str| move |cid: i64, text: &str| cid == -1 && text.contains(word);
        let paused_before = rig.tapped_count(said("speced"));
        let kills_before = rig.status_field("self_kills");
        type_line(&rig, command);
        assert!(
            rig.wait_until(Duration::from_secs(10), || rig.tapped_count(said("speced"))
                > paused_before),
            "{command}: the server never said \"speced\": {:?}",
            rig.tapped.lock().unwrap()
        );
        // the bot's own STATUS says so (the site's «на паузе»)
        assert!(
            rig.wait_until(Duration::from_secs(10), || rig.status_field("paused")
                == Some(serde_json::json!(true))),
            "{command}: the bot never reported paused: {:?}",
            rig.status.lock().unwrap()
        );
        // a few seconds paused: the bot is still in the game and alive, runs on (its tick moves), asks for no kill, does not stop
        let tick_at_pause = rig.status_field("tick").and_then(|t| t.as_i64()).unwrap_or(0);
        std::thread::sleep(Duration::from_secs(4));
        assert_eq!(
            rig.status_field("paused"),
            Some(serde_json::json!(true)),
            "{command}: stays paused"
        );
        assert!(
            rig.status_field("tick").and_then(|t| t.as_i64()).unwrap_or(0) > tick_at_pause + 100,
            "{command}: the bot runs on"
        );
        assert_eq!(
            rig.status_field("self_kills"),
            kills_before,
            "{command}: no kill while paused"
        );
        assert_eq!(rig.status_field("connected"), Some(serde_json::json!(true)));
        assert!(
            !rig.bot_finished(),
            "{command}: the bot stopped (D-058 is for team -1, which this is not)"
        );
        let reply = rig
            .sender
            .send_line("!where", Duration::from_secs(2))
            .expect("the bot answers")
            .text;
        assert!(reply.contains("playing"), "{command}: {reply}");
        // the repeat resumes it
        let resumed_before = rig.tapped_count(said("resumed"));
        type_line(&rig, command);
        assert!(
            rig.wait_until(Duration::from_secs(10), || rig.tapped_count(said("resumed"))
                > resumed_before),
            "{command} again: the server never said \"resumed\": {:?}",
            rig.tapped.lock().unwrap()
        );
        assert!(
            rig.wait_until(Duration::from_secs(10), || rig.status_field("paused")
                == Some(serde_json::json!(false))),
            "{command} again: the bot never reported it plays again: {:?}",
            rig.status.lock().unwrap()
        );
    }
    let report = rig.quit();
    eprintln!(
        "[bot] exit {} gave up {:?}; outgoing {:?}; owner chat {:?}",
        report.exit_code, report.gave_up, report.outgoing, report.owner_chat
    );
    assert_eq!(report.exit_code, 0, "the bot ended on its own: {:?}", report.gave_up);
    assert!(
        !report.outgoing.contains_key("Cl_Kill") && !report.outgoing.contains_key("Cl_SetTeam"),
        "a pause asks for no kill and no team change: {:?}",
        report.outgoing
    );
    assert!(report.kill_ticks.is_empty());
    let bot_log = rig.logbuf.text();
    assert_eq!(
        bot_log.matches("paused by the server: owner /pause or /spec").count(),
        2,
        "one line per pause"
    );
    assert_eq!(
        bot_log.matches("resumed: the server's pause is over").count(),
        2,
        "one line per resume"
    );

    // ---- 4: the accounting ---------------------------------------------------------------------------------
    assert_eq!(
        report.outgoing.get(OWNER_SAY_LABEL).copied(),
        Some((8, 0)),
        "eight `Cl_Say(owner)` (the commands, `/kill` among them): {:?}",
        report.outgoing
    );
    assert!(
        !report.outgoing.contains_key(SERVER_COMMAND_KILL_LABEL),
        "the owner's /kill is not the fallback's: {:?}",
        report.outgoing
    );
    assert!(report.kill_command_ticks.is_empty(), "no fallback /kill was decided");
    assert_eq!(report.owner_chat.sent, 8);
    assert_eq!(report.owner_chat.refused, 0);
    assert!(rig.chat().is_empty(), "still no chat from the bot: {:?}", rig.chat());
    let log = rig.logbuf.text();
    assert_eq!(
        log.lines().filter(|l| l.contains("owner chat sent (len ")).count(),
        8,
        "one 'owner chat sent' line each"
    );
    for text in ["/emote happy", "/info", "/kill"] {
        assert!(!log.contains(text), "the bot's log has a command's text: {text:?}");
    }
    let entries = rig.audit.0.lock().unwrap().clone();
    assert_eq!(entries.iter().filter(|e| e.cmd == "say:all").count(), 8);
    for e in &entries {
        let line = e.to_line();
        assert!(
            !line.contains("emote") && !line.contains("info") && !line.contains("kill"),
            "{line}"
        );
    }
    eprintln!("[e2e] done");
}
