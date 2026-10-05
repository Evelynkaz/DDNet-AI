//! Task 4.9b review F1 e2e: `sv_pauseable 1` makes `/spec` mean `PAUSE_SPEC`, and 20.1 then removes a **still, grounded** tee from the
//! world in the very snapshot that sets the `SPEC` flag (`player.cpp` `ProcessPause`/`CanSpec`, `CCharacter::Pause(true)`). The bot must
//! still see the pause: against a **private** server (UDP 127.0.0.1:8473, econ 8474, `sv_register 0`, own scratch directory and econ
//! password, stopped afterwards), the bot at rest in `!mode hold` gets `/spec` from the website and
//! - the bot's `STATUS` says `paused: true` although its tee is gone (the site's «на паузе»), and its `connected` stays true;
//! - it counts no death, asks for no `Cl_Kill`, and the operator's `!kill` is answered truthfully ("paused by the server") and sends none;
//! - the repeat of `/spec` resumes it (`paused: false`), the process never stops, and the log has one "paused" and one "resumed" line.
//!
//! The rig is shared with `e2e_owner_commands.rs`; this is its own test binary because the bot's `OwnerChannel` is once per process.
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddnet-ai --test e2e_owner_pause_still -- --ignored --nocapture --test-threads=1
//! ```

mod owner_chat_rig;

use std::time::Duration;

use owner_chat_rig::{Rig, RigConfig};

#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8473/8474; DDAI_E2E=1 and --ignored"]
fn spec_on_a_still_bot_with_sv_pauseable_pauses_it_though_the_tee_is_gone() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    let mut rig = Rig::start(&RigConfig {
        game_port: 8473,
        econ_port: 8474,
        bot_name: "E2eStill",
        seed: 96,
        sv_name: "aiddnet e2e 4.9b still (private, 127.0.0.1 only)",
        scratch_tag: "ddai-e2e-ownerstill",
        tap_chat: true,
        server_cfg: "sv_pauseable 1",
    });
    // at rest: hold mode does nothing, the tee lands and stays still
    let reply = rig
        .sender
        .send_line("!mode hold", Duration::from_secs(2))
        .expect("the bot answers");
    assert!(reply.ok, "{}", reply.text);
    std::thread::sleep(Duration::from_secs(5));
    assert_eq!(rig.status_field("paused"), Some(serde_json::json!(false)));
    let deaths = rig.status_field("deaths");
    let kills = rig.status_field("self_kills");

    let (status, body) = rig.site.say(false, "/spec");
    assert_eq!(status, 200, "{body}");
    assert!(
        rig.wait_for_tapped(Duration::from_secs(10), |cid, text| cid == -1
            && text.contains("speced")),
        "the server never said \"speced\": {:?}",
        rig.tapped.lock().unwrap()
    );
    assert!(
        rig.wait_until(Duration::from_secs(10), || rig.status_field("paused")
            == Some(serde_json::json!(true))),
        "the bot never reported the pause: {:?}",
        rig.status.lock().unwrap()
    );
    std::thread::sleep(Duration::from_secs(3));
    assert_eq!(rig.status_field("paused"), Some(serde_json::json!(true)));
    assert_eq!(rig.status_field("connected"), Some(serde_json::json!(true)));
    assert_eq!(rig.status_field("deaths"), deaths, "a paused tee is not a death");
    // the operator's `!kill` is refused truthfully and sends nothing
    let reply = rig
        .sender
        .send_line("!kill", Duration::from_secs(2))
        .expect("the bot answers");
    assert!(!reply.ok && reply.text.contains("paused by the server"), "{reply:?}");
    std::thread::sleep(Duration::from_secs(2));
    assert_eq!(rig.status_field("self_kills"), kills, "no kill while paused");
    assert_eq!(rig.status_field("deaths"), deaths);
    assert!(!rig.bot_finished());

    // the repeat resumes it
    std::thread::sleep(Duration::from_millis(3500));
    let (status, body) = rig.site.say(false, "/spec");
    assert_eq!(status, 200, "{body}");
    assert!(
        rig.wait_for_tapped(Duration::from_secs(10), |cid, text| cid == -1
            && text.contains("resumed")),
        "the server never said \"resumed\": {:?}",
        rig.tapped.lock().unwrap()
    );
    assert!(
        rig.wait_until(Duration::from_secs(10), || rig.status_field("paused")
            == Some(serde_json::json!(false))),
        "the bot never reported that it plays again: {:?}",
        rig.status.lock().unwrap()
    );

    let report = rig.quit();
    eprintln!(
        "[bot] exit {} outgoing {:?} kill_ticks {:?}",
        report.exit_code, report.outgoing, report.kill_ticks
    );
    assert_eq!(report.exit_code, 0, "{:?}", report.gave_up);
    assert!(!report.outgoing.contains_key("Cl_Kill"), "{:?}", report.outgoing);
    assert!(report.kill_ticks.is_empty());
    assert_eq!(report.outgoing.get("Cl_Say(owner)").copied(), Some((2, 0)));
    let log = rig.logbuf.text();
    assert_eq!(log.matches("paused by the server: owner /pause or /spec").count(), 1);
    assert_eq!(log.matches("resumed: the server's pause is over").count(), 1);
}
