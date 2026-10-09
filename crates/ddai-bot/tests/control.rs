//! The web control channel against a real `Bot` (task 5.6): a `ControlServer` on a real Unix socket, a bot thread that
//! drains the command bus between snapshots exactly like the runner does and keeps deciding, and a "web" that writes
//! the lists file and sends typed requests. No network, no game server. Every nickname is a `p<id>` test string.

// Task 5.5a: talks to the bot over a Unix-domain socket, which Windows has no counterpart of yet (ddai_os::ipc, task 5.5b).
#![cfg(unix)]

mod support;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ddai_bot::control::{AuditSink, ControlServer, MemoryAudit};
use ddai_bot::relations::ListKind;
use ddai_bot::{Bot, BotConfig, BrainKind, Relations};
use ddai_botctl::proto::{ControlCommand, ControlReply, ControlRequest, ModeArg, ReplyCode};
use support::*;

fn ask(stream: &mut UnixStream, cmd: ControlCommand) -> ControlReply {
    let mut line = serde_json::to_vec(&ControlRequest::new("00ff00ff00ff00ff", cmd)).unwrap();
    line.push(b'\n');
    stream.write_all(&line).unwrap();
    let mut reply = String::new();
    BufReader::new(stream.try_clone().unwrap())
        .read_line(&mut reply)
        .unwrap();
    serde_json::from_str(reply.trim_end()).unwrap()
}

#[test]
fn the_web_changes_the_running_bot_through_the_socket() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("bot").join("control.sock");
    let lists = dir.path().join("relations.json");
    let (sender, inbox) = ddai_bot::command::CommandBus::open();
    let audit = Arc::new(MemoryAudit::default());
    let server = ControlServer::start(&sock, sender, Arc::clone(&audit) as Arc<dyn AuditSink>).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let target = Arc::new(AtomicI32::new(-2));
    let mode = Arc::new(Mutex::new(String::new()));
    let (stop2, target2, mode2, lists2) = (stop.clone(), target.clone(), mode.clone(), lists.clone());
    let bot_thread = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            let mut cfg: BotConfig = cfg(BrainKind::Idle);
            cfg.relations_path = Some(lists2);
            cfg.clips.dir = Some(std::env::temp_dir().join("ddai-botctl-test-no-clips"));
            cfg.clips.autoclip = false;
            let map = room(&[]);
            let mut bot: Bot = bot_with(Box::new(ddai_brain::IdleBrain), cfg, Relations::new());
            bot.on_map_loaded(Arc::clone(&map));
            let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)]);
            while !stop2.load(Ordering::SeqCst) {
                run(&mut bot, &mut sc, 1);
                while let Some(req) = inbox.try_next() {
                    let reply = bot.command(req.cmd);
                    let _ = req.reply.send(reply);
                }
                target2.store(bot.target_id(), Ordering::SeqCst);
                *mode2.lock().unwrap() = bot.mode().name().to_string();
                std::thread::sleep(Duration::from_millis(5));
            }
            bot.relations().contains(ListKind::Friend, "p1")
        })
        .unwrap();

    let wait = |what: &str, f: &dyn Fn() -> bool| {
        let t = Instant::now();
        while !f() {
            assert!(t.elapsed() < Duration::from_secs(5), "timed out: {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    };
    wait("the bot picks a target", &|| target.load(Ordering::SeqCst) >= 0);
    let first = target.load(Ordering::SeqCst);

    let mut web = UnixStream::connect(&sock).unwrap();
    web.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    // A mode change reaches the bot and takes effect.
    let r = ask(&mut web, ControlCommand::Mode { mode: ModeArg::Hold });
    assert!(r.ok && r.text == "mode: hold", "{r:?}");
    wait("hold", &|| *mode.lock().unwrap() == "hold");
    let r = ask(&mut web, ControlCommand::Go {});
    assert!(r.ok);
    wait("fight", &|| *mode.lock().unwrap() == "fight");

    // The editor writes the lists file, then asks for a reload: the new friend is spared by the running bot.
    let mut edited = Relations::new();
    edited.add(ListKind::Friend, &format!("P{first}"));
    edited.save(&lists).unwrap();
    let r = ask(&mut web, ControlCommand::ReloadRelations {});
    assert!(r.ok, "{r:?}");
    assert!(r.text.contains("friend 1"), "{}", r.text);
    wait("the friend is no longer the target", &|| {
        target.load(Ordering::SeqCst) != first
    });

    // The bot's own refusals come back as replies, not as transport errors.
    let r = ask(
        &mut web,
        ControlCommand::Wb {
            mode: ddai_botctl::proto::WbArg::Off,
        },
    );
    assert!(!r.ok && r.code.is_none() && r.text.contains("navigation"), "{r:?}");

    // Everything above was audited, by tag only, in order.
    let entries = audit.0.lock().unwrap().clone();
    let tags: Vec<&str> = entries.iter().map(|e| e.cmd.as_str()).collect();
    assert_eq!(tags, ["mode:hold", "go", "relations:reload", "wb:off"]);
    assert!(entries.iter().all(|e| e.session == "00ff00ff00ff00ff"));
    let text: String = entries.iter().map(|e| e.to_line()).collect();
    assert!(!text.contains("p1") && !text.contains(&format!("p{first}")), "{text}");

    // Stopping the bot: the next command is told so, never left hanging.
    stop.store(true, Ordering::SeqCst);
    let _ = bot_thread.join().unwrap();
    let r = ask(&mut web, ControlCommand::Go {});
    assert_eq!(r.code, Some(ReplyCode::Gone), "{r:?}");
    drop(server);
    assert!(!sock.exists());
}
