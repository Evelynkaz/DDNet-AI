//! The console commands (task 4.3): the parser, `Bot::command`'s reply and its effect on the bot for every
//! command, the command bus, and the proof that nothing the console accepts can reach the game chat. No
//! network; every nickname is a `p<id>` test string.

mod support;

use std::time::Duration;

use ddai_bot::command::{
    BotCommand, BusError, CommandBus, GotoArg, HomeArg, ListArg, ParseError, TargetArg, parse_line,
};
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode, nav_hooks};
use ddai_bot::relations::ListKind;
use ddai_bot::{Bot, BotConfig, BrainKind, Mode, Relations};
use support::*;

// ---- the parser -----------------------------------------------------------------------------------

#[test]
fn every_command_parses_with_either_prefix_in_any_case() {
    use BotCommand as C;
    let table: Vec<(&str, BotCommand)> = vec![
        ("!help", C::Help),
        ("?HELP", C::Help),
        ("  !mode  ", C::Mode(None)),
        ("!mode fight", C::Mode(Some(Mode::Fight))),
        ("?mode Passive", C::Mode(Some(Mode::Passive))),
        ("!mode hold", C::Mode(Some(Mode::Hold))),
        ("!stop", C::Stop),
        ("?go", C::Go),
        ("!goto", C::Goto(GotoArg::Status)),
        ("!goto stop", C::Goto(GotoArg::Stop)),
        ("!goto -", C::Goto(GotoArg::Stop)),
        ("!goto OFF", C::Goto(GotoArg::Stop)),
        ("!goto tele", C::Goto(GotoArg::Tele)),
        ("!goto tp", C::Goto(GotoArg::Tele)),
        ("!goto 12 34", C::Goto(GotoArg::Tile { x: 12, y: 34 })),
        ("!goto 12,34", C::Goto(GotoArg::Tile { x: 12, y: 34 })),
        ("!goto -1 5", C::Goto(GotoArg::Tile { x: -1, y: 5 })),
        ("!goto bob", C::Goto(GotoArg::Player("bob".into()))),
        ("!goto @stop", C::Goto(GotoArg::Player("stop".into()))),
        ("!goto big bob", C::Goto(GotoArg::Player("big bob".into()))),
        ("!target", C::Target(TargetArg::Query)),
        ("!target -", C::Target(TargetArg::Clear)),
        ("!target Some Name", C::Target(TargetArg::Name("Some Name".into()))),
        (
            "!war",
            C::List {
                kind: ListKind::War,
                arg: ListArg::Show,
            },
        ),
        (
            "!war off",
            C::List {
                kind: ListKind::War,
                arg: ListArg::Clear,
            },
        ),
        (
            "!war bob",
            C::List {
                kind: ListKind::War,
                arg: ListArg::Toggle("bob".into()),
            },
        ),
        (
            "!friend bob",
            C::List {
                kind: ListKind::Friend,
                arg: ListArg::Toggle("bob".into()),
            },
        ),
        (
            "!ignore bob",
            C::List {
                kind: ListKind::Ignore,
                arg: ListArg::Toggle("bob".into()),
            },
        ),
        (
            "!clanwar FOES",
            C::List {
                kind: ListKind::ClanWar,
                arg: ListArg::Toggle("FOES".into()),
            },
        ),
        (
            "!clanfriend",
            C::List {
                kind: ListKind::ClanFriend,
                arg: ListArg::Show,
            },
        ),
        ("!home", C::Home(HomeArg::Here)),
        ("!home 3 4", C::Home(HomeArg::Tile { x: 3, y: 4 })),
        ("!home 3,4", C::Home(HomeArg::Tile { x: 3, y: 4 })),
        ("!home off", C::Home(HomeArg::Off)),
        ("!wb", C::Wb(None)),
        ("!wb on", C::Wb(Some(WbMode::Auto))),
        ("!wb left", C::Wb(Some(WbMode::Left))),
        ("!wb right", C::Wb(Some(WbMode::Right))),
        ("!wb off", C::Wb(Some(WbMode::Off))),
        ("!clip", C::Clip(String::new())),
        ("!clip nice one", C::Clip("nice one".into())),
        ("!stats", C::Stats),
        ("!where", C::Where),
        ("!brain", C::Brain(None)),
        ("!brain planner", C::Brain(Some(BrainKind::Planner))),
        ("!brain hybrid", C::Brain(Some(BrainKind::Hybrid))),
        ("!brain scripted", C::Brain(Some(BrainKind::Scripted))),
        ("!brain idle", C::Brain(Some(BrainKind::Idle))),
        ("!brain fly", C::Brain(Some(BrainKind::Fly))),
        ("!brain net", C::Brain(Some(BrainKind::Fly))),
        ("!low", C::Low(None)),
        ("!low on", C::Low(Some(true))),
        ("!low OFF", C::Low(Some(false))),
        ("!strong on", C::Strong(Some(true))),
        ("!strong", C::Strong(None)),
        ("!spec", C::Spec),
        ("!join", C::Join),
        ("!kill", C::Kill),
        ("!reset", C::Kill),
        ("!quit", C::Quit),
    ];
    for (line, want) in table {
        assert_eq!(parse_line(line), Ok(want), "{line:?}");
    }
}

#[test]
fn malformed_lines_say_why_instead_of_doing_something() {
    assert_eq!(parse_line(""), Err(ParseError::Empty));
    assert_eq!(parse_line("   "), Err(ParseError::Empty));
    assert_eq!(parse_line("!"), Err(ParseError::NoCommand));
    assert_eq!(parse_line("?   "), Err(ParseError::NoCommand));
    assert_eq!(
        parse_line("!frobnicate now"),
        Err(ParseError::Unknown("frobnicate".into()))
    );
    for bad in [
        "!mode dance",
        "!low maybe",
        "!strong 3",
        "!wb up",
        "!brain net2",
        "!home 1",
        "!home a b",
    ] {
        assert!(matches!(parse_line(bad), Err(ParseError::Usage(_))), "{bad}");
    }
    assert!(ParseError::Unknown("x".into()).message().contains("!help"));
}

#[test]
fn the_commands_the_old_bot_had_for_chat_and_chat_orders_are_refused_with_a_reason() {
    for line in [
        "!say hello",
        "?say hello",
        "!owner bob",
        "!llm on",
        "!d hi",
        "!emote happy",
        "!yes",
        "!vote x",
        "!duel on",
        "!style wb",
        "!try fast",
        "!lang ru",
        "!log on",
    ] {
        let Ok(BotCommand::Unsupported(why)) = parse_line(line) else {
            panic!("{line}: must be Unsupported");
        };
        assert!(!why.is_empty());
    }
}

// ---- no chat, ever --------------------------------------------------------------------------------

/// Every variant, listed so that adding one (say, a `Say`) breaks this build and asks for a decision:
/// none of these carries text for the server, and the type has nothing the runner could pass to a chat call.
fn variant_name(c: &BotCommand) -> &'static str {
    match c {
        BotCommand::Help => "help",
        BotCommand::Mode(_) => "mode",
        BotCommand::Stop => "stop",
        BotCommand::Go => "go",
        BotCommand::Goto(_) => "goto",
        BotCommand::Target(_) => "target",
        BotCommand::List { .. } => "list",
        BotCommand::Home(_) => "home",
        BotCommand::Wb(_) => "wb",
        BotCommand::Clip(_) => "clip",
        BotCommand::Stats => "stats",
        BotCommand::Where => "where",
        BotCommand::Brain(_) => "brain",
        BotCommand::Low(_) => "low",
        BotCommand::Strong(_) => "strong",
        BotCommand::Spec => "spec",
        BotCommand::Join => "join",
        BotCommand::Kill => "kill",
        BotCommand::ReloadRelations => "reload_relations",
        BotCommand::Quit => "quit",
        BotCommand::Unsupported(_) => "unsupported",
    }
}

#[test]
fn a_line_without_a_prefix_never_reaches_the_bot_and_nothing_is_sent_anywhere() {
    let (sender, inbox) = CommandBus::open();
    let lines = [
        "hello",
        "hello everyone",
        "say hello",
        "/accept",
        "/kick p1",
        "gg",
        "ё",
        "  hi there",
        "1 2 3",
        "bot stop",
        "% !say hi",
        ".!kill",
        "a!kill",
        "\tx",
        "mode hold",
        "stop",
        "quit",
    ];
    for line in lines {
        let r = sender
            .send_line(line, Duration::from_millis(50))
            .expect("answered locally");
        assert!(!r.ok, "{line:?} must be refused");
        assert!(r.text.contains("never writes in the game chat"), "{line:?}: {}", r.text);
        assert!(!r.quit);
        assert!(inbox.try_next().is_none(), "{line:?} reached the bot's command channel");
    }
    // A prefixed command does go through (and waits for the bot: here nobody answers).
    assert_eq!(
        sender.send_line("!stats", Duration::from_millis(20)),
        Err(BusError::Timeout)
    );
    assert!(inbox.try_next().is_some());
}

#[test]
fn whatever_is_typed_parses_to_a_command_or_an_error_and_never_panics() {
    // A deterministic stream of odd lines: every one is `Ok(command)` or `Err`, and an unprefixed one is
    // always `NoPrefix` (or `Empty`).
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let alphabet: Vec<char> = "!?/ \t\u{e9}\u{44b}-,@0123456789abcxyzABCmodegotsayhelp"
        .chars()
        .collect();
    for _ in 0..20_000 {
        let len = {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 24) as usize
        };
        let mut line = String::new();
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            line.push(alphabet[(x % alphabet.len() as u64) as usize]);
        }
        match parse_line(&line) {
            Ok(c) => {
                assert!(
                    matches!(line.trim().chars().next(), Some('!' | '?')),
                    "{line:?} -> {}",
                    variant_name(&c)
                );
            }
            Err(ParseError::NoPrefix) => assert!(!matches!(line.trim().chars().next(), Some('!' | '?') | None)),
            Err(_) => {}
        }
    }
}

// ---- applying -------------------------------------------------------------------------------------

struct Rig {
    bot: Bot,
    sc: Scenario,
    nav: NavHandle,
    dir: tempfile::TempDir,
}

fn rig_with(tees: Vec<TeeSpec>, kind: BrainKind, mutate: impl FnOnce(&mut BotConfig, &std::path::Path)) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = cfg(kind);
    cfg.relations_path = Some(dir.path().join("relations.json"));
    cfg.settings_path = Some(dir.path().join("settings.toml"));
    cfg.clips.dir = Some(dir.path().join("clips"));
    cfg.clips.autoclip = false;
    cfg.console_names = true; // most tests read the names back; `replies_use_tags_...` turns it off
    mutate(&mut cfg, dir.path());
    let nav = NavHandle::new();
    let navcfg = NavConfig {
        memory_dir: None,
        ..NavConfig::default()
    };
    let brain: Box<dyn ddai_brain::Brain> = Box::new(ddai_brain::IdleBrain);
    let mut bot = Bot::new(cfg, brain, nav_hooks(navcfg, nav.clone()), Relations::new());
    bot.set_nav_handle(nav.clone());
    let map = room(&[]);
    bot.on_map_loaded(std::sync::Arc::clone(&map));
    Rig {
        bot,
        sc: Scenario::new(map, tees),
        nav,
        dir,
    }
}

fn rig(tees: Vec<TeeSpec>) -> Rig {
    rig_with(tees, BrainKind::Idle, |_, _| {})
}

impl Rig {
    fn say(&mut self, line: &str) -> ddai_bot::CommandReply {
        let cmd = parse_line(line).unwrap_or_else(|e| panic!("{line:?}: {e:?}"));
        self.bot.command(cmd)
    }

    fn step(&mut self, n: usize) -> Vec<ddai_bot::Output> {
        let mut outs = Vec::new();
        for _ in 0..n {
            for id in 1..=12 {
                if let Some(t) = self.sc.tees.iter_mut().find(|t| t.id == id) {
                    t.angle = (t.angle + 37) % 1000;
                }
            }
            outs.extend(run(&mut self.bot, &mut self.sc, 1));
        }
        outs
    }
}

#[test]
fn help_says_there_is_no_chat() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000)]);
        let reply = r.say("!help");
        assert!(reply.ok);
        for word in [
            "!goto",
            "!target",
            "!clip",
            "!brain",
            "!kill",
            "!quit",
            "never writes in the game chat",
        ] {
            assert!(reply.text.contains(word), "help lacks {word}");
        }
        assert!(!reply.text.contains("!say"), "no !say in the help");
    });
}

#[test]
fn mode_stop_and_go_change_the_mode_and_clear_the_target() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1200)]);
        assert_eq!(r.say("!mode").text, "mode: fight (fight | passive | hold)");
        r.step(10);
        assert_eq!(r.bot.target_id(), 1);
        let reply = r.say("!mode passive");
        assert_eq!((reply.ok, reply.text.as_str()), (true, "mode: passive"));
        assert_eq!(r.bot.mode(), Mode::Passive);
        r.step(3);
        assert_eq!(r.bot.target_id(), -1, "passive never fights");
        assert_eq!(r.say("!stop").text, "stopped");
        assert_eq!(r.bot.mode(), Mode::Hold);
        let outs = r.step(3);
        assert!(
            outs.iter().all(|o| o.input.is_some_and(|i| i.fire == 0 && i.hook == 0)),
            "hold idles"
        );
        assert_eq!(r.say("!go").text, "playing");
        assert_eq!(r.bot.mode(), Mode::Fight);
        r.step(10);
        assert_eq!(r.bot.target_id(), 1, "and fights again");
        assert_eq!(r.say("!mode hold").text, "mode: hold");
        assert_eq!(r.bot.mode(), Mode::Hold);
    });
}

#[test]
fn goto_starts_a_walk_reports_it_and_stop_ends_it() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1200)]);
        // A tile on the floor of the room, a few tiles away.
        let (tx, ty) = (1000 / 32 + 6, (FLOOR_Y / 32));
        let reply = r.say(&format!("!goto {tx} {ty}"));
        assert!(reply.ok, "{}", reply.text);
        assert!(reply.text.contains("sent to the navigation"), "{}", reply.text);
        r.step(2);
        let lines = r.nav.drain_replies();
        assert!(
            lines.iter().any(|l| l.starts_with("goto:")),
            "the navigation answers: {lines:?}"
        );
        assert_eq!(r.bot.mode(), Mode::Goto, "a walk is running");
        let st = r.say("!goto");
        assert!(st.ok && st.text.starts_with("goto:"), "{}", st.text);
        let stop = r.say("!stop");
        assert!(stop.ok);
        r.step(2);
        let lines = r.nav.drain_replies();
        assert!(lines.iter().any(|l| l.contains("cancelled")), "{lines:?}");
        assert_eq!(r.bot.mode(), Mode::Fight, "back to the mode the walk began from");
        assert!(!r.nav.status().walking);

        // A wall and a place off the map are refused by the navigation with a reason.
        r.say("!goto 0 0");
        r.say("!goto 9999 9999");
        r.step(2);
        let lines = r.nav.drain_replies().join(" | ");
        assert!(lines.contains("is a wall"), "{lines}");
        assert!(lines.contains("is off the map"), "{lines}");
        // A map without teleporters.
        r.say("!goto tele");
        r.step(2);
        assert!(r.nav.drain_replies().join(" ").contains("no teleport layer"));
        // `!goto stop` with nothing running.
        r.say("!goto stop");
        r.step(2);
        assert!(r.nav.drain_replies().join(" ").contains("not going anywhere"));
    });
}

#[test]
fn goto_a_player_follows_the_one_player_it_names_and_complains_about_the_rest() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1500), tee(10, 1800)]);
        r.step(4);
        let reply = r.say("!goto p10");
        assert!(reply.ok && reply.text.contains("following p10"), "{}", reply.text);
        let reply = r.say("!goto @p1");
        assert!(
            reply.ok && reply.text.contains("following p1"),
            "exact beats substring: {}",
            reply.text
        );
        let reply = r.say("!goto p");
        assert!(!reply.ok && reply.text.contains("matches 2 players"), "{}", reply.text);
        let reply = r.say("!goto nobody");
        assert!(
            !reply.ok && reply.text.contains("nobody named 'nobody'"),
            "{}",
            reply.text
        );
    });
}

#[test]
fn target_is_exact_or_a_unique_part_of_a_name_and_fights_only_that_player() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500), tee(10, 1900)]);
        r.sc.player_mut(2).name = "Zed Zedson".to_string();
        r.step(10);
        assert_eq!(r.bot.target_id(), 1, "the nearest");
        assert_eq!(r.say("!target").text, "target: automatic");
        let reply = r.say("!target Zedson");
        assert_eq!(reply.text, "target set to 'Zed Zedson'");
        r.step(10);
        assert_eq!(r.bot.target_id(), 2, "a part of the name is enough when unique");
        let q = r.say("!target").text;
        assert!(q.starts_with("target: only '") && q.contains("zed"), "{q}");
        let reply = r.say("!target p1");
        assert_eq!(reply.text, "target set to 'p1'", "exact wins over the substring in p10");
        r.step(10);
        assert_eq!(r.bot.target_id(), 1);
        let reply = r.say("!target p");
        assert!(!reply.ok && reply.text.contains("matches 2 players"), "{}", reply.text);
        assert_eq!(r.bot.target_id(), 1, "unchanged by a refused command");
        let reply = r.say("!target ghost");
        assert!(
            reply.ok && reply.text.contains("nobody by that name is on the server now"),
            "{}",
            reply.text
        );
        r.step(4);
        assert_eq!(r.bot.target_id(), -1, "no such player: nobody to fight");
        let reply = r.say("!target -");
        assert!(reply.ok);
        r.step(10);
        assert_eq!(r.bot.target_id(), 1, "automatic again");
    });
}

#[test]
fn the_list_commands_toggle_persist_and_move_a_name_between_lists() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500), tee(10, 1900)]);
        r.step(4);
        let path = r.dir.path().join("relations.json");
        assert_eq!(r.say("!friend").text, "friends: nobody");
        assert_eq!(r.say("!friend p1").text, "friends: p1");
        assert!(r.bot.relations().contains(ListKind::Friend, "P1"), "folded, exact");
        assert!(
            Relations::load(&path).unwrap().contains(ListKind::Friend, "p1"),
            "saved to the file"
        );
        r.step(10);
        assert_eq!(r.bot.target_id(), 2, "a friend is never a target");
        assert_eq!(r.say("!friend").text, "friends: p1");
        // The same name again removes it.
        assert_eq!(r.say("!friend p1").text, "friends: removed p1");
        assert!(!Relations::load(&path).unwrap().contains(ListKind::Friend, "p1"));
        // War takes a name off the friend list.
        r.say("!friend p1");
        assert_eq!(r.say("!war p1").text, "war: p1 (was on the friend list)");
        assert!(r.bot.relations().contains(ListKind::War, "p1") && !r.bot.relations().contains(ListKind::Friend, "p1"));
        // Ignore removes it from war; a part of a name completes only to one player.
        assert_eq!(r.say("!ignore p2").text, "ignored: p2");
        let reply = r.say("!war p");
        assert!(!reply.ok && reply.text.contains("matches 3 players"), "{}", reply.text);
        assert!(
            r.bot.relations().contains(ListKind::War, "p1"),
            "unchanged by a refused command"
        );
        // A name nobody has yet goes on the list as typed.
        assert_eq!(r.say("!war Someone Else").text, "war: Someone Else");
        assert!(r.bot.relations().contains(ListKind::War, "Someone Else"));
        // Clans are taken as typed.
        assert_eq!(r.say("!clanwar FOES").text, "clan war: FOES");
        assert_eq!(
            r.say("!clanfriend FOES").text,
            "friendly clans: FOES (was on the clanwar list)"
        );
        assert!(Relations::load(&path).unwrap().contains(ListKind::ClanFriend, "foes"));
        // Off clears one list only.
        assert_eq!(r.say("!war off").text, "war: cleared");
        assert_eq!(r.say("!war").text, "war: nobody");
        assert!(Relations::load(&path).unwrap().contains(ListKind::Ignore, "p2"));
    });
}

#[test]
fn reload_relations_makes_the_lists_the_files_and_keeps_them_when_the_file_is_bad() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)]);
        let path = r.dir.path().join("relations.json");
        r.step(10);
        let first = r.bot.target_id();
        assert!(
            first == 1 || first == 2,
            "nobody listed: somebody is the target ({first})"
        );
        let other = 3 - first;

        // The web editor writes the file (folded keys, like every writer) ...
        let mut edited = Relations::new();
        edited.add(ListKind::Friend, &format!("  P{first} "));
        edited.add(ListKind::ClanWar, "Foes");
        edited.save(&path).unwrap();
        let v0 = r.bot.relations().version();
        // ... and asks the bot to reload it.
        let reply = r.bot.command(BotCommand::ReloadRelations);
        assert!(reply.ok, "{}", reply.text);
        assert_eq!(
            reply.text, "lists reloaded (friend 1, war 0, ignore 0, clanwar 1, clanfriend 0)",
            "counts only, no names"
        );
        let data = reply.data.clone().expect("structured counts and digest");
        assert_eq!(data["counts"]["friend"], 1);
        assert_eq!(
            data["digest"],
            edited.digest(),
            "the digest the web computes from the same lists"
        );
        assert!(r.bot.relations().version() > v0, "the per-player flags are recomputed");
        assert!(r.bot.relations().contains(ListKind::Friend, &format!("p{first}")));
        assert!(r.bot.relations().contains(ListKind::ClanWar, "foes"));
        r.step(10);
        assert_eq!(r.bot.target_id(), other, "the new friend is spared at once");

        // An unchanged file does not bump the counter (nothing to recompute).
        let v1 = r.bot.relations().version();
        assert!(r.bot.command(BotCommand::ReloadRelations).ok);
        assert_eq!(r.bot.relations().version(), v1);

        // Removing the friend in the file takes them off the list.
        Relations::new().save(&path).unwrap();
        assert!(r.bot.command(BotCommand::ReloadRelations).ok);
        assert!(r.bot.relations().is_empty());
        r.step(10);
        assert_ne!(
            r.bot.target_id(),
            -1,
            "(the current target is kept: the picker has hysteresis)"
        );

        // A corrupt file: refused, the running lists stay (a corrupt file must never become "no friends").
        r.say("!friend p1");
        std::fs::write(&path, "{ not json").unwrap();
        let reply = r.bot.command(BotCommand::ReloadRelations);
        assert!(!reply.ok, "{}", reply.text);
        assert!(
            !reply.text.contains("not json") && !reply.text.contains("p1"),
            "{}",
            reply.text
        );
        assert!(r.bot.relations().contains(ListKind::Friend, "p1"), "kept");
        // A missing file is an empty set of lists, as at start.
        std::fs::remove_file(&path).unwrap();
        assert!(r.bot.command(BotCommand::ReloadRelations).ok);
        assert!(r.bot.relations().is_empty());
    });
}

#[test]
fn a_console_edit_starts_from_the_file_and_waits_for_the_other_writer() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)]);
        let path = r.dir.path().join("relations.json");
        // The site added a friend to the file; the bot has not been told to reload yet.
        let mut web = Relations::new();
        web.add(ListKind::Friend, "from the site");
        web.save(&path).unwrap();
        assert!(r.say("!war p2").ok);
        let file = Relations::load(&path).unwrap();
        assert!(
            file.contains(ListKind::Friend, "from the site") && file.contains(ListKind::War, "p2"),
            "the console edit did not erase the site's: it started from the file"
        );
        assert!(r.bot.relations().contains(ListKind::Friend, "from the site"));
        // While the site (or anyone) holds the lock, the edit is refused after a bounded wait, not applied.
        let held = ddai_bot::relations::RelationsLock::acquire(&path).unwrap();
        let started = std::time::Instant::now();
        let reply = r.say("!friend p1");
        assert!(!reply.ok && reply.text.contains("busy"), "{}", reply.text);
        // The edit runs on the decision thread (D-042): it must give up within milliseconds, not wait for the holder.
        assert!(
            started.elapsed() < std::time::Duration::from_millis(150),
            "a busy lock answered after {:?}",
            started.elapsed()
        );
        assert!(!r.bot.relations().contains(ListKind::Friend, "p1"), "not applied");
        drop(held);
        assert!(r.say("!friend p1").ok);
    });
}

#[test]
fn reload_relations_without_a_lists_file_is_refused() {
    big_stack(|| {
        let mut r = rig_with(vec![tee(0, 1000)], BrainKind::Idle, |c, _| c.relations_path = None);
        let reply = r.bot.command(BotCommand::ReloadRelations);
        assert!(!reply.ok, "{}", reply.text);
    });
}

#[test]
fn a_list_that_cannot_be_saved_says_so_but_still_applies() {
    big_stack(a_list_that_cannot_be_saved_body);
}

fn a_list_that_cannot_be_saved_body() {
    let mut r = rig_with(vec![tee(0, 1000), tee(1, 1100)], BrainKind::Idle, |c, dir| {
        // A directory where the file should be: the atomic rename fails.
        std::fs::create_dir_all(dir.join("relations.json")).unwrap();
        c.relations_path = Some(dir.join("relations.json"));
    });
    r.step(2);
    let reply = r.say("!ignore p1");
    assert!(!reply.ok && reply.text.contains("could not be saved"), "{}", reply.text);
    assert!(
        r.bot.relations().contains(ListKind::Ignore, "p1"),
        "it applies to the running bot"
    );
}

#[test]
fn home_and_wb_go_to_the_navigation_and_its_answers_come_back() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1900)]);
        r.step(2);
        r.say("!home 3 4");
        r.say("!home");
        r.say("!home off");
        r.say("!wb left");
        r.step(2);
        let lines = r.nav.drain_replies();
        assert!(lines.iter().any(|l| l.contains("home: (3,4)")), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("home: here (")), "{lines:?}");
        assert!(lines.iter().any(|l| l == "home: off"), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("WB: left")), "{lines:?}");
        let settings = std::fs::read_to_string(r.dir.path().join("settings.toml")).unwrap();
        assert!(settings.contains("wb = \"left\""), "{settings}");
        let reply = r.say("!wb");
        assert!(reply.ok);
    });
}

#[test]
fn clip_saves_the_ring_and_says_where() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1900)]);
        r.step(50);
        let reply = r.say("!clip first test");
        assert!(reply.ok && reply.text.starts_with("saved 2 s to "), "{}", reply.text);
        let files: Vec<_> = std::fs::read_dir(r.dir.path().join("clips"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(files.len(), 1);
        assert!(
            files[0].starts_with("manual-") && files[0].ends_with("first_test.clip"),
            "{files:?}"
        );
        // Without a directory there is nowhere to save.
        let mut plain = rig_with(vec![tee(0, 1000)], BrainKind::Idle, |c, _| c.clips.dir = None);
        plain.step(5);
        let reply = plain.say("!clip");
        assert!(!reply.ok && reply.text.contains("no clip directory"), "{}", reply.text);
    });
}

#[test]
fn stats_and_where_describe_the_bot() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1200)]);
        let before = r.say("!where");
        assert!(before.text.contains("no tee on the map"), "{}", before.text);
        r.step(10);
        let stats = r.say("!stats").text;
        for word in [
            "10 snapshots",
            "decisions",
            "self-kills",
            "blocks",
            "brain idle",
            "mode fight",
            "clip ring 10 frames",
        ] {
            assert!(stats.contains(word), "{stats}");
        }
        let at = r.say("!where").text;
        assert!(at.starts_with("tile ("), "{at}");
        assert!(
            at.contains("playing") && at.contains("free") && at.contains("target c1-"),
            "{at}"
        );
        assert!(!at.contains("p1"), "the target is shown by its tag: {at}");
        r.say("!stop");
        assert!(r.say("!where").text.contains("stopped"));
    });
}

#[test]
fn brain_swaps_live_and_remembers_it() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1200)]);
        r.step(10);
        assert_eq!(
            r.say("!brain").text,
            "brain: idle (hybrid | planner | scripted | idle | fly)"
        );
        for (cmd, name) in [("planner", "planner"), ("scripted", "scripted"), ("idle", "idle")] {
            let reply = r.say(&format!("!brain {cmd}"));
            assert!(reply.ok, "{}", reply.text);
            assert_eq!(reply.text, format!("brain: {name}"));
            assert_eq!(r.bot.config().brain, BrainKind::parse(cmd).unwrap());
            assert!(
                r.bot.brain_name().starts_with(name_of_brain(cmd)),
                "the live brain is the new one: {}",
                r.bot.brain_name()
            );
            r.step(6); // and the bot goes on deciding with it
        }
        // The fly cannot load here (no files): refused, the old brain stays.
        let reply = r.say("!brain fly");
        assert!(
            !reply.ok && reply.text.starts_with("brain: not switched"),
            "{}",
            reply.text
        );
        assert_eq!(r.bot.config().brain, BrainKind::Idle);
        let settings = std::fs::read_to_string(r.dir.path().join("settings.toml")).unwrap();
        assert!(settings.contains("brain = \"idle\""), "{settings}");
        // The clip header names the new brain.
        r.say("!brain scripted");
        r.step(3);
        let saved = r.bot.save_clip("").unwrap();
        assert_eq!(ddai_clip::Clip::read(&saved.path).unwrap().header.brain, "scripted");
    });
}

fn name_of_brain(cmd: &str) -> &'static str {
    match cmd {
        "planner" => "planner",
        "scripted" => "scripted",
        _ => "idle",
    }
}

#[test]
fn low_and_strong_are_remembered_and_exclude_each_other() {
    big_stack(|| {
        let mut r = rig_with(vec![tee(0, 1000), tee(1, 1200)], BrainKind::Planner, |_, _| {});
        r.bot.set_brain_options(ddai_bot::BrainOptions::default());
        assert!(r.say("!low").text.contains("off"));
        let reply = r.say("!low on");
        assert!(
            reply.ok && reply.text.contains("the planner's search is shorter"),
            "{}",
            reply.text
        );
        assert!(r.bot.low());
        r.step(4);
        let reply = r.say("!strong on");
        assert!(
            reply.ok && reply.text.contains("weak-PC mode is off now"),
            "{}",
            reply.text
        );
        assert!(r.bot.strong() && !r.bot.low());
        r.step(2);
        assert!(r.nav.drain_replies().iter().any(|l| l.starts_with("strong mode: on")));
        let s = std::fs::read_to_string(r.dir.path().join("settings.toml")).unwrap();
        assert!(s.contains("strong = true") && s.contains("low = false"), "{s}");
        // With another brain `low` says it changes the planner only.
        r.say("!brain idle");
        let reply = r.say("!low on");
        assert!(
            reply.ok && reply.text.contains("changes the planner brain only"),
            "{}",
            reply.text
        );
        assert!(r.say("!strong off").ok);
        assert!(!r.bot.strong());
    });
}

#[test]
fn spec_and_join_ask_the_server_once_and_a_chosen_spectator_is_not_a_moderation_stop() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1200)]);
        r.step(5);
        assert_eq!(r.say("!spec").text, "going to the spectators");
        assert!(r.bot.wants_spectate());
        let outs = r.step(1);
        assert_eq!(outs[0].set_team, Some(-1));
        assert_eq!(r.step(1)[0].set_team, None, "asked once");
        // The server moved us: no tee, team -1. We asked for it: no stop, no join request.
        let own = r.sc.tees.remove(0);
        r.sc.player_mut(0).team = -1;
        for _ in 0..80 {
            let outs = r.step(1);
            assert_eq!(outs[0].set_team, None, "no auto-join while !spec");
        }
        assert!(
            r.bot.stop_reason().is_none(),
            "our own !spec is not a moderation signal"
        );
        // `!join` asks to come back. The server needs a while: team -1 and no tee for a few snapshots
        // is still our own doing (review F2), not a stop.
        assert_eq!(r.say("!join").text, "joining the game");
        assert!(!r.bot.wants_spectate() && r.bot.join_grace_active());
        assert_eq!(r.step(1)[0].set_team, Some(0));
        for _ in 0..10 {
            r.step(1);
            assert!(r.bot.stop_reason().is_none(), "the join is still settling");
        }
        // The server shows us again: the grace ends.
        r.sc.tees.insert(0, own.clone());
        r.sc.player_mut(0).team = 0;
        r.step(3);
        assert!(!r.bot.join_grace_active(), "cleared once a tee is back");
        assert!(r.bot.stop_reason().is_none());
        // A LATER forced move to the spectators is a moderation signal again.
        r.sc.tees.remove(0);
        r.sc.player_mut(0).team = -1;
        r.step(3);
        assert!(
            r.bot.stop_reason().is_some(),
            "after a completed join a forced move still stops the bot"
        );

        // The grace also ends when the server shows team != -1 although the tee is not there yet (dead, waiting).
        let mut r3 = rig(vec![tee(0, 1000), tee(1, 1200)]);
        r3.step(5);
        r3.say("!spec");
        r3.step(2);
        r3.sc.tees.remove(0);
        r3.sc.player_mut(0).team = -1;
        r3.step(3);
        r3.say("!join");
        r3.sc.player_mut(0).team = 0; // in the game, no tee yet
        r3.step(3);
        assert!(!r3.bot.join_grace_active());
        r3.sc.player_mut(0).team = -1; // now moved out again
        r3.step(3);
        assert!(r3.bot.stop_reason().is_some());

        // A join that never completes: after the grace the usual rule applies (conservative stop).
        let mut r4 = rig(vec![tee(0, 1000), tee(1, 1200)]);
        r4.step(5);
        r4.say("!spec");
        r4.step(2);
        r4.sc.tees.remove(0);
        r4.sc.player_mut(0).team = -1;
        r4.step(3);
        r4.say("!join");
        r4.step(100);
        assert!(
            r4.bot.stop_reason().is_none(),
            "within the grace (15 s = 375 snapshots)"
        );
        r4.step(300);
        assert!(
            r4.bot.stop_reason().is_some(),
            "the join never came: stop, as for a moderation move"
        );

        // Without `!spec` the same situation IS the moderation stop (D-016): unchanged.
        let mut r2 = rig(vec![tee(0, 1000), tee(1, 1200)]);
        r2.step(5);
        r2.sc.tees.remove(0);
        r2.sc.player_mut(0).team = -1;
        r2.step(3);
        assert!(r2.bot.stop_reason().is_some());
    });
}

#[test]
fn kill_sends_cl_kill_once_per_cooldown_and_the_clip_says_why() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1200)]);
        r.step(5);
        r.say("!stop"); // standing still: the unstick rules stay out of this test's kill count
        assert_eq!(r.say("!kill").text, "killing, respawning");
        let outs = r.step(1);
        assert!(outs[0].kill, "Cl_Kill goes out with the next snapshot");
        assert_eq!(r.bot.stats().self_kills, 1);
        let again = r.say("!reset");
        assert!(!again.ok && again.text == "reset is on cooldown", "{}", again.text);
        assert!(r.step(5).iter().all(|o| !o.kill));
        // 500 ticks later (250 snapshots of two ticks) it is allowed again.
        for _ in 0..250 {
            r.step(1);
        }
        assert!(r.say("!kill").ok);
        assert!(r.step(1)[0].kill);
        let saved = r.bot.save_clip("").unwrap();
        let clip = ddai_clip::Clip::read(&saved.path).unwrap();
        let why: Vec<u8> = clip
            .frames
            .iter()
            .flat_map(|f| f.events.iter())
            .filter_map(|e| match e {
                ddai_clip::ClipEvent::KillSent { why } => Some(*why),
                _ => None,
            })
            .collect();
        assert_eq!(
            why,
            vec![ddai_clip::KillWhy::Console as u8; 2],
            "both console kills are in the clip"
        );
    });
}

#[test]
fn quit_asks_the_runner_to_stop() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000)]);
        assert!(!r.bot.quit_requested());
        let reply = r.say("!quit");
        assert!(reply.ok && reply.quit && reply.text == "disconnecting");
        assert!(r.bot.quit_requested());
    });
}

#[test]
fn refused_and_unsupported_commands_change_nothing() {
    big_stack(|| {
        let mut r = rig(vec![tee(0, 1000), tee(1, 1200)]);
        r.step(10);
        let (mode, target, stats) = (r.bot.mode(), r.bot.target_id(), r.bot.stats());
        for line in [
            "!say hello everyone",
            "!owner p1",
            "!duel on",
            "!vote kick",
            "!emote happy",
        ] {
            let reply = r.say(line);
            assert!(!reply.ok, "{line}");
        }
        assert_eq!((r.bot.mode(), r.bot.target_id()), (mode, target));
        assert_eq!(r.bot.stats(), stats);
        assert!(r.step(1).iter().all(|o| !o.kill && o.set_team.is_none()));
    });
}

#[test]
fn the_bus_carries_a_command_to_the_bot_and_the_reply_back() {
    big_stack(|| {
        let (sender, inbox) = CommandBus::open();
        let mut r = rig(vec![tee(0, 1000)]);
        let asker = std::thread::spawn(move || {
            let a = sender.send_line("!mode hold", Duration::from_secs(5));
            let b = sender.send_line("hello", Duration::from_secs(5)); // not a command: answered without the bot
            let c = sender.send_line("!stats", Duration::from_secs(5));
            (a, b, c)
        });
        // The bot's loop: between snapshots, answer what is waiting.
        let mut answered = 0;
        for _ in 0..2000 {
            while let Some(req) = inbox.try_next() {
                let reply = r.bot.command(req.cmd);
                let _ = req.reply.send(reply);
                answered += 1;
            }
            if asker.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let (a, b, c) = asker.join().unwrap();
        assert_eq!(answered, 2, "the unprefixed line never came");
        assert_eq!(a.unwrap().text, "mode: hold");
        assert!(!b.unwrap().ok);
        assert!(c.unwrap().text.contains("snapshots"));
        assert_eq!(r.bot.mode(), Mode::Hold);
        // A bot that has gone: the sender learns it.
        let (sender2, inbox2) = CommandBus::open();
        drop(inbox2);
        assert_eq!(
            sender2.send(BotCommand::Help, Duration::from_millis(50)),
            Err(BusError::Gone)
        );
    });
}

#[test]
fn replies_name_other_players_by_tag_unless_console_names_is_on() {
    big_stack(|| {
        let mut r = rig_with(
            vec![tee(0, 1000), tee(1, 1100), tee(2, 1500), tee(10, 1900)],
            BrainKind::Idle,
            |c, _| c.console_names = false,
        );
        r.step(4);
        let tag1 = r.bot.tag_of(1).to_string();
        let tag2 = r.bot.tag_of(2).to_string();
        let mut said = Vec::new();
        for line in [
            "!target p2",
            "!target",
            "!target ghost",
            "!target p",
            "!friend p1",
            "!friend",
            "!war p",
            "!war Someone Else",
            "!goto p10",
            "!goto p",
            "!goto nobody",
            "!ignore p2",
            "!ignore",
        ] {
            said.push((line, r.say(line).text));
        }
        for (line, text) in &said {
            for nick in ["p1", "p2", "p10", "ghost", "Someone Else", "someone else"] {
                // A tag like `c1-1a2b3c4d` never contains these; a nickname would.
                let mut t = text.clone();
                for tag in [&tag1, &tag2] {
                    t = t.replace(tag.as_str(), "");
                }
                assert!(!t.contains(nick), "{line:?} echoed {nick:?}: {text}");
            }
        }
        let get = |l: &str| said.iter().find(|(x, _)| *x == l).unwrap().1.clone();
        assert_eq!(get("!target p2"), format!("target set to {tag2}"));
        assert!(get("!friend p1").contains(&tag1));
        assert!(get("!target").starts_with("target: one fixed player"));
        assert!(get("!target ghost").contains("nobody by that name"));
        assert!(get("!friend").contains("on the list") && get("!friend").contains("--console-names"));
        assert!(get("!war p").contains("matches 3 players"), "{}", get("!war p"));
        // The effects are the same as with names.
        assert!(r.bot.relations().contains(ListKind::Friend, "p1"));
        assert!(r.bot.relations().contains(ListKind::War, "Someone Else"));

        // With the flag the old text.
        let mut named = rig(vec![tee(0, 1000), tee(1, 1100)]);
        named.step(3);
        assert_eq!(named.say("!target p1").text, "target set to 'p1'");
        assert_eq!(named.say("!friend p1").text, "friends: p1");
    });
}
