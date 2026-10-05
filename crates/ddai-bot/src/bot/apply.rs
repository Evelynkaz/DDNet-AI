//! `Bot::command` (task 4.3): applying a [`BotCommand`] to a running bot. A child module of `bot` so it
//! can reach the bot's state; nothing here touches the network — the effects that need the wire (`Cl_Kill`,
//! `Cl_SetTeam`) are flags the next [`Bot::on_snapshot`] turns into its [`Output`]. There is no code path
//! to the game chat: [`BotCommand`] has no variant that carries text to the server.
//!
//! Names typed by the operator are matched against the roster **exactly after folding** (D-021), and, for
//! `!target`, `!goto` and the list commands, completed to the one player whose folded name contains the
//! typed text (the TS `playersMatching`); two or more matches are an error that lists them. The replies are
//! for the local console and name other players **by tag** (`c<id>-<hash>`) unless `--console-names` is on
//! (`BotConfig::console_names`); the runner prints them and never logs them.

use std::fmt::Write as _;
use std::sync::Arc;

use ddai_brain::ResetContext;
use ddai_clip::format::KillWhy;

use super::{Bot, Mode, Output};
use crate::brains::{BrainKind, make_brain};
use crate::command::{BotCommand, CommandReply, GotoArg, HomeArg, ListArg, TargetArg};
use crate::names::fold_name;
use crate::nav_hooks::NavCommand;
use crate::relations::{ListKind, Relations};

const HELP: &str = "\
  !help                  this text
  !stop / !go            stop playing and stand still (or end a walk) / resume
  !mode [fight|passive|hold]   fight (default) | passive (never engage) | hold
  !goto tele             walk to the nearest teleporter
  !goto <x> <y>          walk to that tile; '!goto' alone reports progress; '!goto stop' calls it off
  !goto <nick> / @nick   walk to that player and follow them
  !target <nick> | -     fight only this player (a part of the name is enough if it is unique) / clear
  !war [name|off]        fight them on sight  ·  !friend [name|off]  never touch them, defend them
  !ignore [name|off]     never touch them, never answer them
  !clanwar [clan|off]    the same by clan tag  ·  !clanfriend [clan|off]
  !home [x y|off]        mark a spot to return to when there is nobody to fight
  !wb [off|left|right|auto]  hold the wayblock (Copy Love Box)
  !clip [note]           save the last 30 s to a file for review (the note goes into the clip and its
                         file name, which are meant to be shared: no nicknames in it)
  !stats / !where        counters / position, target and freeze state
  !brain [hybrid|planner|scripted|idle|fly]   swap the brain live
  !low [on|off]          the planner's short search for a weak PC (the planner brain only)
  !strong [on|off]       a wider search inside the wayblock hall (more CPU)
  !spec / !join          go to the spectators / back into the game
  !kill / !reset         kill and respawn (500-tick cooldown)
  !quit                  disconnect and exit

  '?' works too. A line without ! or ? goes nowhere: the console never writes in the game chat (the owner's chat lines come from the website).";

/// How long a console list edit waits for the lists file's cross-process lock. It runs on the decision thread (D-042:
/// p99 <= 5 ms), so the wait is a few milliseconds, not the web's 2 s: a taken lock is answered with "busy, try again".
pub(super) const LISTS_LOCK_WAIT: std::time::Duration = std::time::Duration::from_millis(15);
// The bound itself is checked here, at compile time, so no test has to time a call on a loaded machine.
const _: () = assert!(LISTS_LOCK_WAIT.as_millis() <= 20);

/// How long after `!join` a spectator state is still read as the operator's own doing (15 s).
pub(super) const JOIN_GRACE_TICKS: i32 = 750;

enum Found {
    One(i32, String),
    Many(Vec<(i32, String)>),
    Nobody,
}

impl Bot {
    /// Applies one command and says what happened.
    pub fn command(&mut self, cmd: BotCommand) -> CommandReply {
        match cmd {
            BotCommand::Help => CommandReply::ok(HELP),
            BotCommand::Unsupported(why) => CommandReply::err(why),
            BotCommand::Mode(None) => CommandReply::ok(format!("mode: {} (fight | passive | hold)", self.mode.name())),
            BotCommand::Mode(Some(m)) => {
                // A walk does not outlive a mode change: the navigation notices it at its next poll.
                let walking = self.walking();
                self.set_mode(m);
                CommandReply::ok(format!(
                    "{}mode: {}",
                    if walking { "goto: ended; " } else { "" },
                    m.name()
                ))
            }
            BotCommand::Stop => {
                if self.walking() {
                    self.nav_send(NavCommand::Stop)
                } else {
                    self.set_mode(Mode::Hold);
                    CommandReply::ok("stopped")
                }
            }
            BotCommand::Go => {
                let walking = self.walking();
                self.set_mode(Mode::Fight);
                CommandReply::ok(format!("{}playing", if walking { "goto: ended; " } else { "" }))
            }
            BotCommand::Goto(a) => self.goto(a),
            BotCommand::Target(a) => self.target(a),
            BotCommand::List { kind, arg } => self.list(kind, arg),
            BotCommand::Home(HomeArg::Here) => self.nav_send(NavCommand::HomeHere),
            BotCommand::Home(HomeArg::Tile { x, y }) => self.nav_send(NavCommand::SetHome { tx: x, ty: y }),
            BotCommand::Home(HomeArg::Off) => self.nav_send(NavCommand::HomeOff),
            BotCommand::Wb(Some(m)) => {
                let r = self.nav_send(NavCommand::Wb(m));
                self.remember(|s| s.wb = Some(m.name().to_string()));
                r
            }
            BotCommand::Wb(None) => match &self.nav {
                Some(nav) => {
                    let wb = nav.status().wb;
                    CommandReply::ok(if wb.is_empty() {
                        "WB: no status yet".to_string()
                    } else {
                        wb
                    })
                }
                None => CommandReply::err("navigation is not available in this bot"),
            },
            BotCommand::Clip(note) => self.clip(&note),
            BotCommand::Stats => CommandReply::ok(self.stats_line()),
            BotCommand::Where => CommandReply::ok(self.where_line()),
            BotCommand::Brain(None) => CommandReply::ok(format!(
                "brain: {} (hybrid | planner | scripted | idle | fly)",
                self.cfg.brain.name()
            )),
            BotCommand::Brain(Some(k)) => self.switch_brain(k),
            BotCommand::Low(None) => CommandReply::ok(format!(
                "mode for a weak PC: {}; !low on | off",
                if self.low { "on" } else { "off" }
            )),
            BotCommand::Low(Some(on)) => self.set_low(on),
            BotCommand::Strong(None) => CommandReply::ok(format!(
                "strong mode: {}; !strong on | off",
                if self.strong { "on" } else { "off" }
            )),
            BotCommand::Strong(Some(on)) => self.set_strong(on),
            BotCommand::Spec => {
                self.wants_spectate = true;
                self.join_grace_until = None;
                self.pending_team = Some(-1);
                CommandReply::ok("going to the spectators")
            }
            BotCommand::Join => {
                self.wants_spectate = false;
                // The server needs a moment to put us back; until then team -1 is not a moderation move.
                self.join_grace_until = Some(self.last_tick.max(0) + JOIN_GRACE_TICKS);
                self.pending_team = Some(0);
                CommandReply::ok("joining the game")
            }
            BotCommand::Kill => {
                if self.paused {
                    return CommandReply::err("paused by the server: no kill sent");
                }
                if !self.unstick.cooldown_ready(self.last_tick) {
                    return CommandReply::err("reset is on cooldown");
                }
                self.pending_kill = true;
                CommandReply::ok("killing, respawning")
            }
            BotCommand::ReloadRelations => self.reload_relations(),
            BotCommand::Quit => {
                self.quit = true;
                CommandReply {
                    text: "disconnecting".to_string(),
                    ok: true,
                    quit: true,
                    data: None,
                }
            }
            // The owner's website line is applied by the runner's `OwnerChat` (rate limits, the queue, in the game only): it
            // needs the wall clock and the client, which this sans-IO state machine has neither of. It never says anything itself.
            BotCommand::Say { .. } => {
                CommandReply::err("chat lines are sent by the runner, not by the bot state machine")
            }
        }
    }

    /// What the commands left for this snapshot's output: a `Cl_SetTeam` and a `Cl_Kill`.
    pub(super) fn apply_pending(&mut self, snap: &ddai_client::LiveWorldSnapshot, out: &mut Output) {
        if let Some(team) = self.pending_team.take() {
            out.set_team = Some(team);
        }
        if self.pending_kill {
            self.pending_kill = false;
            let tick = snap.tick;
            // Paused by the server (task 4.9b): no kill at all, the operator's included (it is dropped, not kept for later).
            if !self.paused && !out.kill && self.unstick.cooldown_ready(tick) {
                self.unstick.note_external_kill(tick);
                self.hooks.navigator.kill_sent(tick, false);
                self.stats.self_kills += 1;
                out.kill = true;
                self.kill_why = Some(KillWhy::Console);
            }
        }
    }

    // ---- pieces ----------------------------------------------------------------------------------

    fn walking(&self) -> bool {
        self.nav.as_ref().is_some_and(|n| n.status().walking)
    }

    fn nav_send(&mut self, cmd: NavCommand) -> CommandReply {
        match &self.nav {
            Some(nav) => {
                nav.send(cmd);
                CommandReply::ok("sent to the navigation; its answer follows")
            }
            None => CommandReply::err("navigation is not available in this bot"),
        }
    }

    /// Remembers a setting, when a settings file is configured.
    fn remember(&self, change: impl FnOnce(&mut crate::settings::Settings)) {
        if let Some(path) = &self.cfg.settings_path
            && let Err(e) = crate::settings::update(path, change)
        {
            tracing::warn!(error = %e, "could not save the settings");
        }
    }

    /// How a reply names a player: by tag (`c<id>-<hash>`), or by nickname only behind `--console-names`
    /// (the console may run under a unit whose journal is shared; the nicknames of others stay out of it).
    fn shown(&self, id: i32, name: &str) -> String {
        if self.cfg.console_names {
            name.to_string()
        } else {
            self.players.tag(id).to_string()
        }
    }

    /// `'name'` behind `--console-names`, the tag otherwise.
    fn quoted(&self, id: i32, name: &str) -> String {
        if self.cfg.console_names {
            format!("'{name}'")
        } else {
            self.players.tag(id).to_string()
        }
    }

    fn shown_list(&self, list: &[(i32, String)]) -> String {
        list.iter()
            .map(|(id, n)| self.shown(*id, n))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The text the operator typed, echoed back only behind `--console-names`.
    fn echo(&self, typed: &str) -> String {
        if self.cfg.console_names {
            format!("'{typed}'")
        } else {
            "that name".to_string()
        }
    }

    /// A player by the typed text: the exact folded name, else the one whose folded name contains it.
    fn find_player(&self, text: &str) -> Found {
        let key = fold_name(text);
        if key.is_empty() {
            return Found::Nobody;
        }
        let own = self.players.own_id();
        let mut contains: Vec<(i32, String)> = Vec::new();
        for (id, slot) in self.players.present() {
            if Some(id) == own {
                continue;
            }
            if slot.name_key == key {
                return Found::One(id, slot.name.clone());
            }
            if slot.name_key.contains(&key) {
                contains.push((id, slot.name.clone()));
            }
        }
        match contains.len() {
            0 => Found::Nobody,
            1 => {
                let (id, name) = contains.remove(0);
                Found::One(id, name)
            }
            _ => Found::Many(contains),
        }
    }

    fn goto(&mut self, a: GotoArg) -> CommandReply {
        match a {
            GotoArg::Status => match &self.nav {
                Some(nav) => {
                    let st = nav.status();
                    CommandReply::ok(if st.walking {
                        format!("goto: {}", st.progress)
                    } else if st.last_walk.is_empty() {
                        "nothing set: '!goto <x> <y>' (tiles), '!goto tele' or '!goto <nick>'".to_string()
                    } else {
                        format!("nothing set; the last walk: {}", st.last_walk)
                    })
                }
                None => CommandReply::err("navigation is not available in this bot"),
            },
            GotoArg::Stop => self.nav_send(NavCommand::Stop),
            GotoArg::Tele => self.nav_send(NavCommand::GotoTele),
            GotoArg::Tile { x, y } => self.nav_send(NavCommand::Goto {
                tx: x,
                ty: y,
                through_freeze: true,
            }),
            GotoArg::Player(name) => {
                if name.is_empty() {
                    return CommandReply::err("whose name? '!goto <nick>' or '!goto @nick'");
                }
                match self.find_player(&name) {
                    Found::One(id, who) => {
                        let who = self.shown(id, &who);
                        let r = self.nav_send(NavCommand::Follow { client_id: id });
                        CommandReply {
                            text: format!("goto: following {who}; {}", r.text),
                            ..r
                        }
                    }
                    Found::Many(list) => CommandReply::err(format!(
                        "goto: {} matches {} players: {} -- be more specific",
                        self.echo(&name),
                        list.len(),
                        self.shown_list(&list)
                    )),
                    Found::Nobody => CommandReply::err(format!("nobody named {} is on the server", self.echo(&name))),
                }
            }
        }
    }

    fn target(&mut self, a: TargetArg) -> CommandReply {
        match a {
            TargetArg::Query => CommandReply::ok(match self.picker.fixed() {
                Some(n) if self.cfg.console_names => format!("target: only '{n}'; '!target -' to pick automatically"),
                Some(_) => "target: one fixed player; '!target -' to pick automatically".to_string(),
                None => "target: automatic".to_string(),
            }),
            TargetArg::Clear => {
                self.picker.set_fixed(None);
                CommandReply::ok("target cleared, back to picking automatically")
            }
            TargetArg::Name(text) => match self.find_player(&text) {
                Found::One(id, name) => {
                    self.picker.set_fixed(Some(&name));
                    CommandReply::ok(format!("target set to {}", self.quoted(id, &name)))
                }
                Found::Many(list) => CommandReply::err(format!(
                    "target: {} matches {} players: {} -- be more specific",
                    self.echo(&text),
                    list.len(),
                    self.shown_list(&list)
                )),
                Found::Nobody => {
                    if fold_name(&text).is_empty() {
                        return CommandReply::err("target: that name is empty");
                    }
                    self.picker.set_fixed(Some(&text));
                    CommandReply::ok(format!(
                        "target set to {} (nobody by that name is on the server now)",
                        self.echo(&text)
                    ))
                }
            },
        }
    }

    fn list(&mut self, kind: ListKind, arg: ListArg) -> CommandReply {
        // An edit is read-modify-write of the lists file, which the web editor writes too: take the cross-process
        // lock for the whole of it and start from what the file holds now, so an edit made on the site a moment ago
        // is neither lost nor torn (task 5.6 review F6). Showing a list needs neither.
        let _lock = if matches!(arg, ListArg::Show) {
            None
        } else {
            match self.lock_relations() {
                Ok(lock) => lock,
                Err(why) => return CommandReply::err(why),
            }
        };
        let label = match kind {
            ListKind::War => "war",
            ListKind::Friend => "friends",
            ListKind::Ignore => "ignored",
            ListKind::ClanWar => "clan war",
            ListKind::ClanFriend => "friendly clans",
        };
        let is_clan = matches!(kind, ListKind::ClanWar | ListKind::ClanFriend);
        let text = match arg {
            ListArg::Show => {
                let names = self.relations.names(kind);
                return CommandReply::ok(if names.is_empty() {
                    format!("{label}: nobody")
                } else if self.cfg.console_names {
                    format!("{label}: {}", names.join(", "))
                } else {
                    format!(
                        "{label}: {} on the list (names hidden: start with --console-names to see them)",
                        names.len()
                    )
                });
            }
            ListArg::Clear => {
                self.relations.clear(kind);
                format!("{label}: cleared")
            }
            ListArg::Toggle(typed) => {
                // A player's name may be completed from a part of it; a clan tag is taken as typed.
                let mut resolved: Option<i32> = None;
                let what = if is_clan {
                    typed
                } else {
                    match self.find_player(&typed) {
                        Found::One(id, name) => {
                            resolved = Some(id);
                            name
                        }
                        Found::Many(list) => {
                            return CommandReply::err(format!(
                                "{label}: {} matches {} players: {} -- be more specific",
                                self.echo(&typed),
                                list.len(),
                                self.shown_list(&list)
                            ));
                        }
                        Found::Nobody => typed,
                    }
                };
                // How the reply names it: the entry itself behind `--console-names`; else the tag of the
                // player it resolved to, or a plain "that name" (a clan tag is not a nickname: shown as is).
                let disp = if self.cfg.console_names || is_clan {
                    what.clone()
                } else if let Some(id) = resolved {
                    self.players.tag(id).to_string()
                } else {
                    "that name".to_string()
                };
                let key = fold_name(&what);
                if key.is_empty() {
                    return CommandReply::err(format!("{label}: that name is empty"));
                }
                if self.relations.contains_key(kind, &key) {
                    self.relations.remove(kind, &what);
                    format!("{label}: removed {disp}")
                } else {
                    let moved: Vec<&str> = Relations::exclusions(kind)
                        .iter()
                        .filter(|&&o| self.relations.contains_key(o, &key))
                        .map(|&o| match o {
                            ListKind::War => "war",
                            ListKind::Friend => "friend",
                            ListKind::Ignore => "ignore",
                            ListKind::ClanWar => "clanwar",
                            ListKind::ClanFriend => "clanfriend",
                        })
                        .collect();
                    self.relations.add(kind, &what);
                    if moved.is_empty() {
                        format!("{label}: {disp}")
                    } else {
                        format!("{label}: {disp} (was on the {} list)", moved.join("/"))
                    }
                }
            }
        };
        match self.save_relations() {
            Ok(()) => CommandReply::ok(text),
            Err(e) => CommandReply::err(format!("{text} -- but the lists could not be saved: {e}")),
        }
    }

    /// Replaces the lists with the file's (the web editor wrote it). A file that cannot be read or does not parse
    /// leaves the running lists as they are: a corrupt file must not become "no friends" (`Relations::load`).
    /// The reply holds counts and a digest only, never a name (`CommandReply::data`): the web compares the digest with
    /// what it wrote, which tells it whether the bot reads the same file.
    fn reload_relations(&mut self) -> CommandReply {
        let Some(path) = &self.cfg.relations_path else {
            return CommandReply::err("lists: no lists file is configured, nothing to reload");
        };
        match Relations::load(path) {
            Ok(new) => {
                if !self.relations.same_lists(&new) {
                    self.relations.replace_with(new);
                }
                let counts: serde_json::Map<String, serde_json::Value> = ListKind::ALL
                    .iter()
                    .map(|&k| (k.name().to_string(), self.relations.len(k).into()))
                    .collect();
                let text = ListKind::ALL
                    .iter()
                    .map(|&k| format!("{} {}", k.name(), self.relations.len(k)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut reply = CommandReply::ok(format!("lists reloaded ({text})"));
                reply.data = Some(serde_json::json!({"counts": counts, "digest": self.relations.digest()}));
                reply
            }
            Err(e) => {
                // `RelationsError`'s text names the file, the line and the column, never a serde message (which
                // would quote a nickname): safe to log and to put in the reply.
                tracing::warn!(error = %e, "the lists file was not reloaded; keeping the running lists");
                CommandReply::err("lists: the file could not be read; keeping the running lists")
            }
        }
    }

    /// The cross-process lock of the lists file (`None`: no file is configured). Waits a bounded time; the lists are
    /// then brought up to date with the file (an unreadable file keeps the running lists).
    fn lock_relations(&mut self) -> Result<Option<ddai_botctl::relations::RelationsLock>, String> {
        let Some(path) = self.cfg.relations_path.clone() else {
            return Ok(None);
        };
        let lock = ddai_botctl::relations::RelationsLock::acquire_within(&path, LISTS_LOCK_WAIT).map_err(|e| {
            format!(
                "lists: the lists file is busy or cannot be locked ({}); try again in a moment",
                e.kind()
            )
        })?;
        if let Ok(file) = Relations::load(&path)
            && !self.relations.same_lists(&file)
        {
            self.relations.replace_with(file);
        }
        Ok(Some(lock))
    }

    fn save_relations(&self) -> std::io::Result<()> {
        match &self.cfg.relations_path {
            Some(path) => self.relations.save(path),
            None => Ok(()),
        }
    }

    fn clip(&mut self, note: &str) -> CommandReply {
        let frames = self.clipper.frames();
        match self.save_clip(note) {
            Ok(saved) => {
                self.stats.clips_saved += 1;
                CommandReply::ok(format!("saved {} s to {}", frames / 25, saved.path.display()))
            }
            Err(e) => CommandReply::err(format!("no clip: {e}")),
        }
    }

    fn stats_line(&self) -> String {
        let s = &self.stats;
        let total = self.latency.total.summary();
        let b = self.clock.stats();
        let mut out = String::new();
        let _ = write!(
            out,
            "tick {}, {} snapshots ({} collapsed), {} decisions ({} brain, {} wander), decide p50 {} us p99 {} us; \
             self-kills {}, deaths {}, blocks {}, blocked by {}, guarded {}, vetoed {} hooks {} hammers; \
             brain {}, mode {}; clip ring {} frames",
            self.last_tick,
            s.snapshots,
            s.collapsed,
            s.decisions,
            s.brain_decisions,
            s.wander_decisions,
            total.p50_us,
            total.p99_us,
            s.self_kills,
            s.deaths,
            b.blocks,
            b.blocked_by,
            s.guarded_inputs,
            s.vetoed_hooks,
            s.vetoed_fires,
            self.cfg.brain.name(),
            self.mode.name(),
            self.clipper.frames(),
        );
        out
    }

    fn where_line(&self) -> String {
        let own = self.players.own_id().and_then(|id| self.tees.get(id));
        let mut out = String::new();
        let Some(own) = own else {
            let _ = write!(
                out,
                "no tee on the map, mode {}{}, tick {}",
                self.mode.name(),
                if self.wants_spectate { ", spectating" } else { "" },
                self.last_tick
            );
            return out;
        };
        let _ = write!(
            out,
            "tile ({},{}), {}, {}",
            (own.pos.x / 32.0).trunc() as i32,
            (own.pos.y / 32.0).trunc() as i32,
            match self.mode {
                Mode::Hold => "stopped",
                _ => "playing",
            },
            if own.frozen { "frozen" } else { "free" }
        );
        let target = self.picker.target();
        match self.tees.get(target) {
            Some(t) if target >= 0 => {
                let _ = write!(
                    out,
                    ", target {} at {} px",
                    self.players.tag(target),
                    crate::tees::dist(own.pos, t.pos).round() as i32
                );
            }
            _ => out.push_str(", target none"),
        }
        let _ = write!(out, ", mode {}", self.mode.name());
        if let Some(nav) = &self.nav {
            let st = nav.status();
            if !st.wb.is_empty() {
                let _ = write!(out, ", {}", st.wb);
            }
            if st.walking {
                let _ = write!(out, ", goto: {}", st.progress);
            }
        }
        let _ = write!(out, ", tick {}", self.last_tick);
        out
    }

    /// Replaces the brain, keeping everything else (the clip ring, the modes, the lists).
    fn switch_brain(&mut self, kind: BrainKind) -> CommandReply {
        self.brain_opts.planner_preset = if self.low {
            ddai_planner::brains::PlannerPreset::Low
        } else {
            ddai_planner::brains::PlannerPreset::Normal
        };
        let brain = match make_brain(kind, &self.brain_opts) {
            Ok(b) => b,
            Err(e) => return CommandReply::err(format!("brain: not switched: {e}")),
        };
        self.brain = brain;
        self.brain_generation += 1;
        self.cfg.brain = kind;
        self.clipper.set_brain(self.brain.name());
        if let (Some(map), Some(own)) = (&self.map, self.players.own_id()) {
            self.brain.reset(&ResetContext {
                map: Arc::clone(map),
                self_id: own,
                seed: self.cfg.seed.wrapping_add(self.lives),
            });
        }
        // The new brain knows nothing of the dead zone and the freeze memory yet.
        self.hooks.navigator.resend_knowledge();
        self.remember(|s| s.brain = Some(kind.name().to_string()));
        CommandReply::ok(format!("brain: {}", kind.name()))
    }

    fn set_low(&mut self, on: bool) -> CommandReply {
        self.low = on;
        self.remember(|s| s.low = Some(on));
        let mut text = if on {
            "mode for a weak PC: on".to_string()
        } else {
            "mode for a weak PC: off (the full search)".to_string()
        };
        if self.cfg.brain == BrainKind::Planner {
            let r = self.switch_brain(BrainKind::Planner);
            if !r.ok {
                return r;
            }
            text.push_str(" (the planner's search is shorter)");
        } else {
            text.push_str(&format!(
                "; it changes the planner brain only, and this one is {} (its search is capped at 5 ms, D-042)",
                self.cfg.brain.name()
            ));
        }
        if on && self.strong {
            let _ = self.set_strong(false);
            text.push_str("; strong mode is off now");
        }
        CommandReply::ok(text)
    }

    fn set_strong(&mut self, on: bool) -> CommandReply {
        if self.nav.is_none() {
            return CommandReply::err("navigation is not available in this bot");
        }
        self.strong = on;
        let mut text = if on {
            "strong mode: on (inside a wayblock hall the planner searches wider: more CPU)".to_string()
        } else {
            "strong mode: off (the plain search)".to_string()
        };
        let _ = self.nav_send(NavCommand::Strong(on));
        self.remember(|s| s.strong = Some(on));
        if on && self.low {
            let _ = self.set_low(false);
            text.push_str("; weak-PC mode is off now");
        }
        CommandReply::ok(text)
    }
}
