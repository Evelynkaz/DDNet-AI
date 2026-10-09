//! `ddnet-ai play --brain planner|scripted|hybrid|fly` (or `--bot`): the live bot (task 4.1,
//! `ddai-bot`) against a DDNet 20.x server. The same `play` command the demo brains use; this file is
//! the bot half. Loopback only until D-043's gate is met (`ClientConfig`'s live-servers allow-list
//! enforces it on every connect).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::Args;
use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::nav_hooks::{NavCommand, NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunReport, RunnerConfig, RunnerError};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;

use crate::play_cmd::{Brain, PlayArgs};

/// The bot-specific flags of `ddnet-ai play`.
#[derive(Debug, Args, Default)]
pub struct BotOpts {
    /// `fight` (default), `passive` (wander only), `hold` (stand still) or `goto` (a walk begun by
    /// `--goto`/`--follow` ends back in the mode the bot started in).
    #[arg(long, default_value = "fight")]
    pub mode: String,
    /// Fight only this player (exact name, folded); every other filter is ignored for them.
    #[arg(long)]
    pub target: Option<String>,
    /// Friend/war/ignore lists (never in git). Default `<data-dir>/bot/relations.json`.
    #[arg(long)]
    pub relations: Option<PathBuf>,
    /// Unix socket for the web unit's live view. Default `<data-dir>/bot/live.sock`.
    #[arg(long)]
    pub bridge: Option<PathBuf>,
    /// Do not open the live-view socket.
    #[arg(long)]
    pub no_bridge: bool,
    /// Send real nicknames to the web unit (default: salted-hash tags only).
    #[arg(long)]
    pub web_names: bool,
    /// Write real nicknames next to their tags into this local file (debug; off by default).
    #[arg(long)]
    pub debug_names: Option<PathBuf>,
    /// Write a JSON report (latency percentiles, counters, outgoing-message audit) at the end; also
    /// turns the outgoing-message audit on.
    #[arg(long)]
    pub report: Option<PathBuf>,
    /// The planner's wall budget per decision, ms (D-042).
    #[arg(long, default_value_t = 5.0)]
    pub planner_budget_ms: f64,
    /// Threads that score the hybrid brain's candidates, the deciding thread included (task 3.7a, D-080): `1` (the
    /// default) = the deciding thread alone; `auto` = the cores nobody is using, at most 4 (opt-in: E-012 found no
    /// gain on a loaded machine). A helper never delays the decision past its deadline.
    #[arg(long, default_value = "1", value_parser = parse_search_threads)]
    pub search_threads: SearchThreads,
    /// Diagnostics (task 3.7a, D-080): leave the proposer's time out of the hybrid's decision cap, as before 3.7a
    /// (the A/B of E-012). The default counts it.
    #[arg(long)]
    pub no_proposal_in_cap: bool,
    /// The hybrid brain's opponent model (task 3.7b, D-090): `on` (the default) predicts the victim's plan by a small search from
    /// its seat; `off` = the victim holds its input, as before 3.7b. Measured against planners, scripted and passive tees, not yet
    /// against people: the way back if the first live session shows it hurts.
    #[arg(long, default_value = "on", value_parser = parse_on_off, action = clap::ArgAction::Set)]
    pub hybrid_mirror: bool,
    /// The hybrid's duel rules (task 3.19, D-116): `off` (the default), `reflex` (swing the hammer when the opponent is free, in reach and the hit would throw it into a
    /// freeze), `reflex-all` (swing at every chance: worse than nothing in the arena), `envelope` (no jump or hook climb that a worst-case hammer hit would carry into a
    /// freeze ceiling) or `both`. They act only in a detected F-DDrace duel. **None passed its go bars** (docs/research/duel-3.19.md): tools for a live measurement.
    /// `duel_hammer = "<mode>"` in `settings.toml` does the same (the flag wins). Off until the arena numbers (docs/research/duel-3.19.md) and a live session say otherwise.
    #[arg(long, default_value = "off", value_parser = parse_duel_hammer)]
    pub duel_hammer: String,
    /// The hybrid's fixes for the weaknesses of the 2026-10-08 duel against a human (task 3.23, D-121): `off` (the default), `all`, or a comma list of `static` (a
    /// standing or AFK opponent: never dropped by the AFK filter, answered by a plan that acts), `counter` (the opponent's "hook from above, pass under, release")
    /// and `finish` (a frozen victim lying off the freeze is dragged into it; no swing at a frozen tee). They act only in a detected F-DDrace duel. Off until the arena
    /// numbers (docs/research/duel-fixes-3.23.md) and a live session say otherwise.
    #[arg(long, default_value = "off", value_parser = parse_duel_fixes)]
    pub duel_fixes: String,
    /// Finish blocks (task 3.10, E-021, D-097): `off` (the default), `target` keeps a frozen current target until it is held (the bot's target
    /// logic only: the part with consistent evidence, the live A/B candidate), `full` adds the hybrid's drag-back shaping for a frozen victim
    /// (a duel gain that did not hold up in review: not recommended), `wb` (task 3.18, D-114) is `target` plus the wayblock hold: in the held
    /// Copy Love Box hall the hybrid throws a frozen victim toward the hall's freeze wall with the guard's wall swings. `on` = `full`, kept for
    /// compatibility. Off until a live session shows what people do against it.
    #[arg(long, default_value = "off", value_parser = parse_finish, action = clap::ArgAction::Set)]
    pub finish: FinishMode,
    /// Copy Love Box targeting (task 3.12, D-103): `off` (the default) or `on`. `on`: the hall is chosen by the number of blockable targets on
    /// each side (more is better, a tie is random, with hysteresis and a memory of failed tube crossings), and an idle (AFK) player is fought
    /// when he is in the way (in the hall we hold, next to us, on the route of our walk). The same switch (task 3.12b, D-104) also turns on
    /// the tube-crossing changes: after a respawn outside the hall the bot stands still and starts its walk at once, the crossing search
    /// rolls out with the other tees of the world, and an idle player on the ledge of the tube's start is jumped over.
    /// Off until the owner turns it on.
    #[arg(long, default_value = "off", value_parser = parse_on_off, action = clap::ArgAction::Set)]
    pub wb_smart: bool,
    /// Live input timing (task 3.11, E-024, D-101; **on** by default, `off` is the way back): a comma-separated list of
    /// `precise` (the driver sleeps on its input channel with a precise timeout instead of blocking in `recv` for up to 2 ms:
    /// a decision is picked up and an input sent when due, not ~1.5 ms later on average, ~4-8 ms at the tail; costs about 2 000
    /// wake-ups a second) and `kind` (a decision that runs the brain is aimed at the input slot by the brain decisions' own
    /// rolling p90, not the p90 over every decision, which is the cheap mode while the bot wanders). `off` = neither, as before 3.11.
    /// Without the flag the environment variable `DDAI_LIVE_TIMING` (same values) is read: the way back for the systemd unit.
    #[arg(long, default_value = "precise,kind", value_parser = parse_live_timing, action = clap::ArgAction::Set)]
    pub live_timing: LiveTiming,
    /// `--brain fly`: the compiled graph.
    #[arg(long)]
    pub fly_flyg: Option<PathBuf>,
    /// `--brain fly`: the brain config (`configs/fly/S-brain.toml`).
    #[arg(long)]
    pub fly_config: Option<PathBuf>,
    /// A trained fly bundle (task 7.4): `--brain fly` plays with its weights, `--brain hybrid` gets the fly as proposer
    /// (`hybrid:fly`). The web's «Муха» tab shows the fly's activity while the bot plays with either.
    #[arg(long)]
    pub fly_bundle: Option<PathBuf>,
    /// A **fixed** `cl_prediction_margin` in ms. Without it the margin is adaptive (task 4.1b, D-063):
    /// it starts at 10 and follows the `INPUTTIMING` feedback between 3 and 20 ms (rolling p1 of
    /// `time_left` kept at about 2 ms: lowered by 1 ms at a time, only when p1 is at least 4 ms and no
    /// input was late for 10 s; raised by 2 ms when two late inputs fall in the 5 s window, or when p1
    /// is below 2 ms over at least 100 samples — one late input never raises it; late inputs sent more
    /// than 18 ms too late are stalls no margin absorbs and are ignored).
    /// The margin is how early before a tick starts its input is due at the server; a smaller one
    /// moves the input slot later after each snapshot, giving a slow brain time to make it
    /// (`docs/formats.md` §21.6); too small and inputs arrive late (`INPUTTIMING` shows it in the
    /// report's `input_margin`).
    /// Task 3.16 (D-115): 0 to 30; also `prediction_margin_ms = ..` in the settings file (the flag wins). It is an **override**, not a floor:
    /// the adaptive controller is off for the run.
    #[arg(long, value_parser = clap::value_parser!(i32).range(0..=30))]
    pub prediction_margin_ms: Option<i32>,
    /// The hybrid's search budget in whole ms, 1 to 8 (task 3.16, D-115; also `hybrid_budget_ms = ..` in the settings file, the flag wins).
    /// The decision cap moves with it (budget + 1 ms: the default 4 ms / 5 ms pair is the point `4`); the extension of D-042 in danger is
    /// untouched. A shorter decision is a smaller share of the input lag the 20 ms tick rounds up (`docs/research/lag-shave.md`); the arena says a
    /// budget below 3 ms loses strength (E-026 section 4). Without it the default, byte for byte.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=8))]
    pub hybrid_budget_ms: Option<u32>,
    /// Walk to this tile `X,Y` as soon as the tee has spawned (task 4.2); afterwards the bot returns to
    /// its mode. Crosses freeze tubes where it must (the Copy Love Box wayblock tubes).
    #[arg(long, value_parser = parse_pair, conflicts_with = "follow")]
    pub goto: Option<(i32, i32)>,
    /// Walk to the player with this client id and keep following them while they move (task 4.2).
    #[arg(long)]
    pub follow: Option<i32>,
    /// Wayblock on Copy Love Box: `auto` (default; the side with fewer players, or with `--wb-smart on` the side with more blockable targets), `left`, `right`, `off`.
    #[arg(long, default_value = "auto")]
    pub wb: String,
    /// Strong mode: inside a wayblock hall the planner searches wider (`STRONG_WB`, more CPU).
    #[arg(long)]
    pub strong: bool,
    /// Never walk to where the game is when it is dull here.
    #[arg(long)]
    pub no_seek: bool,
    /// Directory of the freeze memories (one file per map, keyed by the map's sha256). Default
    /// `<data-dir>/bot/memory`.
    #[arg(long)]
    pub memory_dir: Option<PathBuf>,
    /// Do not read or write a freeze memory.
    #[arg(long)]
    pub no_memory: bool,
    /// `Cl_ShowDistance` half-extents `X,Y` (default 3000,2000 — D-007's replacement for /showall).
    #[arg(long, value_parser = parse_pair)]
    pub show_distance: Option<(i32, i32)>,
    /// Read commands from stdin (`!help` lists them; a line without `!` or `?` goes nowhere — the bot never
    /// writes in the game chat). On by default when stdin is a terminal.
    #[arg(long)]
    pub console: bool,
    /// Console replies name other players by their real nickname. Default: by tag (`c<id>-<hash>`), so the nicknames
    /// of others never reach a journal when the console runs under a unit.
    #[arg(long)]
    pub console_names: bool,
    /// Never read stdin, even from a terminal.
    #[arg(long, conflicts_with = "console")]
    pub no_console: bool,
    /// Where clips are written (default `<data-dir>/bot/clips`; never in git).
    #[arg(long)]
    pub clips_dir: Option<PathBuf>,
    /// Do not save incident and cross-fail clips by themselves (`!clip` still works).
    #[arg(long)]
    pub no_autoclip: bool,
    /// What `!brain`, `!wb`, `!low` and `!strong` remember (default `<data-dir>/bot/settings.toml`). The
    /// command line wins over the file; a file that does not parse is renamed `.bad-<ts>` and ignored.
    #[arg(long)]
    pub settings: Option<PathBuf>,
    /// Do not read or write a settings file.
    #[arg(long, conflicts_with = "settings")]
    pub no_settings: bool,
    /// The clan sent to the server (D-068: `Neuroset` by default; also `clan = "..."` in the settings file).
    #[arg(long)]
    pub clan: Option<String>,
    /// A fixed skin name (also `skin = "..."` in the settings file). Without it the bot picks a random stock skin at
    /// every start (D-068).
    #[arg(long)]
    pub skin: Option<String>,
    /// Makes the random skin pick deterministic (tests). Ignored when `--skin` is given.
    #[arg(long)]
    pub skin_seed: Option<u64>,
    /// Unix socket of the web control channel (task 5.6, D-070): the web unit's owner-only commands and the lists
    /// editor reach the bot through it. Default `<data-dir>/bot/control.sock` (mode 0600 in a 0700 directory).
    #[arg(long)]
    pub control: Option<PathBuf>,
    /// Do not open the web control socket.
    #[arg(long, conflicts_with = "control")]
    pub no_control: bool,
    /// Emergency switch (task 4.9, D-094; also `owner_chat = false` in the settings file): refuse the owner's chat lines from the
    /// website (`say`) while the rest of the control channel works. With it the bot cannot be made to say anything but the typed `/kill`.
    #[arg(long)]
    pub no_owner_chat: bool,
    /// Duel switch (task 4.11, D-102): the bot never kills itself (no unstick `Cl_Kill`, no `/kill` fallback, no route or trek with a
    /// respawn step; in F-DDrace `/1vs1` any death of ours is a point for the opponent). The owner's own `!kill` and website lines
    /// stay. The marker file `<data-dir>/bot/selfkill.off` does the same and is re-read once a second, so it can be toggled live.
    /// Task 5.15: also takes `=true|false` (the equals sign is required, so `--no-selfkill true` is not read as a value): the launch unit passes
    /// `--no-selfkill=${BOT_NO_SELFKILL}`, one argument that is never empty (a bare boolean flag cannot be given an empty one). Bare
    /// `--no-selfkill` is `true`, as before.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", default_value = "false", value_parser = parse_true_false, action = clap::ArgAction::Set)]
    pub no_selfkill: bool,
    /// Task 4.12 (D-108): when the bot kills itself on its own accord. `legacy` (default): the fixed timers (frozen 200/400 ticks, wedged
    /// 200/450, wayblock lying 25). `smart`: only when waiting costs more than a kill: deep freeze, no exit, a thaw slower than the respawn
    /// plus the walk back, and no friend near or hooking; the timers stay as upper bounds; a route's respawn step only when no route on
    /// foot exists. Either way an F-DDrace `/1vs1` duel is detected by itself (our DDRace team holds one other player **and** there is F-DDrace
    /// evidence: the server's `/1vs1` chat lines, or a duel command the owner sent from the website) and then nothing is killed, like
    /// `--no-selfkill`.
    #[arg(long, value_name = "legacy|smart", default_value = "legacy")]
    pub selfkill_policy: ddai_bot::smartkill::SelfKillPolicy,
    /// Task 4.12 (D-108): do not look for an F-DDrace `/1vs1` duel by itself (our DDRace team holds exactly one other player **and** the
    /// server has shown F-DDrace's `/1vs1` chat lines, or those lines say we accepted a fight). Then only `--no-selfkill` (and the marker
    /// file) stops the self-kills of a duel. Escape hatch: by default the detection is on, under either policy. The marker file
    /// `<data-dir>/bot/duel-detect.off` does the same and is re-read once a second.
    #[arg(long)]
    pub no_duel_detect: bool,
    /// The learned model of the opponent's inputs in the input-lag window (task 3.17, D-111; a `.oppnet` file, never in git; **off** by default,
    /// and then the bot decides byte for byte as without the flag). `window_model = "<file>"` in `settings.toml` does the same (the flag wins).
    /// With the hybrid brain the target plays the model's inputs in the roll over the lag window; an online guard benches the model when it
    /// predicts worse than "hold" (STATUS `window_model` / `window_guard`); the marker file `<data-dir>/bot/window-model.off` switches it off
    /// while it exists (re-read once a second). Per-window log: `<data-dir>/bot/oppnet-live.jsonl` (rotated; `ddnet-ai oppnet-live report`).
    /// The file's sha256 is logged at the start; a file that cannot be loaded refuses the start.
    #[arg(long, value_name = "FILE", value_parser = parse_path_or_empty)]
    pub window_model: Option<PathBuf>,
    /// The server's pre-inputs (task 3.20, D-112; `on` or `off`, **off** by default). A DDNet >= 19.4 server sends us the other tees' REAL inputs a
    /// little before their ticks (`Sv_PreInput`; we announce 20010, a bot that announces 19000 gets none). They are always counted and stored
    /// (STATUS `preinput_stats`, a log line every 10 s); with `on` they also play in the prediction, over the window model's and over "hold". The
    /// marker `<data-dir>/bot/preinput.off` switches the use off while it exists (re-read once a second). Receive-only: nothing is sent.
    #[arg(long, default_value = "off", value_parser = parse_on_off, action = clap::ArgAction::Set)]
    pub preinput: bool,
    /// Task 4.10 (D-100): do not send the DDNet timeout code `/timeout <code>` after joining (no seed file is read or made). With it the bot's
    /// only chat is the typed `/kill` and the owner's lines.
    #[arg(long)]
    pub no_timeout_code: bool,
    /// The control channel's audit log (command tags, session tags, outcomes; no nicknames). Default
    /// `<data-dir>/logs/bot/control-audit.log`.
    #[arg(long)]
    pub control_audit: Option<PathBuf>,
}

/// Whether `--flag` (or `--flag=value`) was typed on the command line, as opposed to being a default: a
/// value in the settings file only fills what the command line left open.
/// A path that may be empty (clap refuses an empty `PathBuf` value): `--window-model=` is "off".
fn parse_path_or_empty(s: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(s))
}

fn flag_given(flag: &str) -> bool {
    std::env::args().any(|a| a == flag || a.strip_prefix(flag).is_some_and(|r| r.starts_with('=')))
}

/// `--search-threads`: `None` = `auto`, else the number of threads (a newtype, so that clap does not read the
/// `Option` as "flag may be absent").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchThreads(pub Option<usize>);

impl Default for SearchThreads {
    /// One thread, like the `--search-threads` default.
    fn default() -> Self {
        SearchThreads(Some(1))
    }
}

/// `--finish`: which finishing switches are on (task 3.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FinishMode {
    /// None (the default).
    #[default]
    Off,
    /// The bot's target logic only (`TargetPicker::set_finish`).
    Target,
    /// The target logic and the hybrid's frozen-victim drag shaping (`HybridConfig::with_finish`).
    Full,
    /// The target logic and the wayblock hold (task 3.18, D-114): inside the held hall the hybrid throws a frozen victim toward the hall's freeze
    /// wall with the guard's wall swings (`HybridConfig::wb_hold`, the competitor's `wallDir`).
    Wb,
}

impl FinishMode {
    /// The bot keeps a frozen current target until it is held.
    pub fn target_logic(self) -> bool {
        self != FinishMode::Off
    }

    /// The hybrid's drag shaping is on.
    pub fn hybrid_drag(self) -> bool {
        self == FinishMode::Full
    }

    /// The hybrid's wayblock hold (the hall's wall swings against a frozen victim) is on.
    pub fn hybrid_wb_hold(self) -> bool {
        self == FinishMode::Wb
    }

    /// The word `--finish` takes for this mode (what the log line and the bot's STATUS say).
    pub fn name(self) -> &'static str {
        match self {
            FinishMode::Off => "off",
            FinishMode::Target => "target",
            FinishMode::Full => "full",
            FinishMode::Wb => "wb",
        }
    }
}

/// `--live-timing`: which of the task-3.11 timing switches are on (the flag's default is both; `Default` is neither).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveTiming {
    /// `ClientConfig::precise_wakeups`.
    pub precise: bool,
    /// `BotConfig::kind_estimate`.
    pub kind: bool,
}

fn parse_live_timing(s: &str) -> Result<LiveTiming, String> {
    let mut t = LiveTiming::default();
    for part in s
        .split(',')
        .map(|p| p.trim().to_ascii_lowercase())
        .filter(|p| !p.is_empty())
    {
        match part.as_str() {
            "off" => {}
            "precise" => t.precise = true,
            "kind" => t.kind = true,
            other => return Err(format!("expected `off` or a list of `precise`, `kind`, got {other:?}")),
        }
    }
    Ok(t)
}

fn parse_duel_hammer(s: &str) -> Result<String, String> {
    match ddai_bot::brains::duel_hammer(s) {
        Some(_) => Ok(s.to_ascii_lowercase()),
        None => Err(format!(
            "expected `off`, `reflex`, `reflex-all`, `envelope` or `both`, got {s:?}"
        )),
    }
}

fn parse_duel_fixes(s: &str) -> Result<String, String> {
    match ddai_bot::brains::duel_fixes(s) {
        Some(_) => Ok(s.to_ascii_lowercase()),
        None => Err(format!(
            "expected `off`, `all` or a list of `static`, `counter`, `finish`, got {s:?}"
        )),
    }
}

fn parse_finish(s: &str) -> Result<FinishMode, String> {
    match s.to_ascii_lowercase().as_str() {
        "off" => Ok(FinishMode::Off),
        "target" => Ok(FinishMode::Target),
        "full" | "on" => Ok(FinishMode::Full),
        "wb" => Ok(FinishMode::Wb),
        _ => Err(format!(
            "expected `off`, `target`, `wb` or `full` (`on` = `full`), got {s:?}"
        )),
    }
}

/// `--search-threads`: `auto` or a number from 1 to 16.
/// `on` / `off` (any case) for a switch flag.
fn parse_on_off(s: &str) -> Result<bool, String> {
    match s.to_ascii_lowercase().as_str() {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(format!("expected `on` or `off`, got {s:?}")),
    }
}

fn parse_true_false(s: &str) -> Result<bool, String> {
    match s.to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("expected `true` or `false`, got {s:?}")),
    }
}

fn parse_search_threads(s: &str) -> Result<SearchThreads, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(SearchThreads(None));
    }
    match s.trim().parse::<usize>() {
        Ok(n @ 1..=16) => Ok(SearchThreads(Some(n))),
        _ => Err("expected `auto` or a number from 1 to 16".to_string()),
    }
}

fn parse_pair(s: &str) -> Result<(i32, i32), String> {
    let (a, b) = s.split_once(',').ok_or("expected X,Y")?;
    Ok((
        a.trim().parse().map_err(|e| format!("{e}"))?,
        b.trim().parse().map_err(|e| format!("{e}"))?,
    ))
}

/// `--duration`: `0` means no limit (a systemd unit runs the bot until it is stopped); before task 4.4 it meant "stop at once".
fn run_for(seconds: u64) -> Option<Duration> {
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

/// The one-time startup warning when `MALLOC_MMAP_THRESHOLD_` is not set (Linux, glibc): without it the bot keeps ~160 MiB more
/// memory for good after a map change to a large map (task 4.4, E-009 F1). The variable cannot be set from inside the process
/// (no `unsafe`, no `mallopt`), so a manual run has to be told.
fn mmap_threshold_warning(value: Option<&std::ffi::OsStr>) -> Option<&'static str> {
    if !cfg!(target_os = "linux") || value.is_some_and(|v| !v.is_empty()) {
        return None;
    }
    Some(
        "MALLOC_MMAP_THRESHOLD_ is not set: after a map change to a large map glibc keeps ~160 MiB of the old map's memory for good; \
         start the bot with MALLOC_MMAP_THRESHOLD_=131072 (the systemd unit does; deploy/README.md)",
    )
}

fn kind_of(args: &PlayArgs) -> BrainKind {
    match args.brain {
        Brain::Planner => BrainKind::Planner,
        Brain::Scripted => BrainKind::Scripted,
        Brain::Hybrid => BrainKind::Hybrid,
        Brain::Fly => BrainKind::Fly,
        Brain::Idle | Brain::Circle | Brain::RandomScripted => BrainKind::Idle,
    }
}

fn default_relations(data_dir: &Path) -> PathBuf {
    data_dir.join("bot").join("relations.json")
}

/// Runs the bot; the process exit code is the runner's (0 ok, 3 kick/ban, 4 join failed).
pub fn run(args: &PlayArgs, data_dir: &Path, server: std::net::SocketAddr) -> ExitCode {
    let o = &args.bot_opts;
    if let Some(warning) = mmap_threshold_warning(std::env::var_os("MALLOC_MMAP_THRESHOLD_").as_deref()) {
        tracing::warn!("{warning}");
    }
    let Some(mode) = Mode::parse(&o.mode) else {
        eprintln!("unknown --mode {:?} (fight|passive|hold|goto)", o.mode);
        return ExitCode::FAILURE;
    };
    // The settings file (task 4.3): what `!brain`, `!wb`, `!low` and `!strong` remembered. The command line wins.
    let settings_path = (!o.no_settings).then(|| {
        o.settings
            .clone()
            .unwrap_or_else(|| data_dir.join("bot").join("settings.toml"))
    });
    // The owner chat fails closed (D-094): an unreadable file or an unknown key (a typo of `owner_chat`) switches it off.
    let mut settings_unreadable = false;
    let unknown_settings_keys = settings_path
        .as_deref()
        .map(ddai_bot::settings::unknown_keys)
        .unwrap_or_default();
    let settings = match settings_path.as_deref().map(ddai_bot::settings::load) {
        Some(ddai_bot::settings::Loaded::Ok(s)) => s,
        Some(ddai_bot::settings::Loaded::Corrupt { moved_to, why }) => {
            settings_unreadable = true;
            eprintln!(
                "settings: ignored ({why}){}",
                moved_to.map_or(String::new(), |p| format!("; the file is kept as {}", p.display()))
            );
            ddai_bot::settings::Settings::default()
        }
        None => ddai_bot::settings::Settings::default(),
    };
    let relations_path = o
        .relations
        .clone()
        .or_else(|| settings.relations.clone())
        .unwrap_or_else(|| default_relations(data_dir));
    let relations = match Relations::load(&relations_path) {
        Ok(r) => r,
        Err(e) => {
            // A corrupt list must not silently become "no friends" (the bot would attack them).
            eprintln!("refusing to start: {e}");
            return ExitCode::FAILURE;
        }
    };
    let identity = match ddai_bot::identity::resolve(
        ddai_bot::identity::Overrides {
            clan: o.clan.as_deref(),
            skin: o.skin.as_deref(),
            skin_seed: o.skin_seed,
        },
        &settings,
    ) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("identity: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut client = ClientConfig {
        name: args.name.clone(),
        clan: identity.clan.clone(),
        skin: identity.skin.clone(),
        cache_dir: data_dir.join("maps").join("cache"),
        timeout: args
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(ClientConfig::default().timeout),
        ..ClientConfig::default()
    };
    if let Err(e) = crate::proxy_cmd::prepare_client(&mut client, args.live_servers.as_deref(), server, data_dir) {
        eprintln!("refusing to connect: {e}");
        return ExitCode::from(e.exit);
    }
    // Task 4.10 (D-100): the persistent seed of the timeout code the session sends once after joining (`/timeout <code>`, as the official
    // client does), so a reconnect from a new address takes the old tee back. A seed that cannot be had is not fatal: no timeout code then.
    if o.no_timeout_code {
        eprintln!("timeout code: off (--no-timeout-code)");
    } else {
        match ddai_client::timeout_seed::load_or_create(&ddai_client::timeout_seed::path_in(data_dir)) {
            Ok(seed) => client.timeout_seed = Some(seed),
            Err(e) => eprintln!("timeout code: off ({e})"),
        }
    }
    // Task 3.16 (D-115): the two input-lag knobs, the flag over the settings file; nothing set = the behaviour of the build before them.
    let knobs = ddai_bot::timing_knobs::resolve(o.hybrid_budget_ms, o.prediction_margin_ms, &settings);
    for w in &knobs.warnings {
        eprintln!("{w}");
    }
    match knobs.prediction_margin_ms {
        Some(m) => client.prediction_margin_ms = m,
        None => client.adaptive_margin = true,
    }
    // `--live-timing` wins; without it `DDAI_LIVE_TIMING` (a systemd drop-in `Environment=DDAI_LIVE_TIMING=off` is the way back for
    // the unit, whose `ExecStart` cannot be overridden); without both, the flag's default.
    let live_timing = match std::env::var("DDAI_LIVE_TIMING") {
        Ok(v) if !flag_given("--live-timing") => match parse_live_timing(&v) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("refusing to start: DDAI_LIVE_TIMING: {e}");
                return ExitCode::FAILURE;
            }
        },
        _ => o.live_timing,
    };
    client.precise_wakeups = live_timing.precise;
    eprintln!(
        "live timing: precise wake-ups {}, brain-decision estimate {} (--live-timing / DDAI_LIVE_TIMING; task 3.11)",
        if live_timing.precise { "on" } else { "off" },
        if live_timing.kind { "on" } else { "off" }
    );
    if let Some(sd) = o.show_distance {
        client.show_distance = sd;
    }
    let kind = if flag_given("--brain") {
        kind_of(args)
    } else {
        settings
            .brain
            .as_deref()
            .and_then(BrainKind::parse)
            .unwrap_or_else(|| kind_of(args))
    };
    let low = settings.low.unwrap_or(false);
    let strong = o.strong || (!flag_given("--strong") && settings.strong.unwrap_or(false));
    let mut brain = BrainOptions {
        planner_budget_ms: o.planner_budget_ms,
        seed: args.seed,
        planner_preset: if low {
            ddai_planner::brains::PlannerPreset::Low
        } else {
            ddai_planner::brains::PlannerPreset::Normal
        },
        ..BrainOptions::default()
    };
    // The count is decided once, here: the brain gets the number that is printed.
    let search_threads = o
        .search_threads
        .0
        .unwrap_or_else(ddai_bot::brains::auto_search_threads_here);
    brain.search_threads = Some(search_threads);
    brain.proposal_in_cap = !o.no_proposal_in_cap;
    brain.hybrid_mirror = o.hybrid_mirror;
    brain.hybrid_finish = o.finish.hybrid_drag();
    brain.hybrid_wb_hold = o.finish.hybrid_wb_hold();
    brain.hybrid_budget_ms = knobs.hybrid_budget_ms;
    for line in knobs.start_lines(kind == BrainKind::Hybrid) {
        eprintln!("{line} (task 3.16)");
    }
    let duel_hammer = if flag_given("--duel-hammer") {
        o.duel_hammer.clone()
    } else {
        // A settings value of the wrong kind or an unknown name never keeps the bot from starting nor moves the file aside: warn, run with `off` (3.16 F1).
        match settings.duel_hammer_mode() {
            Ok(Some(m)) if ddai_bot::brains::duel_hammer(&m).is_some() => m,
            Ok(Some(m)) => {
                eprintln!(
                    "duel_hammer {m:?} in the settings file is not off|reflex|reflex-all|envelope|both: ignored, off"
                );
                "off".to_string()
            }
            Ok(None) => o.duel_hammer.clone(),
            Err(why) => {
                eprintln!("{why}: ignored, off");
                "off".to_string()
            }
        }
    };
    brain.reflex = ddai_bot::brains::duel_hammer(&duel_hammer).unwrap_or_default();
    if duel_hammer != "off" {
        eprintln!(
            "duel hammer: {duel_hammer} ({}; D-116): the hybrid's reflex hammer / hammer-safe envelope, only in a detected duel",
            if flag_given("--duel-hammer") {
                "--duel-hammer"
            } else {
                "settings.toml"
            }
        );
    }
    let (duel_fixes, duel_afk) = ddai_bot::brains::duel_fixes(&o.duel_fixes).unwrap_or_default();
    brain.duel_fixes = duel_fixes;
    if o.duel_fixes != "off" {
        eprintln!(
            "duel fixes: {} (--duel-fixes; D-121): the hybrid's answers to a standing opponent, to his hook from above and to a frozen victim off the freeze, only in a detected duel",
            o.duel_fixes
        );
    }
    if o.finish.target_logic() {
        // One line at start (the journal of a launch from the site shows that `--finish` reached the bot; STATUS carries it too).
        eprintln!("finish blocks: {} (--finish; D-097)", o.finish.name());
    }
    if kind == BrainKind::Hybrid {
        eprintln!(
            "hybrid search threads: {search_threads} ({})",
            o.search_threads
                .0
                .map_or("auto: the free cores, at most 4", |_| "--search-threads"),
        );
    }
    if let Some(p) = &o.fly_flyg {
        brain.fly_flyg = p.clone();
    }
    if let Some(p) = &o.fly_bundle {
        brain.fly_bundle = Some(p.clone());
    }
    if let Some(p) = &o.fly_config {
        brain.fly_config = p.clone();
    }
    let bot = BotConfig {
        brain: kind,
        mode,
        fixed_target: o.target.clone(),
        finish: o.finish.target_logic(),
        finish_wb: o.finish.hybrid_wb_hold(),
        duel_afk,
        seed: args.seed,
        clips: ddai_bot::clipper::ClipConfig {
            dir: Some(
                o.clips_dir
                    .clone()
                    .unwrap_or_else(|| data_dir.join("bot").join("clips")),
            ),
            autoclip: !o.no_autoclip,
            async_save: true,
        },
        relations_path: Some(relations_path.clone()),
        settings_path: settings_path.clone(),
        low,
        strong,
        console_names: o.console_names,
        no_selfkill: o.no_selfkill,
        selfkill_policy: o.selfkill_policy,
        duel_detect: !o.no_duel_detect,
        duel_commands: settings.duel_commands.clone().unwrap_or_else(|| {
            ddai_bot::bot::DEFAULT_DUEL_COMMANDS
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        }),
        // An empty value is "off" (the launch unit passes `--window-model=${BOT_WINDOW_MODEL}`, empty when the toggle is off); a relative path is
        // relative to the data directory, not to wherever the unit happens to run.
        window_model: (if flag_given("--window-model") {
            o.window_model.clone()
        } else {
            settings.window_model.clone()
        })
        .filter(|p| !p.as_os_str().is_empty())
        .map(|model| {
            let model = if model.is_relative() {
                data_dir.join(model)
            } else {
                model
            };
            ddai_bot::oppnet::WindowModelConfig::in_data_dir(model, data_dir)
        }),
        selfkill_marker: Some(data_dir.join("bot").join(ddai_bot::selfkill::SELFKILL_OFF_MARKER)),
        preinput: o.preinput,
        preinput_marker: Some(data_dir.join("bot").join(ddai_bot::selfkill::PREINPUT_OFF_MARKER)),
        duel_detect_marker: Some(data_dir.join("bot").join(ddai_bot::selfkill::DUEL_DETECT_OFF_MARKER)),
        kind_estimate: live_timing.kind,
        ..BotConfig::default()
    };
    if let Some(wm) = &bot.window_model {
        if kind == BrainKind::Hybrid {
            eprintln!(
                "window model: {} ({}; task 3.17, D-111): loaded at the start (sha256 in the log), off while the marker {} exists, log {}",
                wm.model.display(),
                if flag_given("--window-model") {
                    "--window-model"
                } else {
                    "settings.toml"
                },
                wm.marker
                    .as_deref()
                    .map_or_else(String::new, |p| p.display().to_string()),
                wm.log.as_deref().map_or_else(String::new, |p| p.display().to_string()),
            );
        } else {
            eprintln!(
                "window model: {} is IGNORED: it belongs to the hybrid brain and this run's brain is {} (task 3.17)",
                wm.model.display(),
                kind.name()
            );
        }
    }
    let wb_text = if flag_given("--wb") {
        o.wb.clone()
    } else {
        settings.wb.clone().unwrap_or_else(|| o.wb.clone())
    };
    let Some(wb_mode) = WbMode::parse(&wb_text) else {
        eprintln!("unknown --wb {wb_text:?} (auto|left|right|off)");
        return ExitCode::FAILURE;
    };
    let nav = NavConfig {
        memory_dir: if o.no_memory {
            None
        } else {
            Some(
                o.memory_dir
                    .clone()
                    .unwrap_or_else(|| data_dir.join("bot").join("memory")),
            )
        },
        wb_mode,
        strong,
        seek: !o.no_seek,
        wb_smart: o.wb_smart,
        ..NavConfig::default()
    };
    if o.wb_smart {
        eprintln!(
            "wb smart: on (--wb-smart; D-103, D-104): more blockable targets, ties random; an AFK player only when in the way; tube crossings: stand still and walk at once after a respawn, search with the other tees, jump over a blocker on the ledge"
        );
    }
    let nav_handle = NavHandle::new();
    if let Some((tx, ty)) = o.goto {
        nav_handle.send(NavCommand::Goto {
            tx,
            ty,
            through_freeze: true,
        });
    }
    if let Some(id) = o.follow {
        nav_handle.follow(id);
    }
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&shutdown);
        if let Err(e) = ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst)) {
            tracing::warn!(error = %e, "failed to install a SIGINT/SIGTERM handler");
        }
    }
    // Windows has no Unix-domain sockets yet (`ddai_os::ipc`): the live-view socket and the web control socket are then off, as if
    // `--no-bridge --no-control` had been given, instead of refusing to start.
    let no_bridge = o.no_bridge || !ddai_os::ipc::SUPPORTED;
    let no_control = o.no_control || !ddai_os::ipc::SUPPORTED;
    if !ddai_os::ipc::SUPPORTED && !(o.no_bridge && o.no_control) {
        println!(
            "this platform has no Unix-domain sockets: playing without the live-view socket and the web control socket"
        );
    }
    let bridge_path = if no_bridge {
        None
    } else {
        Some(
            o.bridge
                .clone()
                .unwrap_or_else(|| data_dir.join("bot").join("live.sock")),
        )
    };
    // The command bus: the console (task 4.3) and the web control socket (task 5.6) both put `BotCommand`s on it and
    // the bot answers between two snapshots. Nothing typed or sent ever reaches the chat.
    let console = o.console || (!o.no_console && ddai_bot::console::stdin_is_terminal());
    let (bus_sender, inbox) = ddai_bot::command::CommandBus::open();
    let mut bus_used = false;
    let console_out = if console {
        let print = ddai_bot::console::stdout_printer();
        match ddai_bot::console::spawn(
            bus_sender.clone(),
            std::io::BufReader::new(std::io::stdin()),
            std::sync::Arc::clone(&print),
        ) {
            Ok(_) => {
                println!("console: type !help for the commands; a line without ! or ? is not sent anywhere");
                bus_used = true;
                Some(print)
            }
            Err(e) => {
                eprintln!("could not start the console: {e}");
                None
            }
        }
    } else {
        None
    };
    // Held until the bot has stopped: dropping it closes and removes the socket.
    let _control = if no_control {
        // No control socket means no chat; the channel is burnt so that nothing else in this process can claim it later.
        ddai_bot::control::forgo_owner_chat();
        None
    } else {
        let socket = o
            .control
            .clone()
            .unwrap_or_else(|| data_dir.join("bot").join(ddai_bot::control::SOCKET_NAME));
        let audit_path = o
            .control_audit
            .clone()
            .unwrap_or_else(|| data_dir.join("logs").join("bot").join("control-audit.log"));
        let audit = match ddai_bot::control::FileAudit::open(&audit_path) {
            Ok(a) => std::sync::Arc::new(a),
            Err(e) => {
                eprintln!(
                    "refusing to start: cannot open the control audit log {}: {e}",
                    audit_path.display()
                );
                return ExitCode::FAILURE;
            }
        };
        // Chat is on unless something switches it off, and it fails closed: the flag, `owner_chat = false`, the marker file
        // `bot/owner-chat.off` (survives `launch apply`), an unreadable settings file, or a settings key the bot does not know.
        let marker = ddai_os::marker::is_present(&data_dir.join("bot").join(ddai_bot::settings::OWNER_CHAT_OFF_MARKER));
        let chat_off = ddai_bot::settings::owner_chat_off(
            o.no_owner_chat,
            &settings,
            settings_unreadable,
            &unknown_settings_keys,
            marker,
        );
        let owner_chat = chat_off.is_none();
        if let Some(why) = &chat_off {
            println!(
                "owner chat: switched off ({}): lines typed on the website are refused",
                why.describe()
            );
        }
        match ddai_bot::control::ControlServer::start_with_owner_chat(&socket, bus_sender.clone(), audit, owner_chat) {
            Ok(server) => {
                bus_used = true;
                Some(server)
            }
            Err(e) => {
                eprintln!("refusing to start: the control socket {}: {e}", socket.display());
                return ExitCode::FAILURE;
            }
        }
    };
    let commands = bus_used.then_some(inbox);
    let cfg = RunnerConfig {
        server,
        client,
        bot,
        brain,
        relations,
        duration: run_for(args.duration),
        bridge_path,
        web_names: o.web_names,
        debug_names_log: o.debug_names.clone(),
        audit_outgoing: o.report.is_some(),
        shutdown,
        nav,
        nav_handle,
        commands,
        console_out,
    };
    // A big stack: the planner's helpers copy whole worlds by value.
    let handle = std::thread::Builder::new()
        .name("ddai-bot".to_string())
        .stack_size(64 << 20)
        .spawn(move || ddai_bot::runner::run(cfg));
    let result = match handle {
        Ok(h) => h.join().unwrap_or_else(|_| {
            Err(RunnerError::Bridge {
                path: PathBuf::new(),
                source: std::io::Error::other("the bot thread panicked"),
            })
        }),
        Err(e) => {
            eprintln!("could not start the bot thread: {e}");
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(report) => {
            println!("{}", summary(&report));
            if let Some(path) = &o.report
                && let Err(e) = std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&report_json(&report)).unwrap_or_default(),
                )
            {
                eprintln!("could not write the report {}: {e}", path.display());
            }
            ExitCode::from(report.exit_code)
        }
        Err(e) => {
            eprintln!("the bot could not start: {e}");
            ExitCode::FAILURE
        }
    }
}

fn summary_of(name: &str, s: ddai_bot::latency::Summary) -> String {
    format!(
        "  {name:<9} n={:<6} p50={:>6} us  p90={:>6} us  p99={:>6} us  max={:>7} us",
        s.count, s.p50_us, s.p90_us, s.p99_us, s.max_us
    )
}

fn summary(r: &RunReport) -> String {
    let l = &r.latency;
    let mut out = format!(
        "bot ran {:.1}s, exit {}, {} decisions ({} collapsed), {} blocks ({} held 5 s, {} of them the victim died, {} escaped), {} blocked-by, {} kills\n",
        r.elapsed.as_secs_f64(),
        r.exit_code,
        r.stats.decisions,
        r.stats.collapsed,
        r.block_stats.blocks,
        r.block_stats.held,
        r.block_stats.died,
        r.block_stats.escaped,
        r.block_stats.blocked_by,
        r.kill_ticks.len()
    );
    out += &summary_of("total", l.total.summary());
    out.push('\n');
    out += &summary_of("brain", l.brain.summary());
    out.push('\n');
    out += &summary_of("overhead", l.overhead.summary());
    out.push('\n');
    out += &summary_of("pick", l.pick.summary());
    out.push('\n');
    out += &summary_of("queue", l.queue.summary());
    out.push('\n');
    out += &summary_of("wire", l.wire.summary());
    out.push('\n');
    out += &summary_of("candidates (count)", l.candidates.summary());
    out.push('\n');
    out += &summary_of("proposal", l.proposal.summary());
    out.push('\n');
    out += &summary_of("search", l.search.summary());
    out.push('\n');
    out += &summary_of("brain (decisions made)", l.brain_made.summary());
    out
}

fn sum_json(s: ddai_bot::latency::Summary) -> serde_json::Value {
    serde_json::json!({"n": s.count, "p50_us": s.p50_us, "p90_us": s.p90_us, "p99_us": s.p99_us, "max_us": s.max_us})
}

/// The JSON report of a run (`--report`). No nicknames: events carry tags only.
pub fn report_json(r: &RunReport) -> serde_json::Value {
    let outgoing: serde_json::Map<String, serde_json::Value> = r
        .outgoing
        .iter()
        .map(|(k, (ok, refused))| (k.clone(), serde_json::json!({"accepted": ok, "refused": refused})))
        .collect();
    serde_json::json!({
        "exit_code": r.exit_code,
        "elapsed_s": r.elapsed.as_secs_f64(),
        "map": r.map_name,
        "gave_up": r.gave_up.as_ref().map(|(reason, cat)| serde_json::json!({"reason": reason, "category": format!("{cat:?}")})),
        "stats": {
            "snapshots": r.stats.snapshots,
            "collapsed": r.stats.collapsed,
            "decisions": r.stats.decisions,
            "brain_decisions": r.stats.brain_decisions,
            "wander_decisions": r.stats.wander_decisions,
            "idle_decisions": r.stats.idle_decisions,
            "hooks_fired": r.stats.hooks_fired,
            "hammer_fires": r.stats.hammer_fires,
            "self_kills": r.stats.self_kills,
            "vetoed_hooks": r.stats.vetoed_hooks,
            "vetoed_fires": r.stats.vetoed_fires,
            "predict_clamped": r.stats.predict_clamped,
            "guarded_inputs": r.stats.guarded_inputs,
            "deaths": r.stats.deaths,
        },
        "blocks": {"blocks": r.block_stats.blocks, "blocked_by": r.block_stats.blocked_by, "held": r.block_stats.held, "died": r.block_stats.died, "escaped": r.block_stats.escaped},
        "kill_ticks": r.kill_ticks,
        "kill_command_ticks": r.kill_command_ticks,
        // The owner's website chat (D-094). `sent` counts the lines handed to the client, which can still drop one if the session left
        // the game in between: the wire count in `outgoing_game_messages["Cl_Say(owner)"]` is at most `sent`.
        "owner_chat": {"accepted": r.owner_chat.accepted, "sent": r.owner_chat.sent, "refused": r.owner_chat.refused, "dropped": r.owner_chat.dropped},
        "outgoing_game_messages": outgoing,
        "latency_us": {
            "total": sum_json(r.latency.total.summary()),
            "brain": sum_json(r.latency.brain.summary()),
            "overhead": sum_json(r.latency.overhead.summary()),
            "pick": sum_json(r.latency.pick.summary()),
            "queue": sum_json(r.latency.queue.summary()),
            "wire": sum_json(r.latency.wire.summary()),
            "slots": {"decisions": r.latency.slots.decisions, "first_slot": r.latency.slots.in_first_slot, "missed_first_slot": r.latency.slots.missed_first_slot, "as_predicted": r.latency.slots.as_predicted, "later_than_predicted": r.latency.slots.later_than_predicted, "earlier_than_predicted": r.latency.slots.earlier_than_predicted},
        },
        // Task 3.11: every series (phases of the decision, the driver's hand-over lag, the horizon) with p95, and the driver's send lag.
        "latency_detail_us": r.latency.json(),
        "send_lag_us": r.margin.as_ref().and_then(|m| m.send_lag_us).map(|v| serde_json::json!({"n": v.count, "p50": v.p50_us, "p90": v.p90_us, "p99": v.p99_us, "max": v.max_us})),
        "brain_detail": {
            "candidates": sum_json(r.latency.candidates.summary()),
            "proposal_us": sum_json(r.latency.proposal.summary()),
            "search_us": sum_json(r.latency.search.summary()),
            "brain_made_us": sum_json(r.latency.brain_made.summary()),
        },
        "input_margin": r.margin.as_ref().map(|m| serde_json::json!({
            "count": m.count, "late": m.late_count, "late_fraction": m.late_fraction, "p50_ms": m.p50_ms, "p99_ms": m.p99_ms, "min_ms": m.min_ms,
            "margin_ms": m.margin_ms, "margin_changes": m.margin_changes, "adaptive": m.adaptive, "margin_stable_ms": m.margin_stable_ms, "margin_changes_last_30s": m.margin_changes_last_30s, "superseded_decisions": m.superseded_decisions, "time_at_cap_ms": m.time_at_margin_ms.last().copied().unwrap_or(0),
        })),
        "events": r.events.iter().map(|e| format!("{e:?}")).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn the_missing_malloc_variable_is_warned_about_once_and_a_set_one_is_not() {
        use std::ffi::OsStr;
        assert!(mmap_threshold_warning(None).is_some_and(|w| w.contains("131072")));
        assert!(mmap_threshold_warning(Some(OsStr::new(""))).is_some());
        assert_eq!(mmap_threshold_warning(Some(OsStr::new("131072"))), None);
    }

    /// F2 (task 4.9): the report carries the owner chat's numbers, which the soak analyser needs to judge `Cl_Say(owner)`.
    #[test]
    fn the_report_json_carries_the_owner_chat_numbers() {
        let report = RunReport {
            exit_code: 0,
            stats: Default::default(),
            latency: Default::default(),
            block_stats: Default::default(),
            outgoing: [("Cl_Say(owner)".to_string(), (2, 0))].into_iter().collect(),
            kill_ticks: Vec::new(),
            kill_command_ticks: Vec::new(),
            owner_chat: ddai_bot::ownerchat::OwnerChatStats {
                accepted: 3,
                sent: 2,
                refused: 1,
                dropped: 1,
            },
            gave_up: None,
            map_name: None,
            events: Vec::new(),
            seal_times: Default::default(),
            reach_times: Default::default(),
            margin: None,
            elapsed: Duration::from_secs(1),
            bridge_clients_dropped: 0,
        };
        let json = report_json(&report);
        assert_eq!(
            json["owner_chat"],
            serde_json::json!({"accepted": 3, "sent": 2, "refused": 1, "dropped": 1})
        );
        assert_eq!(
            json["outgoing_game_messages"]["Cl_Say(owner)"],
            serde_json::json!({"accepted": 2, "refused": 0})
        );
    }

    #[test]
    fn duration_zero_means_no_limit_for_the_bot() {
        assert_eq!(run_for(0), None);
        assert_eq!(run_for(1), Some(Duration::from_secs(1)));
        assert_eq!(run_for(3600), Some(Duration::from_secs(3600)));
    }
}

#[cfg(test)]
mod search_threads_tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        bot: BotOpts,
    }

    #[test]
    fn search_threads_parses_auto_a_number_and_refuses_the_rest() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.search_threads)
        };
        assert_eq!(get(&[]).unwrap(), SearchThreads(Some(1)), "the default is one thread");
        assert_eq!(SearchThreads::default(), SearchThreads(Some(1)));
        assert_eq!(get(&["--search-threads", "auto"]).unwrap(), SearchThreads(None));
        assert_eq!(get(&["--search-threads", "AUTO"]).unwrap(), SearchThreads(None));
        assert_eq!(get(&["--search-threads", "1"]).unwrap(), SearchThreads(Some(1)));
        assert_eq!(get(&["--search-threads", "4"]).unwrap(), SearchThreads(Some(4)));
        for bad in ["0", "17", "-1", "many", ""] {
            assert!(get(&["--search-threads", bad]).is_err(), "{bad:?}");
        }
    }

    /// Task 3.11: the live-timing switches are on by default; `off` and any list narrow them.
    #[test]
    fn live_timing_is_on_by_default_and_off_turns_it_off() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.live_timing)
        };
        let both = LiveTiming {
            precise: true,
            kind: true,
        };
        assert_eq!(get(&[]).unwrap(), both, "on by default");
        assert_eq!(get(&["--live-timing", "off"]).unwrap(), LiveTiming::default());
        assert_eq!(
            get(&["--live-timing", "precise"]).unwrap(),
            LiveTiming {
                precise: true,
                kind: false
            }
        );
        assert_eq!(get(&["--live-timing", "kind, PRECISE"]).unwrap(), both);
        assert!(get(&["--live-timing", "fast"]).is_err());
    }

    /// F4 (task 4.9): the chat-only emergency switch is a flag, off by default.
    #[test]
    fn the_owner_chat_switch_is_a_flag_and_off_by_default() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.no_owner_chat)
        };
        assert!(!get(&[]).unwrap());
        assert!(get(&["--no-owner-chat"]).unwrap());
        assert!(
            get(&["--no-owner-chat", "--no-control"]).unwrap(),
            "it does not conflict with --no-control"
        );
    }

    /// Task 4.11 (D-102): the duel switch is a flag, off by default.
    #[test]
    fn the_no_selfkill_switch_is_a_flag_off_by_default_and_takes_true_or_false_after_an_equals_sign() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.no_selfkill)
        };
        assert!(!get(&[]).unwrap());
        assert!(get(&["--no-selfkill"]).unwrap(), "the bare flag still works");
        // Task 5.15: the value form the launch unit passes (`--no-selfkill=${BOT_NO_SELFKILL}`).
        assert!(get(&["--no-selfkill=true"]).unwrap());
        assert!(!get(&["--no-selfkill=false"]).unwrap());
        assert!(get(&["--no-selfkill=TRUE"]).unwrap());
        // Nothing else is a value: not empty, not on/off, not a word; and a space-separated word is not swallowed as a value.
        for bad in [
            "--no-selfkill=",
            "--no-selfkill=on",
            "--no-selfkill=off",
            "--no-selfkill=1",
            "--no-selfkill=yes",
        ] {
            assert!(get(&[bad]).is_err(), "{bad}");
        }
        assert!(
            get(&["--no-selfkill", "false"]).is_err(),
            "`false` after a space is a stray argument, not a value"
        );
        // The bare flag next to other flags, in either order.
        assert!(get(&["--no-selfkill", "--no-console"]).unwrap());
        assert!(get(&["--no-console", "--no-selfkill"]).unwrap());
        assert!(get(&["--no-console", "--no-selfkill=true", "--strong"]).unwrap());
        // `deploy/install-launcher.sh` greps the help for this exact spelling to refuse a binary older than the unit.
        use clap::CommandFactory;
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--no-selfkill[="), "{help}");
    }

    /// Task 4.12 (D-108): the policy is `legacy` unless asked; only the two words are values.
    #[test]
    fn the_selfkill_policy_is_legacy_by_default_and_takes_legacy_or_smart() {
        use ddai_bot::smartkill::SelfKillPolicy;
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.selfkill_policy)
        };
        assert_eq!(get(&[]).unwrap(), SelfKillPolicy::Legacy);
        assert_eq!(get(&["--selfkill-policy", "smart"]).unwrap(), SelfKillPolicy::Smart);
        assert_eq!(get(&["--selfkill-policy=legacy"]).unwrap(), SelfKillPolicy::Legacy);
        let no_detect = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.no_duel_detect)
        };
        assert!(!no_detect(&[]).unwrap(), "the detection is on by default");
        assert!(no_detect(&["--no-duel-detect"]).unwrap());
        for bad in ["--selfkill-policy=Smart", "--selfkill-policy=on", "--selfkill-policy="] {
            assert!(get(&[bad]).is_err(), "{bad}");
        }
        assert!(get(&["--selfkill-policy"]).is_err());
    }

    /// Task 4.10 (D-100): the timeout code is on by default and `--no-timeout-code` switches it off.
    #[test]
    fn the_timeout_code_switch_is_a_flag_and_off_by_default() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.no_timeout_code)
        };
        assert!(!get(&[]).unwrap());
        assert!(get(&["--no-timeout-code"]).unwrap());
    }

    #[test]
    fn the_window_model_is_off_by_default_and_takes_a_file() {
        let get = |extra: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(extra);
            Cli::try_parse_from(v).map(|c| c.bot.window_model)
        };
        assert_eq!(get(&[]).unwrap(), None);
        assert_eq!(
            get(&["--window-model", "/x/m1.oppnet"]).unwrap(),
            Some(PathBuf::from("/x/m1.oppnet"))
        );
        assert!(get(&["--window-model"]).is_err(), "the flag needs a file");
        // The launch unit's form: `--window-model=` with nothing after it is a valid "off" for the CLI (the bot treats it as no model).
        assert_eq!(get(&["--window-model="]).unwrap(), Some(PathBuf::from("")));
    }

    #[test]
    fn the_preinput_flag_is_off_by_default_and_takes_on_or_off() {
        let get = |extra: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(extra);
            Cli::try_parse_from(v).map(|c| c.bot.preinput)
        };
        assert!(!get(&[]).unwrap());
        assert!(get(&["--preinput", "on"]).unwrap());
        assert!(!get(&["--preinput", "off"]).unwrap());
        assert!(get(&["--preinput", "maybe"]).is_err());
    }

    #[test]
    fn wb_smart_is_off_by_default_and_takes_on_or_off() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.wb_smart)
        };
        assert!(!get(&[]).unwrap(), "the Copy Love Box targeting is opt-in (D-103)");
        assert!(get(&["--wb-smart", "on"]).unwrap());
        assert!(!get(&["--wb-smart", "off"]).unwrap());
        assert!(get(&["--wb-smart", "maybe"]).is_err());
    }

    #[test]
    fn finish_is_off_by_default_and_takes_off_target_wb_or_full() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.finish)
        };
        assert_eq!(get(&[]).unwrap(), FinishMode::Off, "finishing is opt-in");
        assert_eq!(get(&["--finish", "off"]).unwrap(), FinishMode::Off);
        assert_eq!(get(&["--finish", "target"]).unwrap(), FinishMode::Target);
        assert_eq!(get(&["--finish", "FULL"]).unwrap(), FinishMode::Full);
        assert_eq!(get(&["--finish", "wb"]).unwrap(), FinishMode::Wb);
        assert_eq!(
            get(&["--finish", "on"]).unwrap(),
            FinishMode::Full,
            "`on` is the old spelling of `full`"
        );
        assert!(get(&["--finish", "maybe"]).is_err());
        // The switches each mode turns on: the target mode has no drag shaping; `wb` is the target logic plus the wayblock hold (task 3.18), not the drag.
        let on = |m: FinishMode| (m.target_logic(), m.hybrid_drag(), m.hybrid_wb_hold());
        assert_eq!(on(FinishMode::Off), (false, false, false));
        assert_eq!(on(FinishMode::Target), (true, false, false));
        assert_eq!(on(FinishMode::Full), (true, true, false));
        assert_eq!(on(FinishMode::Wb), (true, false, true));
        for m in [FinishMode::Off, FinishMode::Target, FinishMode::Wb, FinishMode::Full] {
            assert_eq!(parse_finish(m.name()).unwrap(), m, "the name round-trips");
        }
    }

    #[test]
    fn duel_hammer_is_off_by_default_and_takes_a_mode() {
        let get = |extra: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(extra);
            Cli::try_parse_from(v).map(|c| c.bot.duel_hammer)
        };
        assert_eq!(get(&[]).unwrap(), "off");
        assert_eq!(get(&["--duel-hammer", "reflex"]).unwrap(), "reflex");
        assert_eq!(get(&["--duel-hammer", "BOTH"]).unwrap(), "both");
        assert!(get(&["--duel-hammer", "sometimes"]).is_err());
        assert!(get(&["--duel-hammer"]).is_err(), "a value is required");
    }

    #[test]
    fn duel_fixes_are_off_by_default_and_take_a_list() {
        let get = |extra: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(extra);
            Cli::try_parse_from(v).map(|c| c.bot.duel_fixes)
        };
        assert_eq!(get(&[]).unwrap(), "off");
        assert_eq!(get(&["--duel-fixes", "static,Counter"]).unwrap(), "static,counter");
        assert_eq!(get(&["--duel-fixes", "ALL"]).unwrap(), "all");
        assert!(get(&["--duel-fixes", "static,sometimes"]).is_err());
        assert!(get(&["--duel-fixes"]).is_err(), "a value is required");
    }

    #[test]
    fn hybrid_mirror_is_on_by_default_and_takes_on_or_off() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| c.bot.hybrid_mirror)
        };
        assert!(get(&[]).unwrap(), "the opponent model is on by default");
        assert!(get(&["--hybrid-mirror", "on"]).unwrap());
        assert!(get(&["--hybrid-mirror", "ON"]).unwrap());
        assert!(!get(&["--hybrid-mirror", "off"]).unwrap());
        assert!(get(&["--hybrid-mirror", "maybe"]).is_err());
        assert!(get(&["--hybrid-mirror"]).is_err(), "a value is required");
    }

    /// Task 3.16 (D-115): the two input-lag flags are absent by default (the old behaviour) and refuse values outside their ranges.
    #[test]
    fn the_input_lag_flags_are_off_by_default_and_checked_against_their_ranges() {
        let get = |args: &[&str]| {
            let mut v = vec!["x"];
            v.extend_from_slice(args);
            Cli::try_parse_from(v).map(|c| (c.bot.hybrid_budget_ms, c.bot.prediction_margin_ms))
        };
        assert_eq!(get(&[]).unwrap(), (None, None));
        assert_eq!(get(&["--hybrid-budget-ms", "1"]).unwrap().0, Some(1));
        assert_eq!(get(&["--hybrid-budget-ms", "8"]).unwrap().0, Some(8));
        for bad in ["0", "9", "-1", "2.5", "x"] {
            assert!(get(&["--hybrid-budget-ms", bad]).is_err(), "--hybrid-budget-ms {bad}");
        }
        assert_eq!(get(&["--prediction-margin-ms", "0"]).unwrap().1, Some(0));
        assert_eq!(get(&["--prediction-margin-ms", "30"]).unwrap().1, Some(30));
        for bad in ["31", "-1", "x"] {
            assert!(
                get(&["--prediction-margin-ms", bad]).is_err(),
                "--prediction-margin-ms {bad}"
            );
        }
        assert!(get(&["--prediction-margin-ms=-1"]).is_err());
    }
}
