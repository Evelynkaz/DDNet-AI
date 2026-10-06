//! The held-block outcome of a game and the reward made of it (task 8.5a).
//!
//! The arena's first-freeze rule ends a game at the first onset, so everything the fly does *after* freezing the
//! opponent used to go unmeasured and unrewarded. A game is played on for [`WINDOW_TICKS`] after the deciding freeze
//! (`Rules::after_ticks`, which `ddai_env::game::play_game_watched` already supports) and a per-tick observer records who is
//! out. From that this module derives the facts of the *held block*: the victim never got free during the window and the
//! focal player was never out. Task 3.10 adds the same facts to `GameReport` (`held_block`, `escape_tick`,
//! `victim_out_ticks`); this is the additive stand-in, and [`EpisodeOutcome::from_records`] is the one place to swap.
//!
//! # Reward
//! Base reward (what the fly is *for*) per episode, see [`RewardConfig`]:
//! * a held block: `+1` (the victim stayed out for the whole window, the focal player was never out);
//! * the focal player out (frozen or dead) at any time in the window: `-1` (an own freeze is the worst outcome);
//! * a block that did not hold: `0` (it froze and let go: not a block);
//! * full games additionally: credited first freeze `+0.3` (and `+0.7` more when it holds), lost `-1`, drawn and timed out `-0.5`.
//!
//! Shaping is **potential-based** (Ng et al. 1999): `F = gamma * Phi(s') - Phi(s)` per tick, `Phi = 0` at a terminal state (the
//! focal player out). The potential is the *hold margin* of the victim ([`hold_potential`]): `1` for a dead victim, up to `0.5` for
//! a frozen one by the share of `sv_freeze_delay` it still has to run, `0` for a free one. What this does and does not guarantee:
//! * Over an episode that **ends** in a terminal state the sum telescopes to `-Phi(first)` (a constant), so it cannot change which
//!   policy is best, whatever `Phi` is.
//! * The 250-tick window, though, **cuts** the episode off; it does not terminate it. If the sum pays `Phi` of the state the window
//!   ends in ([`Cutoff::Pay`], what the E-022 pilot did), it is *not* policy-invariant: it rewards ending the window with a dead
//!   or deeply frozen victim (up to `+0.3` at weight `0.3`) and costs nothing but a thaw-and-refreeze in the last ticks (review of
//!   8.5a, F5). The bonus is small and points the right way, but "leaves the best policy unchanged" is false for it.
//! * The invariant options are [`Cutoff::NoPayout`] (the cutoff counts as a terminal state with `Phi = 0`: the sum is constant)
//!   and [`Cutoff::Bootstrap`] (replace `Phi(s_T)` by a value estimate `V(s_T)` in potential units, what a PPO learner does with its critic).
//!   [`EpisodeOutcome::shaped_return`] computes the sum for any `gamma` and cutoff from the stored potential trace.
//!
//! The base reward is paid once per victim life (one episode = one window after one freeze), and `freeze -> thaw -> freeze` is a
//! zero-sum walk of the potential, so there is nothing to farm.

use ddai_brain::Brain;
use ddai_env::EnvError;
use ddai_env::arena::Arena;
use ddai_env::config::Rules;
use ddai_env::game::{GameReport, Layout, play_game_watched};
use ddai_env::observe;
use ddai_env::sim::PlayerSetup;
use ddai_env::stats::GameResult;
use serde::{Deserialize, Serialize};

/// The window after the deciding freeze: 250 ticks = 5 s, longer than `sv_freeze_delay` (3 s); the same number as
/// task 3.10's `HELD_BLOCK_TICKS`.
pub const WINDOW_TICKS: i32 = 250;

/// `sv_freeze_delay` of the arenas, in ticks (3 s at 50 Hz): the longest a freeze lasts once the tee leaves the freeze tile.
pub const FREEZE_TICKS: f32 = 150.0;

/// What the arena observer saw at the end of one world tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickRecord {
    pub tick: i32,
    /// Out (dead or frozen) after this tick, by slot (the first two slots).
    pub out: [bool; 2],
    /// [`hold_potential`] of slot 1 (the victim in a duel) after this tick.
    pub phi: f32,
}

/// The hold margin of `victim` in `[0, 1]`: `1` when it is dead (it never comes back), `0` when it is free, and between
/// `0` and `0.5` while frozen, by the freeze time it has left (a tee standing on freeze keeps the timer full). A *potential*,
/// not a reward: it only ever enters as a difference.
pub fn hold_potential(world: &ddai_physics::world::World<f32>, victim: i32) -> f32 {
    match observe::character_observation(world, victim) {
        None => 1.0,
        Some(c) if c.freeze_ticks_remaining > 0 => 0.5 * (c.freeze_ticks_remaining as f32 / FREEZE_TICKS).min(1.0),
        Some(_) => 0.0,
    }
}

/// Everything one episode (a game played on through the window) says about the held block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeOutcome {
    pub result: GameResult,
    pub credited: bool,
    pub end_tick: i32,
    pub victim: i32,
    /// Ticks of the window actually played (`WINDOW_TICKS` unless the game stopped early).
    pub window_played: i32,
    /// Ticks of the window the victim was out.
    pub victim_out_ticks: i32,
    /// The first tick after the deciding one on which the victim was free.
    pub escape_tick: Option<i32>,
    /// The focal player was out (frozen or dead) at some tick of the window.
    pub focal_out_in_window: bool,
    /// A won game whose victim never got free during the whole window while the focal player was never out. It also needs the freeze to be
    /// **credited** to the focal player (D-059): an opponent that froze itself into a pit is not our block.
    pub held_block: bool,
    /// `Phi` at the deciding tick and at the last tick of the window (or `0` when the focal player went out: terminal).
    pub phi_start: f32,
    pub phi_end: f32,
    /// The sum of the per-tick shaping terms `Phi(s') - Phi(s)` over the window (`gamma = 1`, [`Cutoff::Pay`]: the E-022 pilot's).
    pub shaping: f32,
    /// The potential at the deciding tick and at every tick of the window up to (excluding) the one the focal player went out on.
    pub phis: Vec<f32>,
    /// The focal player went out: the trace ends in a terminal state (`Phi = 0`) and not in a cutoff.
    pub terminal: bool,
}

/// What the shaping does at the end of a window that was cut off (not terminated); see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Cutoff {
    /// `Phi(s_T)` of the last state is paid (the E-022 pilot); not policy-invariant.
    Pay,
    /// The cutoff is a terminal state with `Phi = 0`: the sum is the constant `-Phi(first)`.
    NoPayout,
    /// `Phi(s_T)` is replaced by this value estimate (in potential units).
    Bootstrap(f32),
}

impl EpisodeOutcome {
    /// Derives the outcome from the game's report and the per-tick records of the observer.
    pub fn from_records(report: &GameReport, records: &[TickRecord], window: i32) -> EpisodeOutcome {
        let end = report.end_tick;
        let won = report.result == GameResult::W;
        // The window: the ticks after the deciding one.
        let win: Vec<&TickRecord> = records
            .iter()
            .filter(|r| r.tick > end && r.tick <= end + window)
            .collect();
        let victim_slot = usize::try_from(report.victim).ok().filter(|&v| v < 2);
        let mut victim_out_ticks = 0;
        let mut escape_tick = None;
        let mut focal_out = false;
        let phi_start = records.iter().find(|r| r.tick == end).map_or(0.0, |r| r.phi);
        let (mut shaping, mut prev, mut phi_end) = (0.0f32, phi_start, phi_start);
        let mut phis = vec![phi_start];
        for r in &win {
            // The victim keeps being tracked after the focal player goes out (`escape_tick` and `victim_out_ticks` are about the
            // victim); only the shaping stops there, at the terminal state.
            if let Some(v) = victim_slot {
                if r.out[v] {
                    victim_out_ticks += 1;
                } else {
                    escape_tick.get_or_insert(r.tick);
                }
            }
            if r.out[0] && victim_slot != Some(0) && !focal_out {
                focal_out = true;
                // Terminal state: the potential of what follows is 0.
                shaping += 0.0 - prev;
                phi_end = 0.0;
            }
            if !focal_out {
                shaping += r.phi - prev;
                prev = r.phi;
                phi_end = r.phi;
                phis.push(r.phi);
            }
        }
        let window_played = win.len() as i32;
        let held_block = won
            && report.credited
            && victim_slot == Some(1)
            && !focal_out
            && window_played == window
            && victim_out_ticks == window;
        EpisodeOutcome {
            result: report.result,
            credited: report.credited,
            end_tick: end,
            victim: report.victim,
            window_played,
            victim_out_ticks,
            escape_tick,
            focal_out_in_window: focal_out,
            held_block,
            phi_start,
            phi_end,
            shaping,
            phis,
            terminal: focal_out,
        }
    }

    /// `sum_t gamma^t (gamma * Phi(s_{t+1}) - Phi(s_t))` over the episode; a terminal ends with `Phi = 0`, and `cutoff` says what stands for
    /// `Phi(s_T)` of the state a cut-off window ends in.
    pub fn shaped_return(&self, gamma: f32, cutoff: Cutoff) -> f32 {
        let mut trace: Vec<f32> = self.phis.clone();
        if self.terminal {
            trace.push(0.0);
        } else if let Some(last) = trace.last_mut() {
            // The last recorded state is the state the window ended in, `s_T`.
            match cutoff {
                Cutoff::Pay => {}
                Cutoff::NoPayout => *last = 0.0,
                Cutoff::Bootstrap(v) => *last = v,
            }
        }
        let (mut sum, mut disc) = (0.0f32, 1.0f32);
        for w in trace.windows(2) {
            sum += disc * (gamma * w[1] - w[0]);
            disc *= gamma;
        }
        sum
    }

    /// `Phi(last) - Phi(first)`: what [`EpisodeOutcome::shaping`] must equal if the shaping telescopes.
    pub fn shaping_closed_form(&self) -> f32 {
        self.phi_end - self.phi_start
    }
}

/// Weights of the reward (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardConfig {
    /// A block that held for the whole window.
    pub held: f32,
    /// The focal player out (frozen or dead) during the window, or lost a game.
    pub self_freeze: f32,
    /// Full games: a credited first freeze (a held one adds `held - credited`).
    pub credited: f32,
    /// Full games: a draw or a timeout.
    pub draw_or_timeout: f32,
    /// Weight of the potential-based shaping (`0` = none).
    pub shaping: f32,
    /// Discount of the shaping sum, `F = gamma * Phi(s') - Phi(s)` (1 = the E-022 pilot).
    pub shaping_gamma: f32,
    /// What the shaping does at the window's cutoff (`pay` = the E-022 pilot; `none` = a terminal `Phi = 0`: no payout). A value
    /// bootstrap needs a critic and is asked of [`EpisodeOutcome::shaped_return`] directly.
    pub shaping_cutoff: CutoffKind,
}

/// The cutoff choices a config can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CutoffKind {
    Pay,
    None,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            held: 1.0,
            self_freeze: 1.0,
            credited: 0.3,
            draw_or_timeout: 0.5,
            shaping: 0.3,
            shaping_gamma: 1.0,
            shaping_cutoff: CutoffKind::Pay,
        }
    }
}

impl RewardConfig {
    fn shaped(&self, o: &EpisodeOutcome) -> f32 {
        let cutoff = match self.shaping_cutoff {
            CutoffKind::Pay => Cutoff::Pay,
            CutoffKind::None => Cutoff::NoPayout,
        };
        o.shaped_return(self.shaping_gamma, cutoff)
    }

    /// The return of a post-freeze start: the freeze is already credited, only what follows is rewarded.
    pub fn post_freeze(&self, o: &EpisodeOutcome) -> f32 {
        let base = if o.focal_out_in_window {
            -self.self_freeze
        } else if o.held_block {
            self.held
        } else {
            0.0
        };
        base + self.shaping * self.shaped(o)
    }

    /// The return of a full game from its first freeze: a credited freeze pays a little, a held one the rest.
    pub fn full_game(&self, o: &EpisodeOutcome) -> f32 {
        let base = match o.result {
            GameResult::W if o.focal_out_in_window => -self.self_freeze,
            GameResult::W if o.credited && o.held_block => self.held,
            GameResult::W if o.credited => self.credited,
            // The opponent froze itself: no credit, no plus (an idle brain collects those too).
            GameResult::W => 0.0,
            GameResult::L => -self.self_freeze,
            GameResult::D | GameResult::T => -self.draw_or_timeout,
        };
        let shaped = if o.result == GameResult::W && o.credited {
            self.shaping * self.shaped(o)
        } else {
            0.0
        };
        base + shaped
    }
}

/// One game on `arena` played through the window with a per-tick record. `rules.after_ticks` is set to `window`.
pub fn play_episode(
    arena: &Arena,
    rules: &Rules,
    seed: u64,
    layout: Layout,
    focal: Box<dyn Brain>,
    opponent: Box<dyn Brain>,
    window: i32,
) -> Result<EpisodeOutcome, EnvError> {
    let rules = Rules {
        after_ticks: window,
        ..rules.clone()
    };
    let players = vec![
        PlayerSetup {
            brain: focal,
            lag: 0,
            label: "focal".into(),
        },
        PlayerSetup {
            brain: opponent,
            lag: 0,
            label: "opponent".into(),
        },
    ];
    let mut records: Vec<TickRecord> = Vec::with_capacity((rules.max_ticks + window) as usize);
    let report = play_game_watched(arena, &rules, seed, layout, players, &mut |sim, tick| {
        let w = sim.pw.inner();
        records.push(TickRecord {
            tick,
            out: [observe::is_out(w, sim.ids[0]), observe::is_out(w, sim.ids[1])],
            phi: hold_potential(w, sim.ids[1]),
        });
        true
    })?;
    Ok(EpisodeOutcome::from_records(&report, &records, window))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(result: GameResult, end: i32, victim: i32, credited: bool) -> GameReport {
        // Only the fields `from_records` reads matter; build one through the public constructor path of a real game.
        let mut r = tiny_report();
        r.result = result;
        r.end_tick = end;
        r.victim = victim;
        r.credited = credited;
        r
    }

    fn tiny_report() -> GameReport {
        use ddai_env::arena::{Arena, ArenaDef};
        let def = ArenaDef::parse(include_str!("../../../configs/arenas/pit.toml")).unwrap();
        let arena = Arena::build(&def, std::path::Path::new("/nonexistent")).unwrap();
        let rules = Rules {
            max_ticks: 10,
            after_ticks: 0,
            ..Rules::default()
        };
        let mk = || PlayerSetup {
            brain: Box::new(ddai_brain::IdleBrain),
            lag: 0,
            label: "idle".into(),
        };
        ddai_env::game::play_game(&arena, &rules, 1, Layout::default(), vec![mk(), mk()]).unwrap()
    }

    fn rec(tick: i32, out0: bool, out1: bool, phi: f32) -> TickRecord {
        TickRecord {
            tick,
            out: [out0, out1],
            phi,
        }
    }

    #[test]
    fn a_held_block_needs_the_victim_out_for_the_whole_window_and_the_focal_player_never_out() {
        let r = report(GameResult::W, 10, 1, true);
        let all_out: Vec<TickRecord> = (10..=14).map(|t| rec(t, false, true, 0.4)).collect();
        let o = EpisodeOutcome::from_records(&r, &all_out, 4);
        assert!(o.held_block && o.victim_out_ticks == 4 && o.escape_tick.is_none());
        // The victim gets free on the last tick of the window: not held.
        let mut escaped = all_out.clone();
        escaped[4].out[1] = false;
        let o = EpisodeOutcome::from_records(&r, &escaped, 4);
        assert!(!o.held_block && o.escape_tick == Some(14) && o.victim_out_ticks == 3);
        // The focal player freezes itself on the second tick: not held, and the shaping stops there (terminal).
        let mut selfie = all_out.clone();
        selfie[2].out[0] = true;
        let o = EpisodeOutcome::from_records(&r, &selfie, 4);
        assert!(!o.held_block && o.focal_out_in_window && o.phi_end == 0.0);
        // A win that was not credited to the focal player (the opponent froze itself) is never a held block.
        let uncredited = report(GameResult::W, 10, 1, false);
        let o = EpisodeOutcome::from_records(&uncredited, &all_out, 4);
        assert!(!o.held_block && o.victim_out_ticks == 4);
        // The game ended before the window did (cut short): not held either.
        let o = EpisodeOutcome::from_records(&r, &all_out[..3], 4);
        assert!(!o.held_block);
    }

    #[test]
    fn shaping_telescopes_to_the_difference_of_the_end_potentials() {
        let r = report(GameResult::W, 100, 1, true);
        // An arbitrary wandering potential, never terminal.
        let phis = [0.5, 0.3, 0.45, 0.1, 0.0, 0.25, 0.5, 0.35];
        let recs: Vec<TickRecord> = phis
            .iter()
            .enumerate()
            .map(|(i, &p)| rec(100 + i as i32, false, true, p))
            .collect();
        let o = EpisodeOutcome::from_records(&r, &recs, 7);
        assert!(
            (o.shaping - o.shaping_closed_form()).abs() < 1e-6,
            "{} vs {}",
            o.shaping,
            o.shaping_closed_form()
        );
        assert!((o.shaping - (0.35 - 0.5)).abs() < 1e-6);
        // Ending in a terminal state (the focal player out): the sum is `0 - Phi(first)` however it got there.
        let mut dead = recs.clone();
        dead[3].out[0] = true;
        let o = EpisodeOutcome::from_records(&r, &dead, 7);
        assert!((o.shaping - o.shaping_closed_form()).abs() < 1e-6);
        assert!((o.shaping + 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_victim_is_tracked_after_the_focal_player_goes_out() {
        let r = report(GameResult::W, 10, 1, true);
        // The focal player goes out on the second tick; the victim gets free on the fourth.
        let mut recs: Vec<TickRecord> = (10..=14).map(|t| rec(t, false, true, 0.4)).collect();
        recs[2].out[0] = true;
        recs[3].out[0] = true;
        recs[4].out[0] = true;
        recs[4].out[1] = false;
        let o = EpisodeOutcome::from_records(&r, &recs, 4);
        assert!(
            o.focal_out_in_window && !o.held_block,
            "focal out makes the strict held false"
        );
        assert_eq!(o.escape_tick, Some(14), "the escape is still seen");
        assert_eq!(o.victim_out_ticks, 3);
        assert!(o.terminal && o.phi_end == 0.0);
    }

    #[test]
    fn the_shaped_return_is_gamma_correct_and_the_cutoff_options_are_what_the_docs_say() {
        let r = report(GameResult::W, 100, 1, true);
        let phis = [0.5, 0.3, 0.45, 0.1, 0.0, 0.25];
        let recs: Vec<TickRecord> = phis
            .iter()
            .enumerate()
            .map(|(i, &p)| rec(100 + i as i32, false, true, p))
            .collect();
        let o = EpisodeOutcome::from_records(&r, &recs, 5);
        // gamma = 1 with the potential paid at the cutoff is the pilot's sum.
        assert!((o.shaped_return(1.0, Cutoff::Pay) - o.shaping).abs() < 1e-6);
        // Bootstrapping with the last potential is the same thing.
        assert!((o.shaped_return(0.9, Cutoff::Bootstrap(0.25)) - o.shaped_return(0.9, Cutoff::Pay)).abs() < 1e-6);
        // No payout: a terminal with Phi = 0, so the sum is `-Phi(first)` whatever the path and gamma (policy-invariant).
        for g in [1.0, 0.99, 0.9] {
            assert!((o.shaped_return(g, Cutoff::NoPayout) + 0.5).abs() < 1e-5, "gamma {g}");
            let mut other = recs.clone();
            other[3].phi = 0.5;
            let o2 = EpisodeOutcome::from_records(&r, &other, 5);
            assert!((o2.shaped_return(g, Cutoff::NoPayout) + 0.5).abs() < 1e-5);
        }
        // Paying the potential at the cutoff does depend on how the window ends.
        let mut dead_end = recs.clone();
        dead_end[5].phi = 1.0;
        let o3 = EpisodeOutcome::from_records(&r, &dead_end, 5);
        assert!(o3.shaped_return(1.0, Cutoff::Pay) > o.shaped_return(1.0, Cutoff::Pay) + 0.7);
        // A hand-computed discounted sum over the potentials 0.5, 0.3, and the state the window ends in (0.2 instead of 0.4 from the
        // critic): (0.5*0.3 - 0.5) + 0.5 * (0.5*0.2 - 0.3) = -0.35 - 0.1
        let short = recs[..3].to_vec();
        let o4 = EpisodeOutcome::from_records(&r, &short, 2);
        assert_eq!(o4.phis, vec![0.5, 0.3, 0.45]);
        assert!((o4.shaped_return(0.5, Cutoff::Bootstrap(0.2)) + 0.45).abs() < 1e-6);
    }

    #[test]
    fn freeze_thaw_freeze_is_not_a_way_to_earn_reward() {
        // Two victims' potential paths with the same endpoints: one holds, one thaws and is frozen again. The shaping
        // part of the return is equal (it only sees the endpoints); the thaw costs the base reward (not held).
        let r = report(GameResult::W, 0, 1, true);
        let hold: Vec<TickRecord> = (0..=4).map(|t| rec(t, false, true, 0.5)).collect();
        let farm: Vec<TickRecord> = [0.5, 0.2, 0.0, 0.3, 0.5]
            .iter()
            .enumerate()
            .map(|(t, &p)| rec(t as i32, false, p > 0.0, p))
            .collect();
        let cfg = RewardConfig::default();
        let (h, f) = (
            EpisodeOutcome::from_records(&r, &hold, 4),
            EpisodeOutcome::from_records(&r, &farm, 4),
        );
        assert!((h.shaping - f.shaping).abs() < 1e-6);
        assert!(cfg.post_freeze(&h) > cfg.post_freeze(&f) + 0.9);
        assert!(f.escape_tick.is_some() && !f.held_block);
    }

    #[test]
    fn rewards_follow_the_module_docs() {
        let cfg = RewardConfig {
            shaping: 0.0,
            ..RewardConfig::default()
        };
        let base = |result, credited, held, focal_out| EpisodeOutcome {
            result,
            credited,
            end_tick: 0,
            victim: 1,
            window_played: 250,
            victim_out_ticks: 250,
            escape_tick: None,
            focal_out_in_window: focal_out,
            held_block: held,
            phi_start: 0.0,
            phi_end: 0.0,
            shaping: 0.0,
            phis: vec![0.0],
            terminal: focal_out,
        };
        assert_eq!(cfg.post_freeze(&base(GameResult::W, true, true, false)), 1.0);
        assert_eq!(cfg.post_freeze(&base(GameResult::W, true, false, false)), 0.0);
        assert_eq!(cfg.post_freeze(&base(GameResult::W, true, false, true)), -1.0);
        assert_eq!(cfg.full_game(&base(GameResult::W, true, true, false)), 1.0);
        assert_eq!(cfg.full_game(&base(GameResult::W, true, false, false)), 0.3);
        assert_eq!(cfg.full_game(&base(GameResult::W, false, false, false)), 0.0);
        assert_eq!(cfg.full_game(&base(GameResult::L, false, false, false)), -1.0);
        assert_eq!(cfg.full_game(&base(GameResult::T, false, false, false)), -0.5);
    }
}
