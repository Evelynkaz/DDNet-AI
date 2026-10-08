//! Task 3.15 (E-028): a game recorded as the dataset record of the opponent-input predictor (`ddai_oppnet::data::GameRec`).
//!
//! For every world tick up to the first freeze (which decides a 1v1 game) the record holds the snapshot-observable state of both tees ([`TeeFrame`]), the inputs they
//! applied in the step that led to the tick, and the geometry rays around slot 1 (the modelled opponent); see `ddai_oppnet::data` for the indexing. The recorder only
//! reads the game: the brains decide exactly as without it.

use ddai_oppnet::data::{GameRec, TickRec};
use ddai_oppnet::frame::{InputRec, N_RAYS, TeeFrame, rays};

use crate::EnvError;
use crate::arena::Arena;
use crate::config::Rules;
use crate::game::{GameReport, Layout, play_game_watched};
use crate::observe;
use crate::sim::PlayerSetup;

/// Plays one two-player game (slot 0 = us, slot 1 = the opponent to model) and returns its record with the usual report.
/// The game stops at the first freeze, so the report's result is a timeout; the record is what counts.
pub fn record_game(
    arena: &Arena,
    rules: &Rules,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
) -> Result<(GameRec, GameReport), EnvError> {
    if players.len() != 2 {
        return Err(EnvError::new("the recorder needs exactly two players"));
    }
    let mut rec = GameRec {
        arena: arena.name.clone(),
        seed,
        lag: [players[0].lag.min(255) as u8, players[1].lag.min(255) as u8],
        swap: layout.swap,
        decide_every: rules.decide_every.clamp(1, 255) as u8,
        tick0: 0,
        ticks: Vec::new(),
    };
    let report = play_game_watched(arena, rules, seed, layout, players, &mut |sim, tick| {
        let w = sim.pw.inner();
        if rec.ticks.is_empty() {
            rec.tick0 = tick;
        }
        let frames = [
            TeeFrame::from_world(w, 0, 1).unwrap_or_default(),
            TeeFrame::from_world(w, 1, 0).unwrap_or_default(),
        ];
        let input = |id: u8| {
            w.cores
                .get(id)
                .map_or_else(InputRec::default, |c| InputRec::from_wire(&c.input))
        };
        let mut ray = [1.0f32; N_RAYS];
        if frames[1].alive {
            rays(w, frames[1].pos, &mut ray);
        }
        rec.ticks.push(TickRec {
            frames,
            applied: [input(0), input(1)],
            rays: ray,
        });
        // The first freeze decides a 1v1 game: nothing after it is the opponent playing on.
        !(observe::is_out(w, 0) || observe::is_out(w, 1))
    })?;
    Ok((rec, report))
}

/// Task 3.21 (E-036): [`record_game`] in the v2 format (`ddai_oppnet::v2::data`): the rays around **both** tees.
pub fn record_game_v2(
    arena: &Arena,
    rules: &Rules,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
) -> Result<(ddai_oppnet::v2::data::GameRec, GameReport), EnvError> {
    use ddai_oppnet::v2::data::{GameRec as GameRec2, TickRec as TickRec2};
    if players.len() != 2 {
        return Err(EnvError::new("the recorder needs exactly two players"));
    }
    let mut rec = GameRec2 {
        arena: arena.name.clone(),
        seed,
        lag: [players[0].lag.min(255) as u8, players[1].lag.min(255) as u8],
        swap: layout.swap,
        decide_every: rules.decide_every.clamp(1, 255) as u8,
        tick0: 0,
        ticks: Vec::new(),
    };
    let report = play_game_watched(arena, rules, seed, layout, players, &mut |sim, tick| {
        let w = sim.pw.inner();
        if rec.ticks.is_empty() {
            rec.tick0 = tick;
        }
        let frames = [
            TeeFrame::from_world(w, 0, 1).unwrap_or_default(),
            TeeFrame::from_world(w, 1, 0).unwrap_or_default(),
        ];
        let input = |id: u8| {
            w.cores
                .get(id)
                .map_or_else(InputRec::default, |c| InputRec::from_wire(&c.input))
        };
        let mut ray = [[1.0f32; N_RAYS]; 2];
        for (r, f) in ray.iter_mut().zip(&frames) {
            if f.alive {
                rays(w, f.pos, r);
            }
        }
        rec.ticks.push(TickRec2 {
            frames,
            applied: [input(0), input(1)],
            rays: ray,
        });
        !(observe::is_out(w, 0) || observe::is_out(w, 1))
    })?;
    Ok((rec, report))
}
