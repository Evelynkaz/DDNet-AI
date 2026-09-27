//! `random-v1`: a deterministic generator for "interesting" scenarios — inputs designed to
//! exercise movement, jumping (including in-air double-jump attempts), hooking (ground and
//! player-vs-player), and aiming, rather than realistic human play.
//!
//! Determinism contract: `random_v1(params)` called twice with the same [`Params`] produces
//! byte-identical [`Scenario`]s (and therefore identical `write_bytes()` and sha256). This holds
//! because the only source of "randomness" is [`crate::prng::SplitMix64`] seeded from
//! `params.seed`, consumed in a fixed order that depends only on `params` — never on `HashMap`
//! iteration, wall-clock time, or thread scheduling.

use crate::hash::sha256;
use crate::prng::SplitMix64;
use crate::rawmap;
use crate::scenario::{CharacterSpawn, MapRef, Scenario, ScenarioInput};
use crate::synthetic;
use ddai_physics::map::{MapData, TILE_AIR};
use std::f64::consts::PI;

/// Parameters for [`random_v1`]. `recipe`/`seed`/`ticks`/`characters` — exactly the knobs the
/// task spec calls for; nothing else varies scenario bytes for this generator (world flags and
/// tuning always take `random-v1`'s fixed defaults: `no_weak_hook = false`, default tuning).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub seed: u64,
    pub ticks: u32,
    pub characters: u32,
}

/// Why [`random_v1`] refused to build a scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratorError {
    UnknownRecipe(String),
    /// `characters` was `0` or greater than 4.
    CharacterCountOutOfRange {
        characters: u32,
    },
    /// The map doesn't have enough free (open, in both game and front layers) cells to place
    /// every character without overlapping.
    NotEnoughFreeCells {
        needed: u32,
        available: usize,
    },
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneratorError::UnknownRecipe(name) => write!(f, "unknown recipe '{name}'"),
            GeneratorError::CharacterCountOutOfRange { characters } => {
                write!(f, "character count {characters} out of range (must be 1..=4)")
            }
            GeneratorError::NotEnoughFreeCells { needed, available } => {
                write!(
                    f,
                    "need {needed} free cells to spawn every character, map only has {available}"
                )
            }
        }
    }
}

impl std::error::Error for GeneratorError {}

const TILE_SIZE: i32 = 32;
const TEE_HALF_SIZE: i32 = TILE_SIZE / 2; // spawns land on tile centers, well inside a 28px tee box

/// Builds a deterministic scenario for `recipe` (a [`synthetic::RECIPES`] name).
pub fn random_v1(recipe: &str, params: Params) -> Result<Scenario, GeneratorError> {
    if !(1..=4).contains(&params.characters) {
        return Err(GeneratorError::CharacterCountOutOfRange {
            characters: params.characters,
        });
    }
    let map = synthetic::build(recipe).ok_or_else(|| GeneratorError::UnknownRecipe(recipe.to_string()))?;
    let map_sha256 = sha256(&rawmap::write(&map));

    let mut rng = SplitMix64::new(params.seed);
    let free_cells = free_spawn_cells(&map);
    let spawns = place_characters(&mut rng, &free_cells, params.characters)?;

    let characters: Vec<CharacterSpawn> = spawns
        .iter()
        .enumerate()
        .map(|(i, &(tx, ty))| CharacterSpawn {
            id: i as u32,
            spawn_x: tile_to_px(tx),
            spawn_y: tile_to_px(ty),
        })
        .collect();

    let mut gens: Vec<CharacterGen> = (0..characters.len()).map(|_| CharacterGen::new()).collect();
    // Each character's fixed "buddy" to aim at (the next character, wrapping around) — `-1` if
    // there's no one else. Only the *slot index* is needed now (not a position): `aim_slot`
    // resolves against whatever position that slot is actually at on a given tick (see
    // `resolve_input`), not a value baked in here at generation time (review round 1, finding F3).
    let buddy_slot: Vec<i32> = if characters.len() > 1 {
        (0..characters.len())
            .map(|i| ((i + 1) % characters.len()) as i32)
            .collect()
    } else {
        vec![-1]
    };

    let mut inputs = Vec::with_capacity(params.ticks as usize);
    for _ in 0..params.ticks {
        let mut tick_inputs = Vec::with_capacity(characters.len());
        for (i, character_gen) in gens.iter_mut().enumerate() {
            tick_inputs.push(character_gen.tick(&mut rng, buddy_slot[i]));
        }
        inputs.push(tick_inputs);
    }

    Ok(Scenario {
        map_ref: MapRef::Recipe {
            name: recipe.to_string(),
        },
        map_sha256,
        no_weak_hook: false,
        tuning_overrides: Vec::new(),
        characters,
        inputs,
    })
}

fn tile_to_px(tile: i32) -> i32 {
    tile * TILE_SIZE + TEE_HALF_SIZE
}

/// Tile coordinates, in row-major order, whose game (and front, if present) tile is `TILE_AIR` —
/// i.e. where a spawned tee's 28x28 box (which fits well within one free 32x32 tile) is
/// guaranteed clear.
fn free_spawn_cells(map: &MapData) -> Vec<(i32, i32)> {
    let w = map.width as i32;
    let h = map.height as i32;
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let game_open = map.game[i].index == TILE_AIR;
            let front_open = map.front.as_ref().map(|f| f[i].index == TILE_AIR).unwrap_or(true);
            if game_open && front_open {
                out.push((x, y));
            }
        }
    }
    out
}

/// Picks one free cell per character, in `free_cells`' order for reproducibility. About half the
/// time, a character after the first spawns within Chebyshev distance 1..=3 tiles of a
/// previously-placed character (so player-vs-player collision and hooking actually happen);
/// otherwise it spawns at an independently chosen free cell. Never places two characters on the
/// same cell.
fn place_characters(
    rng: &mut SplitMix64,
    free_cells: &[(i32, i32)],
    count: u32,
) -> Result<Vec<(i32, i32)>, GeneratorError> {
    if free_cells.len() < count as usize {
        return Err(GeneratorError::NotEnoughFreeCells {
            needed: count,
            available: free_cells.len(),
        });
    }
    let mut placed: Vec<(i32, i32)> = Vec::with_capacity(count as usize);
    for i in 0..count {
        let cell = if i == 0 || !rng.chance(1, 2) {
            pick_unused(rng, free_cells, &placed)
        } else {
            let anchor = *rng.pick(&placed);
            pick_near(rng, free_cells, &placed, anchor).unwrap_or_else(|| pick_unused(rng, free_cells, &placed))
        };
        placed.push(cell);
    }
    Ok(placed)
}

fn pick_unused(rng: &mut SplitMix64, free_cells: &[(i32, i32)], placed: &[(i32, i32)]) -> (i32, i32) {
    let available: Vec<(i32, i32)> = free_cells.iter().copied().filter(|c| !placed.contains(c)).collect();
    *rng.pick(&available)
}

/// Tries a handful of random offsets within Chebyshev distance 1..=3 of `anchor` that land on an
/// unused free cell; `None` if none of the tries land on one.
fn pick_near(
    rng: &mut SplitMix64,
    free_cells: &[(i32, i32)],
    placed: &[(i32, i32)],
    anchor: (i32, i32),
) -> Option<(i32, i32)> {
    for _ in 0..20 {
        let dx = rng.range_inclusive(-3, 3);
        let dy = rng.range_inclusive(-3, 3);
        if dx == 0 && dy == 0 {
            continue; // not "near", it's the same cell
        }
        let candidate = (anchor.0 + dx, anchor.1 + dy);
        if free_cells.contains(&candidate) && !placed.contains(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Per-character input state machine driving [`random_v1`]'s "interesting" input stream.
struct CharacterGen {
    dir: i32,
    dir_left: u32,
    jump_on: bool,
    jump_left: u32,
    jump_chain: u32,
    hook_on: bool,
    hook_left: u32,
    /// `-1` (explicit target, in `aim_target_or_noise`) or a character slot to aim at live
    /// (`aim_target_or_noise` is then small integer noise added on top) — see `resolve_input`.
    aim_slot: i32,
    aim_target_or_noise: (i32, i32),
    aim_left: u32,
    fire_state: i32,
    fire_left: u32,
    wanted_weapon: i32,
}

impl CharacterGen {
    fn new() -> Self {
        CharacterGen {
            dir: 0,
            dir_left: 0,
            jump_on: false,
            jump_left: 0,
            jump_chain: 0,
            hook_on: false,
            hook_left: 0,
            aim_slot: -1,
            aim_target_or_noise: (1000, 0),
            aim_left: 0,
            fire_state: 0,
            fire_left: 0,
            wanted_weapon: 0,
        }
    }

    fn tick(&mut self, rng: &mut SplitMix64, buddy_slot: i32) -> ScenarioInput {
        self.tick_direction(rng);
        self.tick_jump(rng);
        self.tick_aim(rng, buddy_slot);
        self.tick_hook(rng);
        self.tick_fire(rng);
        self.tick_weapon(rng);

        ScenarioInput {
            direction: self.dir,
            target_x: self.aim_target_or_noise.0,
            target_y: self.aim_target_or_noise.1,
            aim_slot: self.aim_slot,
            jump: self.jump_on as i32,
            fire: self.fire_state,
            hook: self.hook_on as i32,
            player_flags: 0,
            wanted_weapon: self.wanted_weapon,
            next_weapon: 0,
            prev_weapon: 0,
        }
    }

    fn tick_direction(&mut self, rng: &mut SplitMix64) {
        if self.dir_left > 0 {
            self.dir_left -= 1;
        } else {
            self.dir = *rng.pick(&[-1, 0, 1]);
            self.dir_left = hold_ticks(rng, 5, 40);
        }
    }

    /// Presses/releases the jump key, occasionally chaining a second short press shortly after
    /// releasing the first — an in-air double-jump *attempt* (the generator doesn't simulate
    /// physics, so it cannot guarantee the character really is airborne; it produces the same
    /// input *shape* a player double-jumping would).
    fn tick_jump(&mut self, rng: &mut SplitMix64) {
        if self.jump_left > 0 {
            self.jump_left -= 1;
            return;
        }
        if self.jump_on {
            self.jump_on = false;
            if self.jump_chain > 0 {
                self.jump_chain -= 1;
                self.jump_left = hold_ticks(rng, 2, 5);
            } else {
                self.jump_left = hold_ticks(rng, 5, 30);
            }
        } else {
            self.jump_on = true;
            self.jump_left = hold_ticks(rng, 1, 3);
            if self.jump_chain == 0 && rng.chance(2, 5) {
                self.jump_chain = 1;
            }
        }
    }

    fn tick_aim(&mut self, rng: &mut SplitMix64, buddy_slot: i32) {
        if self.aim_left > 0 {
            self.aim_left -= 1;
        } else {
            self.redraw_aim(rng, buddy_slot);
            self.aim_left = hold_ticks(rng, 10, 50);
        }
        // Occasional aim change mid-hook, independent of the periodic timer above. `aim_slot`
        // itself (who to track) is much stickier than this — only the noise/explicit-target
        // redraw happens this often; live tracking means the *resolved* target already moves
        // every tick without needing a redraw at all (see `resolve_input`).
        if self.hook_on && rng.chance(1, 10) {
            self.redraw_aim(rng, buddy_slot);
        }
    }

    /// Sets `aim_slot`/`aim_target_or_noise` for the *next* stretch of ticks: with a buddy
    /// available, 2/3 of the time aim live at them (small integer noise on top — see
    /// `resolve_input`); otherwise (or with no buddy) fall back to a random-angle explicit
    /// target of fixed magnitude 1000, exactly as `random-v1` v1 always did.
    fn redraw_aim(&mut self, rng: &mut SplitMix64, buddy_slot: i32) {
        if buddy_slot >= 0 && rng.chance(2, 3) {
            self.aim_slot = buddy_slot;
            let mut nx = rng.range_inclusive(-30, 30);
            let ny = rng.range_inclusive(-30, 30);
            if nx == 0 && ny == 0 {
                nx = 1; // guarantee the (0, 0)-fallback path in `resolve_input` always has a
                // nonzero value to fall back to, per DDNet's "never aim exactly at the center"
                // rule (see docs/formats.md).
            }
            self.aim_target_or_noise = (nx, ny);
            return;
        }
        self.aim_slot = -1;
        let angle_deg = rng.range_inclusive(0, 359) as f64;
        let rad = angle_deg * PI / 180.0;
        // Fixed magnitude 1000 (an arbitrary "far away" cursor distance, matching a real
        // client's typical target vector magnitude) guarantees (target_x, target_y) != (0, 0).
        self.aim_target_or_noise = ((1000.0 * rad.cos()).round() as i32, (1000.0 * rad.sin()).round() as i32);
    }

    /// Holds the hook 1..=60 ticks when engaged (per the original task spec), with a shorter
    /// idle gap between attempts — **except** about 15% of holds run 61..=120 ticks instead.
    /// That tail deliberately exceeds the spec's literal "1-60" wording: review round 1 (finding
    /// F2) found that a hold capped at 60 means releasing the `hook` *input* (which immediately
    /// un-grabs a hooked player, `gamecore.cpp`'s `if(m_Input.m_Hook) {...} else { ... HOOK_IDLE
    /// ... }`) always happens before the core's own internal 60-tick auto-release timeout
    /// (`HookTick > SERVER_TICK_SPEED + SERVER_TICK_SPEED/5` while `HOOK_GRABBED` on a player,
    /// `gamecore.cpp` ~454-457) can ever fire — making that branch permanently dead in every
    /// trace this generator could ever produce. The orchestrator's explicit fix for F2 was this
    /// tail; see `docs/formats.md` §4 for the same note in the format spec.
    fn tick_hook(&mut self, rng: &mut SplitMix64) {
        if self.hook_left > 0 {
            self.hook_left -= 1;
            return;
        }
        if self.hook_on {
            self.hook_on = false;
            self.hook_left = hold_ticks(rng, 3, 30);
        } else {
            self.hook_on = true;
            self.hook_left = if rng.chance(3, 20) {
                hold_ticks(rng, 61, 120)
            } else {
                hold_ticks(rng, 1, 60)
            };
        }
    }

    /// `fire` is a press-counter (odd = pressed), matching the real `CNetObj_PlayerInput`
    /// convention — inert for Oracle A (the core-only tick never reads it), kept realistic for
    /// forward compatibility with a weapon-handling oracle.
    fn tick_fire(&mut self, rng: &mut SplitMix64) {
        if self.fire_left > 0 {
            self.fire_left -= 1;
            return;
        }
        self.fire_state += 1;
        self.fire_left = if self.fire_state % 2 == 1 {
            hold_ticks(rng, 1, 10)
        } else {
            hold_ticks(rng, 5, 40)
        };
    }

    fn tick_weapon(&mut self, rng: &mut SplitMix64) {
        if rng.chance(1, 20) {
            self.wanted_weapon = rng.below(6) as i32;
        }
    }
}

/// Draws a hold duration in `[lo, hi]` (inclusive) ticks *including the current tick*, and
/// returns how many more ticks the caller's `_left` countdown should hold after this one (i.e.
/// `duration - 1`) — the tick that draws a new duration always immediately applies the new state
/// for that same tick, so the countdown must not count it twice.
fn hold_ticks(rng: &mut SplitMix64, lo: i32, hi: i32) -> u32 {
    rng.range_inclusive(lo, hi) as u32 - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::resolve_input;

    fn params() -> Params {
        Params {
            seed: 1,
            ticks: 200,
            characters: 3,
        }
    }

    #[test]
    fn same_seed_and_params_give_identical_bytes() {
        let a = random_v1("arena", params()).unwrap();
        let b = random_v1("arena", params()).unwrap();
        assert_eq!(a.write_bytes(), b.write_bytes());
    }

    #[test]
    fn different_seed_gives_different_bytes() {
        let a = random_v1("arena", params()).unwrap();
        let mut p = params();
        p.seed = 2;
        let b = random_v1("arena", p).unwrap();
        assert_ne!(a.write_bytes(), b.write_bytes());
    }

    #[test]
    fn rejects_unknown_recipe() {
        assert_eq!(
            random_v1("not-a-recipe", params()),
            Err(GeneratorError::UnknownRecipe("not-a-recipe".to_string()))
        );
    }

    #[test]
    fn rejects_zero_characters() {
        let mut p = params();
        p.characters = 0;
        assert_eq!(
            random_v1("arena", p),
            Err(GeneratorError::CharacterCountOutOfRange { characters: 0 })
        );
    }

    #[test]
    fn rejects_more_than_four_characters() {
        let mut p = params();
        p.characters = 5;
        assert_eq!(
            random_v1("arena", p),
            Err(GeneratorError::CharacterCountOutOfRange { characters: 5 })
        );
    }

    #[test]
    fn accepts_one_character() {
        let mut p = params();
        p.characters = 1;
        let s = random_v1("arena", p).unwrap();
        assert_eq!(s.characters.len(), 1);
        // With no buddy, target must still never be (0, 0).
        for tick in &s.inputs {
            assert_ne!((tick[0].target_x, tick[0].target_y), (0, 0));
        }
    }

    #[test]
    fn stored_target_or_noise_is_never_the_origin_for_any_character_or_tick() {
        // `(target_x, target_y)` is the *stored* field — an explicit target when `aim_slot ==
        // -1`, or noise when `aim_slot >= 0` (see `ScenarioInput`) — never the *resolved* input
        // `resolve_input` would apply. It must never be `(0, 0)` regardless of mode: in explicit
        // mode it's used directly; in aim-slot mode it's exactly what `resolve_input` falls back
        // to if the live vector-plus-noise happens to cancel out to `(0, 0)`, so it must be a
        // valid target on its own too.
        for recipe in synthetic::RECIPES {
            let s = random_v1(
                recipe,
                Params {
                    seed: 7,
                    ticks: 300,
                    characters: 4,
                },
            )
            .unwrap();
            for tick in &s.inputs {
                for input in tick {
                    assert_ne!((input.target_x, input.target_y), (0, 0), "recipe {recipe}");
                }
            }
        }
    }

    #[test]
    fn resolved_target_is_never_the_origin_even_with_live_aim_slot_tracking() {
        // End-to-end check of the actual applied input (via `resolve_input`), not just the
        // stored field: feeds every tick's inputs through `resolve_input` with a few different
        // (fake, since this crate doesn't simulate physics) position sequences, including one
        // where the aimed-at character sits exactly on top of `self`.
        let s = random_v1(
            "arena",
            Params {
                seed: 13,
                ticks: 1000,
                characters: 3,
            },
        )
        .unwrap();
        let spawns = s.spawn_positions();
        let position_sequences: [Vec<(i32, i32)>; 2] = [
            spawns.clone(),                              // nobody ever moves
            spawns.iter().map(|_| (500, 500)).collect(), // everyone stacked on the same point
        ];
        for prev_positions in &position_sequences {
            for tick in &s.inputs {
                for (slot, input) in tick.iter().enumerate() {
                    let resolved = resolve_input(input, slot, prev_positions);
                    assert_ne!((resolved.target_x, resolved.target_y), (0, 0), "slot {slot}");
                }
            }
        }
    }

    #[test]
    fn spawns_are_distinct_and_within_map_bounds() {
        let map = synthetic::build("arena").unwrap();
        let s = random_v1(
            "arena",
            Params {
                seed: 3,
                ticks: 1,
                characters: 4,
            },
        )
        .unwrap();
        let mut positions = Vec::new();
        for c in &s.characters {
            assert!(c.spawn_x >= 0 && c.spawn_x < map.width as i32 * TILE_SIZE);
            assert!(c.spawn_y >= 0 && c.spawn_y < map.height as i32 * TILE_SIZE);
            positions.push((c.spawn_x, c.spawn_y));
        }
        let mut sorted = positions.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), positions.len(), "spawn positions must be distinct");
    }

    fn hook_hold_runs(s: &Scenario, slot: usize) -> Vec<u32> {
        let mut run = 0u32;
        let mut runs = Vec::new();
        for tick in &s.inputs {
            if tick[slot].hook != 0 {
                run += 1;
            } else if run > 0 {
                runs.push(run);
                run = 0;
            }
        }
        if run > 0 {
            runs.push(run);
        }
        runs
    }

    #[test]
    fn hook_holds_are_within_one_to_one_twenty_ticks() {
        // 1..=60 is the original task spec's range; 61..=120 is the review-round-1 tail (finding
        // F2) that lets the core's own >60-tick auto-release timeout actually fire — see
        // `tick_hook`'s doc comment.
        let s = random_v1(
            "arena",
            Params {
                seed: 11,
                ticks: 2000,
                characters: 2,
            },
        )
        .unwrap();
        for slot in 0..2 {
            let runs = hook_hold_runs(&s, slot);
            assert!(!runs.is_empty(), "expected at least one hook hold");
            for r in runs {
                assert!((1..=120).contains(&r), "hook hold {r} outside 1..=120");
            }
        }
    }

    #[test]
    fn some_hook_holds_exceed_sixty_ticks() {
        // Proves the F2 tail actually fires (not just that it's syntactically reachable): over
        // enough ticks/seeds, some fraction of holds must land in 61..=120.
        let mut over_60 = 0;
        let mut total = 0;
        for seed in 0..20u64 {
            let s = random_v1(
                "arena",
                Params {
                    seed,
                    ticks: 3000,
                    characters: 3,
                },
            )
            .unwrap();
            for slot in 0..3 {
                for r in hook_hold_runs(&s, slot) {
                    total += 1;
                    if r > 60 {
                        over_60 += 1;
                    }
                }
            }
        }
        assert!(total > 50, "expected plenty of hook holds to sample from, got {total}");
        assert!(
            over_60 > 0,
            "expected at least one hook hold over 60 ticks across {total} holds, got 0"
        );
    }

    #[test]
    fn jump_produces_press_and_release_edges() {
        let s = random_v1(
            "arena",
            Params {
                seed: 5,
                ticks: 500,
                characters: 1,
            },
        )
        .unwrap();
        let jumps: Vec<i32> = s.inputs.iter().map(|t| t[0].jump).collect();
        let presses = jumps.windows(2).filter(|w| w[0] == 0 && w[1] == 1).count();
        let releases = jumps.windows(2).filter(|w| w[0] == 1 && w[1] == 0).count();
        assert!(presses > 0 && releases > 0);
    }

    #[test]
    fn errors_when_character_count_exceeds_free_cells() {
        // Build a tiny map with fewer than 4 free cells directly, bypassing synthetic recipes,
        // by exercising place_characters' error path through a recipe would require a new
        // fixture map; instead check the underlying helper directly.
        let free_cells = [(1, 1), (2, 2)];
        let mut rng = SplitMix64::new(0);
        let err = place_characters(&mut rng, &free_cells, 3).unwrap_err();
        assert_eq!(
            err,
            GeneratorError::NotEnoughFreeCells {
                needed: 3,
                available: 2
            }
        );
    }

    #[test]
    fn place_characters_never_collides() {
        let map = synthetic::build("freeze").unwrap();
        let free_cells = free_spawn_cells(&map);
        let mut rng = SplitMix64::new(99);
        let placed = place_characters(&mut rng, &free_cells, 4).unwrap();
        let mut sorted = placed.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), placed.len());
    }
}
