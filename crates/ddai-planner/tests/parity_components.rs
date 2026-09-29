//! Acceptance criterion 4: dedicated parity checks for `scripted::scripted_action`,
//! `seal::{rests_in_freeze, touches_freeze, sealed_in}`, `throw_lines::{throw_lines,
//! frozen_throw_lines}`, `shield::{escape_exists, safer_input}` and the `hazard_field`/
//! `unfreeze_field` BFS grids, against dumps from `tools/ts-trace/gen-component-dump.mjs`
//! (separate from the full-planner dumps, which already exercise all of these indirectly).
//!
//! Review round 1, F5: run this on a real map (`synthetic:*` has almost no freeze tiles adjacent
//! to standable ground, so `escapeExists`/`sealedIn`/`restsInFreeze` would get ~no meaningful
//! cases) -- the generator biases a majority of its shield/seal cases toward freeze-adjacent
//! tiles, so both `escapeExists` verdicts actually occur (measured 171/400 = 42.75%
//! `escapeExists=false` on Copy Love Box, 131/400 = 32.75% on BlmapChill).
//!
//! `#[ignore]`d: needs the `ts-parity` feature and a locally-generated dump file (one file per
//! run -- run once per map to cover more than one). Run:
//! ```text
//! node tools/ts-trace/gen-component-dump.mjs --map "$HOME/aiddnet/data/maps/copy-love-box/Copy Love Box_....map" \
//!   --seed 1 --cases 400 --out ~/aiddnet/data/traces/planner-components/clb.jsonl
//! DDAI_COMPONENT_DUMP=~/aiddnet/data/traces/planner-components/clb.jsonl \
//!   cargo test -p ddai-planner --features ts-parity --release --test parity_components -- --ignored --nocapture
//! ```

#![cfg(feature = "ts-parity")]

use ddai_jsmath::Rng;
use ddai_planner::fields::{hazard_field, unfreeze_field};
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::scripted::scripted_action;
use ddai_planner::seal::{rests_in_freeze, sealed_in, touches_freeze};
use ddai_planner::shield::{escape_exists, safer_input};
use ddai_planner::throw_lines::{frozen_throw_lines, throw_lines};
use ddai_planner::types::TeeState;
use ddai_planner::vmath::Vec2;
use serde::Deserialize;
use std::collections::HashMap;

fn bits_to_f64(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex f64 bits"))
}

#[derive(Deserialize)]
struct TeeJson {
    id: i32,
    alive: bool,
    #[serde(rename = "posX")]
    pos_x: String,
    #[serde(rename = "posY")]
    pos_y: String,
    #[serde(rename = "velX")]
    vel_x: String,
    #[serde(rename = "velY")]
    vel_y: String,
    #[serde(rename = "hookState")]
    hook_state: i32,
    #[serde(rename = "hookPosX")]
    hook_pos_x: String,
    #[serde(rename = "hookPosY")]
    hook_pos_y: String,
    #[serde(rename = "hookDirX")]
    hook_dir_x: String,
    #[serde(rename = "hookDirY")]
    hook_dir_y: String,
    #[serde(rename = "hookedPlayer")]
    hooked_player: i32,
    jumped: i32,
    #[serde(rename = "jumpsLeft")]
    jumps_left: i32,
    direction: i32,
    angle: String,
    #[serde(rename = "activeWeapon")]
    active_weapon: i32,
    frozen: bool,
    #[serde(rename = "freezeTicksLeft")]
    freeze_ticks_left: i64,
    #[serde(rename = "attackTick")]
    attack_tick: i64,
}

impl TeeJson {
    fn to_tee_state(&self) -> TeeState {
        TeeState {
            id: self.id,
            alive: self.alive,
            pos: Vec2 {
                x: bits_to_f64(&self.pos_x),
                y: bits_to_f64(&self.pos_y),
            },
            vel: Vec2 {
                x: bits_to_f64(&self.vel_x),
                y: bits_to_f64(&self.vel_y),
            },
            hook_state: self.hook_state,
            hook_pos: Vec2 {
                x: bits_to_f64(&self.hook_pos_x),
                y: bits_to_f64(&self.hook_pos_y),
            },
            hook_dir: Vec2 {
                x: bits_to_f64(&self.hook_dir_x),
                y: bits_to_f64(&self.hook_dir_y),
            },
            hooked_player: self.hooked_player,
            jumped: self.jumped,
            jumps_left: self.jumps_left,
            direction: self.direction,
            angle: bits_to_f64(&self.angle),
            active_weapon: self.active_weapon,
            frozen: self.frozen,
            freeze_ticks_left: self.freeze_ticks_left,
            attack_tick: self.attack_tick,
            hook_tick: None,
            jumped_total: None,
            reload_ticks: None,
            frozen_for: None,
            deep_frozen: None,
            jumps: None,
            ddnet_flags: None,
            since_attack: None,
        }
    }
}

#[derive(Deserialize)]
struct InputJson {
    direction: i32,
    #[serde(rename = "targetX")]
    target_x: String,
    #[serde(rename = "targetY")]
    target_y: String,
    jump: i32,
    fire: i32,
    hook: i32,
    #[serde(rename = "wantedWeapon")]
    wanted_weapon: i32,
}

impl InputJson {
    fn to_input(&self) -> ddai_planner::types::PlayerInput {
        ddai_planner::types::PlayerInput {
            direction: self.direction,
            target_x: bits_to_f64(&self.target_x),
            target_y: bits_to_f64(&self.target_y),
            jump: self.jump,
            fire: self.fire,
            hook: self.hook,
            player_flags: 0,
            wanted_weapon: self.wanted_weapon,
            next_weapon: 0,
            prev_weapon: 0,
        }
    }
}

#[derive(Deserialize)]
struct RngStateJson {
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    #[serde(rename = "haveSpare")]
    have_spare: bool,
    spare: String,
}

#[derive(Deserialize)]
struct PlanStepJson {
    dir: i32,
    jump: i32,
    hook: i32,
    fire: i32,
    aim: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
// Test-only deserialization enum, one value per JSON line -- not a hot path, so the size gap
// between variants (`Meta` vs the data-heavy `Case`/`Decision`) is not worth boxing fields over.
#[allow(clippy::large_enum_variant)]
enum Line {
    Meta {
        #[serde(rename = "mapPath")]
        map_path: String,
        #[serde(rename = "mapSha256")]
        map_sha256: String,
    },
    Throwlines {
        steps: i32,
        at: String,
        lines: Vec<Vec<PlanStepJson>>,
    },
    Frozenthrowlines {
        steps: i32,
        at: String,
        lines: Vec<Vec<PlanStepJson>>,
    },
    Scripted {
        #[serde(rename = "self")]
        self_tee: TeeJson,
        enemy: TeeJson,
        prev: InputJson,
        #[serde(rename = "rngBefore")]
        rng_before: RngStateJson,
        out: InputJson,
    },
    Seal {
        #[serde(rename = "posX")]
        pos_x: String,
        #[serde(rename = "posY")]
        pos_y: String,
        #[serde(rename = "velX")]
        vel_x: String,
        #[serde(rename = "velY")]
        vel_y: String,
        rests: i32,
    },
    /// Review round 1, F5: the full `hazardField`/`unfreezeField` BFS grids, dumped once.
    Fields {
        width: i32,
        height: i32,
        #[serde(rename = "hazardDist")]
        hazard_dist: Vec<i32>,
        #[serde(rename = "unfreezeDist")]
        unfreeze_dist: Vec<i32>,
    },
    /// Review round 1, F5: `seal::touches_freeze` at freeze-adjacent and plain standable tiles.
    Touchesfreeze { x: String, y: String, touches: bool },
    /// Review round 1, F5: `shield::escape_exists`/`safer_input` + `seal::sealed_in`, biased
    /// toward freeze-adjacent tiles so both verdicts of `escapeExists` actually occur (the
    /// dedicated dump measured 171/400 = 42.75% `escapeExists=false` on Copy Love Box, vs. ~0.1%
    /// `shielded=true` in the full-planner corpus -- see BUILD REPORT).
    Shield {
        #[serde(rename = "posX")]
        pos_x: String,
        #[serde(rename = "posY")]
        pos_y: String,
        #[serde(rename = "velX")]
        vel_x: String,
        #[serde(rename = "velY")]
        vel_y: String,
        frozen: bool,
        #[serde(rename = "freezeTicksLeft")]
        freeze_ticks_left: i64,
        #[serde(rename = "enemyX")]
        enemy_x: String,
        #[serde(rename = "enemyY")]
        enemy_y: String,
        input: InputJson,
        #[serde(rename = "holdTicks")]
        hold_ticks: i32,
        #[serde(rename = "escapeExists")]
        escape_exists: bool,
        #[serde(rename = "saferInput")]
        safer_input: Option<InputJson>,
        #[serde(rename = "sealedIn")]
        sealed_in: bool,
    },
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[test]
#[ignore = "needs the ts-parity feature and a locally-generated dump, see this file's doc comment"]
fn scripted_seal_and_throwlines_match_ts() {
    let path =
        std::env::var("DDAI_COMPONENT_DUMP").expect("set DDAI_COMPONENT_DUMP to a gen-component-dump.mjs output file");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    let meta_line = lines.next().expect("empty dump");
    let Line::Meta { map_path, map_sha256 } = serde_json::from_str(meta_line).expect("bad meta line") else {
        panic!("first line must be meta");
    };
    let bytes = std::fs::read(&map_path).unwrap_or_else(|e| panic!("reading map {map_path}: {e}"));
    let actual_sha = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(&bytes);
        hex_encode(&h.finalize())
    };
    assert_eq!(actual_sha, map_sha256, "map changed since the dump was generated");
    let loaded = ddai_tsworld::load_map_bytes(&bytes).unwrap_or_else(|e| panic!("loading map: {e:?}"));
    let collision = loaded.collision;

    let mut world = ddai_tsworld::SimWorld::new(
        collision.clone(),
        ddai_tsworld::world::SimWorldOptions {
            respawn_delay_ticks: Some(0),
            infinite_ammo: Some(true),
            sv_hit: Some(true),
            all_weapons: None,
            no_weak_hook: None,
        },
    );
    <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut world, 0, Vec2 { x: 0.0, y: 0.0 });
    <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut world, 1, Vec2 { x: 0.0, y: 0.0 });

    let (mut throw_n, mut scripted_n, mut seal_n) = (0usize, 0usize, 0usize);
    let (mut throw_mismatch, mut scripted_mismatch, mut seal_mismatch) = (0usize, 0usize, 0usize);
    let (mut fields_n, mut fields_mismatch) = (0usize, 0usize);
    let (mut touches_n, mut touches_mismatch) = (0usize, 0usize);
    let (mut shield_n, mut shield_mismatch, mut shield_false_n) = (0usize, 0usize, 0usize);

    let mut opp_rng: Option<Rng> = None;

    for line in lines {
        let parsed: Line = serde_json::from_str(line).unwrap_or_else(|e| panic!("bad line: {e}\n{line}"));
        match parsed {
            Line::Meta { .. } => panic!("unexpected second meta line"),
            Line::Throwlines { steps, at, lines: want } => {
                throw_n += 1;
                let got = throw_lines(steps, bits_to_f64(&at));
                if !plans_match(&got, &want) {
                    throw_mismatch += 1;
                    eprintln!("throwLines mismatch at steps={steps}");
                }
            }
            Line::Frozenthrowlines { steps, at, lines: want } => {
                throw_n += 1;
                let got = frozen_throw_lines(steps, bits_to_f64(&at));
                if !plans_match(&got, &want) {
                    throw_mismatch += 1;
                    eprintln!("frozenThrowLines mismatch at steps={steps}");
                }
            }
            Line::Scripted {
                self_tee,
                enemy,
                prev,
                rng_before,
                out,
            } => {
                scripted_n += 1;
                let mut rng = Rng::from_state(
                    rng_before.s0,
                    rng_before.s1,
                    rng_before.s2,
                    rng_before.s3,
                    rng_before.have_spare,
                    bits_to_f64(&rng_before.spare),
                );
                let self_state = self_tee.to_tee_state();
                let enemy_state = enemy.to_tee_state();
                <ddai_tsworld::SimWorld as PlanWorld>::apply_tee_state(&mut world, 0, &self_state);
                <ddai_tsworld::SimWorld as PlanWorld>::apply_tee_state(&mut world, 1, &enemy_state);
                let got = scripted_action(&world, 0, 1, &prev.to_input(), &mut rng);
                let want = out.to_input();
                let ok = got.direction == want.direction
                    && got.target_x.to_bits() == want.target_x.to_bits()
                    && got.target_y.to_bits() == want.target_y.to_bits()
                    && got.jump == want.jump
                    && got.fire == want.fire
                    && got.hook == want.hook
                    && got.wanted_weapon == want.wanted_weapon;
                if !ok {
                    scripted_mismatch += 1;
                    eprintln!(
                        "scriptedAction mismatch: got {got:?} want direction={} target=({:?},{:?})",
                        want.direction, want.target_x, want.target_y
                    );
                }
                opp_rng = Some(rng); // keep the crate's own Rng type exercised end to end
            }
            Line::Seal {
                pos_x,
                pos_y,
                vel_x,
                vel_y,
                rests,
            } => {
                seal_n += 1;
                let pos = Vec2 {
                    x: bits_to_f64(&pos_x),
                    y: bits_to_f64(&pos_y),
                };
                let vel = Vec2 {
                    x: bits_to_f64(&vel_x),
                    y: bits_to_f64(&vel_y),
                };
                let got = rests_in_freeze(&collision, pos, vel);
                if got != rests {
                    seal_mismatch += 1;
                    eprintln!("restsInFreeze mismatch: got {got} want {rests} at {pos:?}");
                }
            }
            // Review round 1, F5.
            Line::Fields {
                width,
                height,
                hazard_dist,
                unfreeze_dist,
            } => {
                fields_n += 1;
                let got_hazard = hazard_field(&collision);
                let got_unfreeze = unfreeze_field(&collision);
                let ok = got_hazard.width == width
                    && got_hazard.height == height
                    && got_hazard.dist == hazard_dist
                    && got_unfreeze.width == width
                    && got_unfreeze.height == height
                    && got_unfreeze.dist == unfreeze_dist;
                if !ok {
                    fields_mismatch += 1;
                    let hd = got_hazard.dist.iter().zip(&hazard_dist).position(|(a, b)| a != b);
                    let ud = got_unfreeze.dist.iter().zip(&unfreeze_dist).position(|(a, b)| a != b);
                    eprintln!(
                        "hazard/unfreeze field mismatch: dims got=({},{}) want=({width},{height}) first hazard diff idx={hd:?} first unfreeze diff idx={ud:?}",
                        got_hazard.width, got_hazard.height
                    );
                }
            }
            Line::Touchesfreeze { x, y, touches } => {
                touches_n += 1;
                let got = touches_freeze(&collision, bits_to_f64(&x), bits_to_f64(&y));
                if got != touches {
                    touches_mismatch += 1;
                    eprintln!("touchesFreeze mismatch: got {got} want {touches} at ({x},{y})");
                }
            }
            Line::Shield {
                pos_x,
                pos_y,
                vel_x,
                vel_y,
                frozen,
                freeze_ticks_left,
                enemy_x,
                enemy_y,
                input,
                hold_ticks,
                escape_exists: want_escape,
                safer_input: want_safer,
                sealed_in: want_sealed,
            } => {
                shield_n += 1;
                if !want_escape {
                    shield_false_n += 1;
                }
                let pos = Vec2 {
                    x: bits_to_f64(&pos_x),
                    y: bits_to_f64(&pos_y),
                };
                let vel = Vec2 {
                    x: bits_to_f64(&vel_x),
                    y: bits_to_f64(&vel_y),
                };
                let enemy_pos = Vec2 {
                    x: bits_to_f64(&enemy_x),
                    y: bits_to_f64(&enemy_y),
                };
                let held = input.to_input();

                // escapeExists/saferInput: self (id 0) + enemy (id 1) present, matching the dump
                // generator's own fresh-`SimWorld`-per-case construction.
                let mut w = ddai_tsworld::SimWorld::new(
                    collision.clone(),
                    ddai_tsworld::world::SimWorldOptions {
                        respawn_delay_ticks: Some(0),
                        infinite_ammo: Some(true),
                        sv_hit: Some(true),
                        all_weapons: None,
                        no_weak_hook: None,
                    },
                );
                <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut w, 0, pos);
                <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut w, 1, enemy_pos);
                let base0 = <ddai_tsworld::SimWorld as PlanWorld>::get_tee(&w, 0).unwrap();
                <ddai_tsworld::SimWorld as PlanWorld>::apply_tee_state(
                    &mut w,
                    0,
                    &TeeState {
                        pos,
                        vel,
                        frozen: false,
                        freeze_ticks_left: 0,
                        ..base0
                    },
                );
                let others: HashMap<i32, ddai_planner::types::PlayerInput> =
                    HashMap::from([(1, ddai_planner::types::empty_input())]);
                let got_escape = escape_exists(&mut w, 0, &held, hold_ticks, &others);
                let got_safer = safer_input(&mut w, 0, &held, hold_ticks, &others, None);

                let input_eq = |a: &ddai_planner::types::PlayerInput, b: &ddai_planner::types::PlayerInput| {
                    a.direction == b.direction
                        && a.target_x.to_bits() == b.target_x.to_bits()
                        && a.target_y.to_bits() == b.target_y.to_bits()
                        && a.jump == b.jump
                        && a.fire == b.fire
                        && a.hook == b.hook
                        && a.wanted_weapon == b.wanted_weapon
                };
                let safer_ok = match (&got_safer, &want_safer) {
                    (None, None) => true,
                    (Some(g), Some(w)) => input_eq(g, &w.to_input()),
                    _ => false,
                };

                // sealedIn: its own fresh world (strips every other tee) -- same self state.
                let mut w2 = ddai_tsworld::SimWorld::new(
                    collision.clone(),
                    ddai_tsworld::world::SimWorldOptions {
                        respawn_delay_ticks: Some(0),
                        infinite_ammo: Some(true),
                        sv_hit: Some(true),
                        all_weapons: None,
                        no_weak_hook: None,
                    },
                );
                <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut w2, 0, pos);
                let base_seal = <ddai_tsworld::SimWorld as PlanWorld>::get_tee(&w2, 0).unwrap();
                let state = TeeState {
                    pos,
                    vel,
                    frozen,
                    freeze_ticks_left,
                    ..base_seal
                };
                let got_sealed = sealed_in(&mut w2, 0, &state, held);

                if got_escape != want_escape || !safer_ok || got_sealed != want_sealed {
                    shield_mismatch += 1;
                    eprintln!(
                        "shield mismatch at {pos:?}: escapeExists got={got_escape} want={want_escape}, saferInput_ok={safer_ok}, sealedIn got={got_sealed} want={want_sealed}"
                    );
                }
            }
        }
    }
    let _ = opp_rng;

    eprintln!("throwLines/frozenThrowLines: {throw_n} cases, {throw_mismatch} mismatches");
    eprintln!("scriptedAction: {scripted_n} cases, {scripted_mismatch} mismatches");
    eprintln!("restsInFreeze: {seal_n} cases, {seal_mismatch} mismatches");
    eprintln!("hazard/unfreeze fields: {fields_n} cases, {fields_mismatch} mismatches");
    eprintln!("touchesFreeze: {touches_n} cases, {touches_mismatch} mismatches");
    eprintln!(
        "shield (escapeExists/saferInput/sealedIn): {shield_n} cases ({shield_false_n} with escapeExists=false), {shield_mismatch} mismatches"
    );
    assert_eq!(throw_mismatch, 0);
    assert_eq!(scripted_mismatch, 0);
    assert_eq!(seal_mismatch, 0);
    assert_eq!(fields_mismatch, 0);
    assert_eq!(touches_mismatch, 0);
    assert_eq!(shield_mismatch, 0);
    assert!(
        throw_n > 0 && scripted_n > 0 && seal_n > 0 && fields_n > 0 && touches_n > 0 && shield_n > 0,
        "dump produced no cases for at least one component"
    );
    assert!(
        shield_false_n > 0,
        "review round 1, F5: dedicated shield dump should contain at least one escapeExists=false case, got 0"
    );
}

fn plans_match(got: &[Vec<ddai_planner::planner::PlanStep>], want: &[Vec<PlanStepJson>]) -> bool {
    if got.len() != want.len() {
        return false;
    }
    for (g, w) in got.iter().zip(want) {
        if g.len() != w.len() {
            return false;
        }
        for (gs, ws) in g.iter().zip(w) {
            if gs.dir != ws.dir
                || gs.jump != ws.jump
                || gs.hook != ws.hook
                || gs.fire != ws.fire
                || gs.aim.to_bits() != bits_to_f64(&ws.aim).to_bits()
            {
                return false;
            }
        }
    }
    true
}
