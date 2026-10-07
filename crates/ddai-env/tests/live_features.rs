//! Task 3.17 (D-111): the features the predictor sees **live** are the ones it was trained on.
//!
//! The trainer's frames come from the arena's exact `World<f32>` (`ddai_env::oppdata::record_game`, `TeeFrame::from_world`). Live, the same
//! function reads the world `LiveWorld` rebuilds from the server's snapshots. This test plays arena games between two scripted puppets
//! (walking, jumping, hooking each other, swinging the hammer, walking into a freeze pit), turns the arena world of every tick into the
//! snapshot a DDNet 20.1 server would send for it (`CNetObj_Character` + `CNetObj_DDNetCharacter`, positions and velocities quantised as on
//! the wire), feeds it to a `LiveWorld`, and compares the two worlds' [`TeeFrame`]s, the 29 frame features and the 16 geometry rays. Then the
//! full 305-number input of the predictor is built from both and compared.
//!
//! A second test replays a real live clip (read-only; skipped when the clip is not on this machine) and checks that what `LiveWorld`
//! rebuilds from real wire data is what the wire says.

use std::path::Path;
use std::sync::Arc;

use ddai_brain::Action;
use ddai_env::arena::{Arena, ArenaDef};
use ddai_env::brains::TimelineBrain;
use ddai_env::config::Rules;
use ddai_env::game::{Layout, play_game_watched};
use ddai_env::sim::PlayerSetup;
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::DEFAULT_TUNE_PARAMS;
use ddai_net::view::CharacterView;
use ddai_oppnet::feature::{FD, frame_features};
use ddai_oppnet::frame::{N_RAYS, TeeFrame, rays};
use ddai_physics::world::World;
use ddai_world::{LiveWorld, SnapshotInput};

fn pit() -> Arena {
    let toml = r#"
name = "pit"
tag = "train"
[map]
kind = "synthetic"
width = 70
height = 18
border = true
rects = [
  { x0 = 1, y0 = 12, x1 = 68, y1 = 15, tile = "solid" },
  { x0 = 40, y0 = 11, x1 = 44, y1 = 11, tile = "freeze" },
  { x0 = 25, y0 = 7, x1 = 27, y1 = 8, tile = "solid" },
]
[spawn]
min_tiles = 0.0
max_tiles = 100.0
rows = [{ y = 11, x0 = 22, x1 = 22 }, { y = 11, x0 = 30, x1 = 30 }]
"#;
    Arena::build(&ArenaDef::parse(toml).unwrap(), Path::new("/nonexistent")).unwrap()
}

fn act(direction: i32, jump: bool, hook: bool, fire: bool) -> Action {
    Action {
        direction,
        jump,
        hook,
        fire,
        ..Action::neutral()
    }
}

/// `(from tick, direction, jump, hook, fire)`.
type Script = Vec<(i32, i32, bool, bool, bool)>;

fn timeline(name: &str, s: &Script) -> TimelineBrain {
    TimelineBrain::new(name, s.iter().map(|&(t, d, j, h, f)| (t, act(d, j, h, f))).collect())
}

/// The snapshot objects a server would send for character `id` of `world` (the extension's freeze end as the server computes it).
/// What the server remembers of one tee for dead reckoning (`m_SendCore` and `m_ReckoningTick`).
#[derive(Default)]
struct Reck {
    sent: Option<objects::Character>,
}

/// Dead reckoning as the server does it (`CCharacter::TickDeferred`): the snapshot carries the old `m_SendCore` and its tick until the idealized
/// continuation of it (no input, an empty world, default tuning) differs from the true core; then both are re-synced to the true core of this tick.
fn reckon(world: &World<f32>, id: u8, cur: &objects::Character, rk: &mut Reck) -> objects::Character {
    let core = world.cores.get(id).expect("alive");
    let holds = rk.sent.as_ref().is_some_and(|sent| {
        ddai_world::reckoning::evolve_character_core(sent, world.tick, &world.collision).write() == core.write()
    });
    if !holds {
        rk.sent = Some(objects::Character {
            tick: world.tick,
            ..*cur
        });
    }
    let sent = rk.sent.as_ref().unwrap();
    // The core fields and the tick are the old send core's; the input direction, the weapon and the attack tick are the live ones.
    objects::Character {
        direction: cur.direction,
        weapon: cur.weapon,
        attack_tick: cur.attack_tick,
        ..*sent
    }
}

fn wire_of(world: &World<f32>, id: u8, rk: Option<&mut Reck>) -> Option<CharacterView> {
    let core = world.cores.get(id)?;
    let ch = world.characters[id as usize].as_ref().filter(|c| c.alive)?;
    let n = core.write();
    let freeze_end = if core.deep_frozen {
        -1
    } else if ch.freeze_time > 0 {
        world.tick + ch.freeze_time
    } else {
        0
    };
    let cur = objects::Character {
        tick: 0,
        x: n.x,
        y: n.y,
        vel_x: n.vel_x,
        vel_y: n.vel_y,
        angle: n.angle,
        direction: n.direction,
        jumped: n.jumped,
        hooked_player: n.hooked_player,
        hook_state: n.hook_state,
        hook_tick: n.hook_tick,
        hook_x: n.hook_x,
        hook_y: n.hook_y,
        hook_dx: n.hook_dx,
        hook_dy: n.hook_dy,
        player_flags: playerflagflag::PLAYING,
        health: 10,
        armor: 0,
        ammo_count: -1,
        weapon: core.active_weapon,
        emote: 0,
        attack_tick: ch.attack_tick,
    };
    let character = match rk {
        Some(rk) => reckon(world, id, &cur, rk),
        None => cur,
    };
    Some(CharacterView {
        id: i32::from(id),
        character,
        ddnet: Some(objects::DDNetCharacter {
            flags: 0,
            freeze_end,
            jumps: core.jumps,
            tele_checkpoint: -1,
            strong_weak_id: i32::from(id),
            jumped_total: -1,
            ninja_activation_tick: -1,
            freeze_start: -1,
            target_x: 0,
            target_y: 0,
            tune_zone_override: -1,
        }),
    })
}

/// The differing fields of two frames, by name.
fn diff(a: &TeeFrame, b: &TeeFrame) -> Vec<&'static str> {
    let mut d = Vec::new();
    macro_rules! f {
        ($($n:ident),*) => { $( if a.$n != b.$n { d.push(stringify!($n)); } )* };
    }
    f!(
        alive,
        pos,
        vel,
        angle,
        direction,
        jumped,
        hook_state,
        hook_on_other,
        hook_pos,
        hook_tick,
        freeze_left,
        attack_age,
        grounded
    );
    d
}

struct Outcome {
    /// Frames whose snapshot character carried an old send core (`tick` neither 0 nor the snapshot's).
    dead_reckoned: u32,
    ticks: u32,
    frames_compared: u32,
    differing: std::collections::BTreeMap<&'static str, (u32, i32)>,
    feature_mismatch: u32,
    ray_mismatch: u32,
    frozen_frames: u32,
    attack_frames: u32,
    hooked_frames: u32,
}

fn run(seed: u64, ours: &Script, theirs: &Script, stride: i32, reckoned: bool) -> Outcome {
    let arena = pit();
    let rules = Rules {
        max_ticks: 200,
        after_ticks: 40,
        ..Rules::default()
    };
    let players = vec![
        PlayerSetup {
            brain: Box::new(timeline("ours", ours)),
            lag: 0,
            label: "ours".into(),
        },
        PlayerSetup {
            brain: Box::new(timeline("theirs", theirs)),
            lag: 0,
            label: "theirs".into(),
        },
    ];
    let mut live = LiveWorld::new(Arc::clone(&arena.map), 0, seed);
    let mut recks = [Reck::default(), Reck::default()];
    let mut out = Outcome {
        dead_reckoned: 0,
        ticks: 0,
        frames_compared: 0,
        differing: Default::default(),
        feature_mismatch: 0,
        ray_mismatch: 0,
        frozen_frames: 0,
        attack_frames: 0,
        hooked_frames: 0,
    };
    play_game_watched(&arena, &rules, seed, Layout::default(), players, &mut |sim, _tick| {
        let w = sim.pw.inner();
        out.ticks += 1;
        // The live bot gets a snapshot every `stride` ticks (25 Hz = 2); the world in between is not seen.
        if w.tick % stride != 0 {
            return true;
        }
        let [r0, r1] = &mut recks;
        let views: Vec<CharacterView> = [(0u8, r0), (1u8, r1)]
            .into_iter()
            .filter_map(|(i, r)| wire_of(w, i, reckoned.then_some(r)))
            .collect();
        out.dead_reckoned += views
            .iter()
            .filter(|v| v.character.tick != 0 && v.character.tick != w.tick)
            .count() as u32;
        if views.len() < 2 {
            return true;
        }
        let mut input = SnapshotInput::new(w.tick, &views, DEFAULT_TUNE_PARAMS);
        input.own_input_at_tick = None;
        live.on_snapshot(input);
        let lw = live.base_world();
        for (id, other) in [(0, 1), (1, 0)] {
            let (Some(a), Some(b)) = (TeeFrame::from_world(w, id, other), TeeFrame::from_world(lw, id, other)) else {
                continue;
            };
            out.frames_compared += 1;
            for name in diff(&a, &b) {
                let e = out.differing.entry(name).or_insert((0, w.tick));
                e.0 += 1;
            }
            if id == 1 {
                out.frozen_frames += u32::from(a.freeze_left > 0);
                out.attack_frames += u32::from(a.attack_age < 30);
                out.hooked_frames += u32::from(a.hook_state > 0);
            }
        }
        // The predictor's per-frame features and rays, as both worlds give them.
        let (am, ao) = (
            TeeFrame::from_world(w, 0, 1).unwrap(),
            TeeFrame::from_world(w, 1, 0).unwrap(),
        );
        let (lm, lo) = (
            TeeFrame::from_world(lw, 0, 1).unwrap(),
            TeeFrame::from_world(lw, 1, 0).unwrap(),
        );
        let (mut fa, mut fl) = ([0.0f32; FD], [0.0f32; FD]);
        frame_features(&am, &ao, &mut fa);
        frame_features(&lm, &lo, &mut fl);
        out.feature_mismatch += u32::from(fa.map(f32::to_bits) != fl.map(f32::to_bits));
        let (mut ra, mut rl) = ([0.0f32; N_RAYS], [0.0f32; N_RAYS]);
        rays(w, ao.pos, &mut ra);
        rays(lw, lo.pos, &mut rl);
        out.ray_mismatch += u32::from(ra.map(f32::to_bits) != rl.map(f32::to_bits));
        true
    })
    .unwrap();
    out
}

fn scripts() -> (Script, Script) {
    let ours: Script = vec![
        (0, 0, false, false, false),
        (6, 1, false, false, false),
        (14, 1, true, false, false),
        (22, 0, false, true, false),
        (30, -1, false, false, true),
        (40, 1, true, true, false),
        (60, -1, false, false, false),
        (80, 1, false, false, true),
    ];
    let theirs: Script = vec![
        (0, 1, false, false, false),
        (8, -1, false, false, false),
        (16, -1, true, true, false),
        (24, 1, false, false, true),
        (34, 1, false, false, false),
        (44, 1, false, false, false),
        (70, 1, false, true, true),
        (90, -1, true, false, false),
    ];
    (ours, theirs)
}

#[test]
fn the_world_a_snapshot_rebuilds_gives_the_arena_features() {
    let (ours, theirs) = scripts();
    for (seed, stride, reckoned) in [(1u64, 2, false), (2, 1, false), (3, 2, true), (4, 1, true)] {
        let o = run(seed, &ours, &theirs, stride, reckoned);
        eprintln!(
            "seed {seed} stride {stride} reckoned {reckoned}: {} dead-reckoned character frames, {} ticks, {} frames compared; differing fields {:?}; feature vectors differing {}, ray sets differing {}; opponent frozen in {}, attacked recently in {}, hooking in {} frames",
            o.dead_reckoned,
            o.ticks,
            o.frames_compared,
            o.differing,
            o.feature_mismatch,
            o.ray_mismatch,
            o.frozen_frames,
            o.attack_frames,
            o.hooked_frames
        );
        assert!(o.frames_compared > 60, "{}", o.frames_compared);
        assert!(o.differing.is_empty(), "frame fields differ: {:?}", o.differing);
        if reckoned {
            assert!(
                o.dead_reckoned > 20,
                "the server's old send cores really were used: {}",
                o.dead_reckoned
            );
        }
        assert_eq!(o.feature_mismatch, 0);
        assert_eq!(o.ray_mismatch, 0);
        assert!(
            o.hooked_frames > 2 && o.attack_frames > 3 && (reckoned || o.frozen_frames > 3),
            "the scripts exercise the hook, the hammer and (in the exact-core runs) the freeze"
        );
    }
}

/// A live duel clip of the lead's diagnosis folder (read-only; two tees in every frame): `None` when it is not on this machine.
fn duel_clip() -> Option<(ddai_clip::format::Clip, Arc<ddai_physics::map::MapData>)> {
    let home = std::path::PathBuf::from(std::env::var_os("HOME")?);
    let clip =
        ddai_clip::format::Clip::read(&home.join("aiddnet/data/scratch/duel-diag/self-freeze-11626416-s186.clip"))
            .ok()?;
    let hex: String = clip.header.map_sha256.iter().map(|b| format!("{b:02x}")).collect();
    let map_file = std::fs::read_dir(home.join("aiddnet/data/maps/cache"))
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().ends_with(&format!("{hex}.map")))?;
    let map = ddai_map::load_map(&std::fs::read(map_file.path()).ok()?).ok()?.data;
    Some((clip, Arc::new(map)))
}

/// What the wire of a real snapshot says, against what `LiveWorld` rebuilds and the predictor reads: the fields the dead reckoning does not
/// touch must be the wire's own values, the rest must be what the server's continuation gives (finite, on the map, in the ranges the features clip to).
#[test]
fn a_real_clip_frame_gives_the_features_the_wire_says() {
    let Some((clip, map)) = duel_clip() else {
        eprintln!("skipped: the duel clip or its map is not on this machine");
        return;
    };
    let own = clip.header.own_id;
    let mut lw = LiveWorld::new(map, own, clip.header.world_seed);
    let (mut frames, mut exact, mut reckoned) = (0u32, 0u32, 0u32);
    let mut worst_feature = 0.0f32;
    for f in &clip.frames {
        ddai_clip::replay::feed(&mut lw, &clip, f);
        let (Some(_), Some(opp)) = (f.tee(own), f.tees.iter().find(|t| t.id != own)) else {
            continue;
        };
        let w = lw.base_world();
        let (Some(a), Some(b)) = (
            TeeFrame::from_world(w, own, opp.id),
            TeeFrame::from_world(w, opp.id, own),
        ) else {
            continue;
        };
        frames += 1;
        let ch = &opp.ch;
        // Never touched by the reckoning: who we are paired with, how long it is frozen, when it last used a weapon, where it aims,
        // which way it walks.
        assert!(a.alive && b.alive);
        assert_eq!(i32::from(b.direction), ch.direction.clamp(-1, 1), "tick {}", f.tick);
        assert_eq!(b.angle, ch.angle as f32 / 256.0);
        assert_eq!(
            i32::from(b.attack_age),
            (f.tick - ch.attack_tick).clamp(0, 120),
            "tick {}: the age of the last weapon use",
            f.tick
        );
        let want_freeze = match opp.dd.as_ref() {
            Some(d) if d.freeze_end > 0 => (d.freeze_end - f.tick).max(0),
            _ => 0,
        };
        assert_eq!(i32::from(b.freeze_left), want_freeze, "tick {}", f.tick);
        if ch.tick == 0 || ch.tick == f.tick {
            // The snapshot's own core is the state at its tick (the server re-synced it this very tick): nothing was evolved, so the wire is the world.
            exact += 1;
            assert_eq!(b.pos, [ch.x as f32, ch.y as f32], "tick {}", f.tick);
            assert_eq!(b.vel, [ch.vel_x as f32 / 256.0, ch.vel_y as f32 / 256.0]);
            assert_eq!(b.hook_state, ch.hook_state.clamp(-1, 8) as i8);
            assert_eq!(b.jumped, (ch.jumped & 3) as u8);
            assert_eq!(b.hook_on_other, ch.hooked_player == own);
        } else {
            reckoned += 1;
            // The server's continuation moved it at most a few ticks of travel from the wire's position.
            let dist = ((b.pos[0] - ch.x as f32).powi(2) + (b.pos[1] - ch.y as f32).powi(2)).sqrt();
            assert!(dist < 200.0, "tick {}: {dist} px from the wire", f.tick);
        }
        let mut feat = [0.0f32; FD];
        frame_features(&a, &b, &mut feat);
        assert!(feat.iter().all(|v| v.is_finite()), "tick {}: {feat:?}", f.tick);
        worst_feature = worst_feature.max(feat.iter().fold(0.0f32, |m, v| m.max(v.abs())));
        let mut ray = [0.0f32; N_RAYS];
        rays(w, b.pos, &mut ray);
        assert!(ray.iter().all(|v| (0.0..=1.0).contains(v)), "tick {}: {ray:?}", f.tick);
    }
    eprintln!(
        "clip: {frames} duel frames, {exact} read straight off the wire, {reckoned} dead-reckoned; largest feature {worst_feature}"
    );
    assert!(frames > 600 && exact > 50, "{frames} {exact}");
    assert!(
        worst_feature <= 4.0 + 1e-6,
        "the features stay inside their clips: {worst_feature}"
    );
}

/// The whole live pipeline (`LiveOpp`: history, window prediction, scoring against the snapshots that follow, guard, log) over the real duel clip
/// with the real model, when both are on this machine. It prints the report the analysis tool would, and checks that the pipeline produces a sound
/// one (its numbers are the transfer check of 3.15 in the live metric; the clip is one duel, so no claim is made about the model here).
#[test]
fn the_live_pipeline_over_a_real_clip_scores_the_model_against_hold() {
    use ddai_oppnet::OppPredictor;
    use ddai_oppnet::live::analyze::Report;
    use ddai_oppnet::live::guard::GuardConfig;
    use ddai_oppnet::live::{LiveOpp, Pair, WindowUse};
    use ddai_physics::core::PlayerInput as Wire;
    use ddai_world::player_input_from_net;

    let Some((clip, map)) = duel_clip() else {
        eprintln!("skipped: the duel clip or its map is not on this machine");
        return;
    };
    let home = std::path::PathBuf::from(std::env::var_os("HOME").unwrap());
    let model_path = std::env::var_os("DDAI_WINDOW_MODEL").map_or_else(
        || home.join("aiddnet/data/runs/E-028/m1.oppnet"),
        std::path::PathBuf::from,
    );
    let Ok(pred) = OppPredictor::load(&model_path) else {
        eprintln!("skipped: no model at {}", model_path.display());
        return;
    };
    let mut l = LiveOpp::new(pred, GuardConfig::default(), [0; 32]).unwrap();
    let own = clip.header.own_id;
    let mut lw = LiveWorld::new(map, own, clip.header.world_seed);
    let lag = 3usize;
    let mut victim = Vec::new();
    let mut used = 0u32;
    let mut decisions = 0u32;
    let mut tag = String::new();
    // What one decision costs in this process (the snapshot's history and scoring work, the forward pass, the log line): microseconds.
    let mut costs_us: Vec<f64> = Vec::new();
    for (i, f) in clip.frames.iter().enumerate() {
        ddai_clip::replay::feed(&mut lw, &clip, f);
        let Some(opp) = f.tees.iter().find(|t| t.id != own).map(|t| t.id) else {
            continue;
        };
        if !f.own_alive || f.tees.len() != 2 || f.tees_dropped != 0 {
            continue;
        }
        tag.clear();
        tag.push_str(&format!("c{opp}-test"));
        // Our inputs for the next `lag` ticks, as the bot has them: those sent for the ticks after this snapshot.
        let sent_at = |t: i32| {
            clip.frames[i + 1..(i + 3).min(clip.frames.len())]
                .iter()
                .flat_map(|f| f.sent.iter())
                .find(|s| s.tick == t)
                .map(|s| player_input_from_net(s.input.to_net()))
        };
        let ours: Option<Vec<Wire>> = (1..=lag as i32).map(|k| sent_at(f.tick + k)).collect();
        match ours {
            Some(o) => {
                let hold = lw.held_input_of(opp).unwrap_or_default();
                decisions += 1;
                let t = std::time::Instant::now();
                let pair = Pair {
                    world: lw.base_world(),
                    self_id: own,
                    target: opp,
                    tag: &tag,
                };
                let how = l.window(&pair, &o, &hold, &mut victim);
                costs_us.push(t.elapsed().as_secs_f64() * 1e6);
                used += u32::from(how == WindowUse::Model);
            }
            None => l.observe(&Pair {
                world: lw.base_world(),
                self_id: own,
                target: opp,
                tag: &tag,
            }),
        }
    }
    let mut r = Report::new();
    for line in String::from_utf8(l.take_log()).unwrap().lines() {
        r.add_line(line);
    }
    let g = l.guard_status();
    eprintln!("{}", r.render(false));
    costs_us.sort_by(f64::total_cmp);
    let q = |p: f64| costs_us[((costs_us.len() - 1) as f64 * p) as usize];
    eprintln!(
        "guard: {g:?}; {decisions} decisions, the model drove {used}; LiveOpp::window cost p50 {:.1} us, p90 {:.1} us, p99 {:.1} us, max {:.1} us",
        q(0.5),
        q(0.9),
        q(0.99),
        q(1.0)
    );
    assert!(
        q(0.5) < 100.0,
        "a decision costs microseconds, not milliseconds: p50 {:.1} us",
        q(0.5)
    );
    assert!(decisions > 300 && r.samples() > 800, "{decisions} {}", r.samples());
    assert_eq!(r.bad_lines, 0);
    assert!(g.model_cost.is_finite() && g.hold_cost.is_finite() && g.hold_cost > 0.0);
    // Sanity, not a verdict: a model within a factor of two of hold on a real duel is reading the live features sensibly.
    assert!(g.model_cost < 2.0 * g.hold_cost, "{g:?}");
}
