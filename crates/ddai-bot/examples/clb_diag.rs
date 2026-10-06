//! Task 3.12b diagnosis tool: where did a crossing of a Copy Love Box freeze tube leave the plan, and who was near?
//!
//! `cargo run --release -p ddai-bot --example clb_diag -- [--tail N] [--map-cache DIR] <clip>...`
//!
//! For every clip it rebuilds the recorded world frame by frame (the bot's own `LiveWorld`) and, for each step
//! between two frames, rolls our tee **alone** (every other tee removed) on the inputs we sent and compares the
//! result with the recording. A step that the isolated roll does not reproduce, while the roll with all the
//! recorded tees does, was moved by someone else (a shove, a hook, a hammer); a step neither reproduces is the
//! network (input timing, a correction). The first such step of the last crossing is the *divergence*; the
//! tool prints it with the tees near us, and the last `N` frames as a timeline.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_clip::{Clip, Frame};
use ddai_net::tuning::{DEFAULT_TUNE_PARAMS, NUM_TUNE_PARAMS, TeamsState, from_array};
use ddai_net::view::CharacterView;
use ddai_physics::core::PlayerInput;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_world::{LiveWorld, SnapshotInput, player_input_from_net};

fn feed(lw: &mut LiveWorld, clip: &Clip, f: &Frame) {
    let characters: Vec<CharacterView> = f
        .tees
        .iter()
        .map(|t| CharacterView {
            id: t.id,
            character: t.ch.to_net(),
            ddnet: t.dd.map(|d| d.to_net()),
        })
        .collect();
    let projectiles: Vec<_> = f.projectiles.iter().map(|p| p.to_view()).collect();
    let switches: Vec<_> = f.switches.iter().map(|s| s.to_net()).collect();
    let mut tuning = DEFAULT_TUNE_PARAMS;
    for c in &clip.header.tuning {
        if c.from_tick <= f.tick && c.values.len() == NUM_TUNE_PARAMS {
            let mut a = [0i32; NUM_TUNE_PARAMS];
            a.copy_from_slice(&c.values);
            tuning = from_array(c.received as usize, a);
        }
    }
    let mut teams_state = None;
    for c in &clip.header.teams {
        if c.from_tick <= f.tick {
            let mut teams = [0i32; 128];
            for (i, t) in c.teams.iter().take(128).enumerate() {
                teams[i] = *t;
            }
            teams_state = Some(TeamsState {
                teams,
                received: (c.received as usize).min(128),
            });
        }
    }
    lw.on_snapshot(SnapshotInput {
        tick: f.tick,
        characters: &characters,
        tuning,
        switch_states: &switches,
        teams: teams_state.as_ref(),
        own_input_at_tick: f.sent.last().map(|s| player_input_from_net(s.input.to_net())),
        projectiles: &projectiles,
    });
}

fn input_at(next: &Frame, tick: i32, held: &mut PlayerInput) {
    if let Some(s) = next.sent.iter().find(|s| s.tick == tick) {
        *held = player_input_from_net(s.input.to_net());
    }
}

/// One frame of our track: tick, tile, frozen, who hooks us, the unfrozen tees within 420 px.
type Track = (i32, (i32, i32), bool, Vec<i32>, Vec<i32>);

fn tag(clip: &Clip, id: i32) -> String {
    clip.header
        .players
        .iter()
        .find(|p| p.id == id)
        .map_or(format!("#{id}"), |p| p.tag.clone())
}

fn near(
    clip: &Clip,
    f: &Frame,
    own_id: i32,
    me: (f64, f64),
    rec: &ddai_physics::world::World<f32>,
    within: f64,
) -> Vec<String> {
    let mut v: Vec<(f64, String)> = f
        .tees
        .iter()
        .filter(|t| t.id != own_id)
        .filter_map(|t| {
            // reckoned core (the raw integers can be old, see `main`)
            let c = rec.cores.get(t.id as u8).map(|c| c.write())?;
            let p = (f64::from(c.x), f64::from(c.y));
            let d = (p.0 - me.0).hypot(p.1 - me.1);
            (d <= within).then(|| {
                let hooks_us = c.hooked_player == own_id;
                (
                    d,
                    format!(
                        "{} d{:.0} ({:+.0},{:+.0}) v({:.1},{:.1}){}{}{}",
                        tag(clip, t.id),
                        d,
                        p.0 - me.0,
                        p.1 - me.1,
                        f64::from(c.vel_x) / 256.0,
                        f64::from(c.vel_y) / 256.0,
                        if t.frozen { " FROZEN" } else { "" },
                        if hooks_us { " HOOKS-US" } else { "" },
                        if c.hook_state > 0 && !hooks_us { " hook-out" } else { "" },
                    ),
                )
            })
        })
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v.into_iter().map(|x| x.1).collect()
}

fn main() {
    let mut tail = 60usize;
    let mut cache = PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache");
    let mut files = Vec::new();
    let mut ascii: Option<(i32, i32, i32, i32)> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ascii" => {
                let v: Vec<i32> = it
                    .next()
                    .expect("--ascii X0,Y0,X1,Y1")
                    .split(',')
                    .map(|x| x.parse().expect("number"))
                    .collect();
                ascii = Some((v[0], v[1], v[2], v[3]));
            }
            "--tail" => tail = it.next().and_then(|v| v.parse().ok()).expect("--tail N"),
            "--map-cache" => cache = PathBuf::from(it.next().expect("--map-cache DIR")),
            _ => files.push(PathBuf::from(a)),
        }
    }
    if let Some((x0, y0, x1, y1)) = ascii {
        // `--ascii` needs one clip only for the map identity.
        let clip = Clip::read(&files[0]).expect("clip");
        let bytes = ddai_client::map_cache::read_cached(&cache, &clip.header.map_name, &clip.header.map_sha256)
            .expect("map in the cache");
        let map = Arc::new(ddai_map::load_map(&bytes).expect("map").data);
        let w = PhysicsWorld::new(map, 1);
        let col = w.collision();
        use ddai_planner::plan_world::PlanCollision;
        print!("     ");
        for x in x0..=x1 {
            print!("{}", x % 10);
        }
        println!();
        for y in y0..=y1 {
            print!("{y:>4} ");
            for x in x0..=x1 {
                let (px, py) = (f64::from(x * 32 + 16), f64::from(y * 32 + 16));
                let c = if PlanCollision::is_death(col, px, py) {
                    'X'
                } else if PlanCollision::is_solid(col, px, py) {
                    if PlanCollision::is_no_hook(col, px, py) {
                        'N'
                    } else {
                        '#'
                    }
                } else if PlanCollision::is_freeze(col, px, py) {
                    'f'
                } else if PlanCollision::is_un_freeze(col, px, py) {
                    'u'
                } else {
                    '.'
                };
                print!("{c}");
            }
            println!();
        }
        return;
    }
    for file in files {
        let clip = Clip::read(&file).expect("clip");
        let bytes = ddai_client::map_cache::read_cached(&cache, &clip.header.map_name, &clip.header.map_sha256)
            .expect("map in the cache");
        let map = Arc::new(ddai_map::load_map(&bytes).expect("map").data);
        let own_id = clip.header.own_id;
        println!(
            "== {} reason {} tick {} note {}",
            file.file_name()
                .map_or(String::new(), |n| n.to_string_lossy().to_string()),
            clip.header.reason.kind,
            clip.header.reason.tick,
            clip.header.reason.note
        );
        let mut lw = LiveWorld::new(Arc::clone(&map), own_id, clip.header.world_seed);
        feed(&mut lw, &clip, &clip.frames[0]);
        let n = clip.frames.len();
        let from_frame = n.saturating_sub(tail);
        let mut first_div: Option<usize> = None;
        let mut last_label = String::new();
        // Per frame (reckoned): tick, our tile, frozen, who hooks us, the unfrozen tees within 420 px.
        let mut track: Vec<Track> = Vec::new();
        for i in 1..n {
            let (prev, next) = (&clip.frames[i - 1], &clip.frames[i]);
            let Some(prev_own) = prev.tee(own_id) else {
                feed(&mut lw, &clip, next);
                continue;
            };
            let have = next.tee(own_id).is_some() && prev.own_alive && next.own_alive;
            // the roll: state of `prev`, then `next.tick - prev.tick` steps on the sent inputs.
            let mut full = PhysicsWorld::new(Arc::clone(&map), 1);
            full.sync_from(lw.base_world());
            let mut alone = PhysicsWorld::new(Arc::clone(&map), 1);
            alone.sync_from(lw.base_world());
            for id in 0..128 {
                if id != own_id {
                    alone.remove_tee(id);
                }
            }
            feed(&mut lw, &clip, next);
            if !have {
                continue;
            }
            let rec = lw.base_world();
            let mut held = prev
                .sent
                .last()
                .map_or_else(PlayerInput::default, |s| player_input_from_net(s.input.to_net()));
            let mut t = prev.tick + 1;
            while t <= next.tick {
                input_at(next, t, &mut held);
                let planner_in = ddai_planner::physics_adapter::from_ddnet_input(&held);
                full.set_input(own_id, planner_in);
                alone.set_input(own_id, planner_in);
                let _ = full.step();
                let _ = alone.step();
                t += 1;
            }
            let rec_own = rec.cores.get(own_id as u8).map(|c| c.write());
            if let Some(c) = &rec_own {
                let me = (f64::from(c.x), f64::from(c.y));
                let (mut hookers, mut close) = (Vec::new(), Vec::new());
                for t in &next.tees {
                    if t.id == own_id {
                        continue;
                    }
                    if let Some(o) = rec.cores.get(t.id as u8).map(|c| c.write()) {
                        if o.hooked_player == own_id {
                            hookers.push(t.id);
                        }
                        let d = (f64::from(o.x) - me.0).hypot(f64::from(o.y) - me.1);
                        if d <= 420.0 && !t.frozen {
                            close.push(t.id);
                        }
                    }
                }
                let frozen = next.tee(own_id).is_some_and(|t| t.frozen);
                track.push((
                    next.tick,
                    ((me.0 / 32.0) as i32, (me.1 / 32.0) as i32),
                    frozen,
                    hookers,
                    close,
                ));
            }
            let a = alone.get_tee(own_id);
            let f = full.get_tee(own_id);
            let (Some(rc), Some(a), Some(f)) = (rec_own, a, f) else {
                continue;
            };
            let (rx, ry) = (f64::from(rc.x), f64::from(rc.y));
            let err = |s: &ddai_planner::types::TeeState| ((s.pos.x - rx).hypot(s.pos.y - ry)).round();
            let (ea, ef) = (err(&a), err(&f));
            let crossing = next.bot.has(ddai_clip::BotRec::BIT_CROSSING);
            let diverged = ea > 1.5;
            if diverged && first_div.is_none() && crossing {
                first_div = Some(i);
            }
            if i >= from_frame || diverged && crossing {
                let label = clip
                    .header
                    .labels
                    .get(usize::from(next.bot.walk))
                    .map_or("", String::as_str);
                if label != last_label {
                    println!(
                        "      walk: {label:?} (brain {} wb-holding {})",
                        next.bot.brain,
                        next.bot.has(ddai_clip::BotRec::BIT_WB_HOLDING)
                    );
                    last_label = label.to_string();
                }
                let own = next.tee(own_id).expect("own");
                let inp = next.own_input();
                // The server sends our core only when it leaves the client's dead reckoning, so the raw
                // integers can be old; what the bot knew is the reckoned core of the base world.
                let (px, py) = (f64::from(rc.x), f64::from(rc.y));
                let (vx, vy) = (f64::from(rc.vel_x) / 256.0, f64::from(rc.vel_y) / 256.0);
                println!(
                    "#{i:<4} t{} {} tile({:>3},{:>3}) xy({:>5.0},{:>5.0}) v({:>5.1},{:>5.1}) hk{}/{} {} in[d{} j{} h{} aim({},{})] alone-err {:>4} full-err {:>4}{} | {}",
                    next.tick,
                    if crossing { "X" } else { "." },
                    (px / 32.0) as i32,
                    (py / 32.0) as i32,
                    px,
                    py,
                    vx,
                    vy,
                    own.ch.hook_state,
                    own.ch.hooked_player,
                    if own.frozen { "FROZEN" } else { "      " },
                    inp.map_or(0, |s| s.input.direction),
                    inp.map_or(0, |s| s.input.jump),
                    inp.map_or(0, |s| s.input.hook),
                    inp.map_or(0, |s| s.input.target_x),
                    inp.map_or(0, |s| s.input.target_y),
                    ea,
                    ef,
                    if diverged && ef <= 1.5 {
                        "  <== OTHERS"
                    } else if diverged {
                        "  <== UNEXPLAINED"
                    } else {
                        ""
                    },
                    near(&clip, next, own_id, (px, py), rec, 220.0).join("; "),
                );
            }
            let _ = prev_own;
        }
        // Landings: we come out of the freeze plug under the passage (tile y 66..67 -> 68+) frozen.
        for k in 1..track.len() {
            let (prev, cur) = (&track[k - 1], &track[k]);
            if prev.1.1 < 68 && cur.1.1 >= 68 && cur.2 && (cur.1.0 < 110 || cur.1.0 > 124) {
                let horizon = cur.0 + 160;
                let end = track.iter().take_while(|t| t.0 <= horizon).last().expect("track");
                let first_hook = track[k..].iter().find(|t| t.0 <= horizon && !t.3.is_empty());
                let mut every: Vec<i32> = Vec::new();
                for t in track[k..].iter().take_while(|t| t.0 <= horizon) {
                    for h in &t.3 {
                        if !every.contains(h) {
                            every.push(*h);
                        }
                    }
                }
                println!(
                    "LANDING t{} tile({},{}) unfrozen-within-420px {} hooked-by {} (first at +{} ticks) -> at +160 ticks tile({},{}) {}{}",
                    cur.0,
                    cur.1.0,
                    cur.1.1,
                    cur.4.len(),
                    every.len(),
                    first_hook.map_or(-1, |t| t.0 - cur.0),
                    end.1.0,
                    end.1.1,
                    if end.2 { "frozen" } else { "free" },
                    if end.0 < horizon { " (clip ends)" } else { "" },
                );
            }
        }
        match first_div {
            Some(i) => println!("first crossing divergence at frame #{i} (tick {})", clip.frames[i].tick),
            None => println!("no crossing step diverged from the isolated roll"),
        }
    }
}
