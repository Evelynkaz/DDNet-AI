//! The **privileged critic** of the recurrent PPO (task 8.5b): a small MLP on the exact simulator state, never part of a bundle and
//! never run in play.
//!
//! The actor sees only rays; the critic sees what the simulator knows: both tees' position, velocity, freeze timer, hook state and
//! hook point, the vector between them, the tiles around each (solid / freeze / death, 7 x 7 cells), BFS distances to the nearest
//! freeze and the nearest non-freeze cell of the map ([`MapFields`]) and the episode's clock (whether the opponent has been frozen
//! yet, ticks since, ticks left in the held-block window). That is the asymmetric actor-critic of Pinto et al.: a better baseline for
//! the advantage, with no change to what the policy has to infer.
//!
//! The network is a feed-forward `tanh` MLP with a hand-written backward (flat parameter vector, Adam through
//! [`crate::trainer::Adam`]). Mini-batch gradients are summed in fixed chunks, so they do not depend on the thread count.

use std::collections::VecDeque;

use ddai_brain::{CharacterObservation, HOOK_FLYING, HOOK_GRABBED, HOOK_IDLE, Observation};
use ddai_fly::rng::SplitMix64;
use ddai_physics::map::{MapData, TILE_DEATH, TILE_DFREEZE, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Half the side of the tile window around a tee (7 x 7 cells).
pub const TILE_HALF: i32 = 3;
const TILE_CHANNELS: usize = 3;
/// BFS distances are capped (and normalised) at this many tiles.
pub const FIELD_CAP: u8 = 8;
/// A freeze lasts at most this many ticks after the tee leaves the freeze tile (`sv_freeze_delay` 3 s).
pub const FREEZE_TICKS: f32 = 150.0;

/// Number of critic inputs.
pub const INPUT_DIM: usize = 2 * TEE_DIM + 3 + 4 + 2 * (2 * TILE_HALF as usize + 1).pow(2) * TILE_CHANNELS + 4;
const TEE_DIM: usize = 15;

const CLASS_AIR: u8 = 0;
const CLASS_SOLID: u8 = 1;
const CLASS_FREEZE: u8 = 2;
const CLASS_DEATH: u8 = 3;

/// Per-map tile classes and two BFS distance fields (8-connected, through non-solid cells), computed once per arena.
#[derive(Debug, Clone)]
pub struct MapFields {
    pub width: i32,
    pub height: i32,
    class: Vec<u8>,
    /// Steps from a cell to the nearest cell that is neither freeze nor solid (`0` on such a cell), capped at [`FIELD_CAP`]: how deep
    /// in a freeze zone a tee is, i.e. how far it has to get to be free.
    to_nonfreeze: Vec<u8>,
    /// Steps from a cell to the nearest freeze cell (`0` on one), capped.
    to_freeze: Vec<u8>,
}

fn class_of(index: u8) -> u8 {
    match index {
        TILE_SOLID | TILE_NOHOOK => CLASS_SOLID,
        TILE_FREEZE | TILE_DFREEZE => CLASS_FREEZE,
        TILE_DEATH => CLASS_DEATH,
        _ => CLASS_AIR,
    }
}

impl MapFields {
    pub fn new(map: &MapData) -> MapFields {
        let (w, h) = (map.width as i32, map.height as i32);
        let n = (w * h) as usize;
        let mut class = vec![CLASS_AIR; n];
        for (i, c) in class.iter_mut().enumerate() {
            let mut k = class_of(map.game[i].index);
            if k == CLASS_AIR
                && let Some(front) = &map.front
            {
                k = class_of(front[i].index);
            }
            *c = k;
        }
        let bfs = |is_source: &dyn Fn(u8) -> bool| -> Vec<u8> {
            let mut dist = vec![FIELD_CAP; n];
            let mut q = VecDeque::new();
            for i in 0..n {
                if is_source(class[i]) {
                    dist[i] = 0;
                    q.push_back(i);
                }
            }
            while let Some(i) = q.pop_front() {
                let (x, y) = ((i as i32) % w, (i as i32) / w);
                if dist[i] >= FIELD_CAP {
                    continue;
                }
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if (dx, dy) == (0, 0) || nx < 0 || ny < 0 || nx >= w || ny >= h {
                            continue;
                        }
                        let j = (ny * w + nx) as usize;
                        if class[j] != CLASS_SOLID && dist[j] > dist[i] + 1 {
                            dist[j] = dist[i] + 1;
                            q.push_back(j);
                        }
                    }
                }
            }
            dist
        };
        let to_nonfreeze = bfs(&|c| c == CLASS_AIR);
        let to_freeze = bfs(&|c| c == CLASS_FREEZE);
        MapFields {
            width: w,
            height: h,
            class,
            to_nonfreeze,
            to_freeze,
        }
    }

    fn at(&self, tx: i32, ty: i32) -> u8 {
        if tx < 0 || ty < 0 || tx >= self.width || ty >= self.height {
            CLASS_SOLID
        } else {
            self.class[(ty * self.width + tx) as usize]
        }
    }

    fn field(&self, f: &[u8], tx: i32, ty: i32) -> u8 {
        if tx < 0 || ty < 0 || tx >= self.width || ty >= self.height {
            FIELD_CAP
        } else {
            f[(ty * self.width + tx) as usize]
        }
    }

    /// From `(px, py)` along the unit vector `(ux, uy)` (pixels, in steps of 8): the distance to the first freeze or death tile and to the first
    /// solid tile; `max` where there is none within `max`.
    pub fn ray_distances(&self, px: f32, py: f32, ux: f32, uy: f32, max: f32) -> (f32, f32) {
        let (mut d_freeze, mut d_solid) = (max, max);
        let mut d = 0.0f32;
        while d <= max {
            let (tx, ty) = (
                ((px + ux * d) / 32.0).floor() as i32,
                ((py + uy * d) / 32.0).floor() as i32,
            );
            match self.at(tx, ty) {
                CLASS_FREEZE | CLASS_DEATH if d_freeze >= max => d_freeze = d,
                CLASS_SOLID => {
                    d_solid = d;
                    break;
                }
                _ => {}
            }
            d += 8.0;
        }
        (d_freeze, d_solid)
    }

    /// Distance (tiles, capped) of the tile at `px` to the nearest non-freeze cell.
    pub fn exit_distance(&self, px: f32, py: f32) -> u8 {
        self.field(
            &self.to_nonfreeze,
            (px / 32.0).floor() as i32,
            (py / 32.0).floor() as i32,
        )
    }

    pub fn freeze_distance(&self, px: f32, py: f32) -> u8 {
        self.field(&self.to_freeze, (px / 32.0).floor() as i32, (py / 32.0).floor() as i32)
    }
}

/// What the critic needs beyond one observation: the episode's clock.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpisodeClock {
    /// The tick of the deciding freeze of the opponent (a bank start's handover; a full game's first block once it has happened).
    pub freeze_tick: Option<i32>,
    /// The held-block window.
    pub window: i32,
    pub max_ticks: i32,
}

fn tee_features(c: &CharacterObservation, f: &MapFields, out: &mut Vec<f32>) {
    let (wpx, hpx) = (f.width as f32 * 32.0, f.height as f32 * 32.0);
    out.push(c.pos.x / wpx);
    out.push(c.pos.y / hpx);
    out.push((c.vel.x / 16.0).clamp(-3.0, 3.0));
    out.push((c.vel.y / 16.0).clamp(-3.0, 3.0));
    out.push((c.freeze_ticks_remaining as f32 / FREEZE_TICKS).clamp(0.0, 2.0));
    out.push(f32::from(u8::from(c.is_frozen || c.freeze_ticks_remaining > 0)));
    out.push(1.0); // alive
    out.push(f32::from(u8::from(c.hook_state == HOOK_FLYING)));
    out.push(f32::from(u8::from(c.hook_state == HOOK_GRABBED)));
    out.push(f32::from(u8::from(c.hooked_player >= 0)));
    let hooking = c.hook_state != HOOK_IDLE;
    out.push(if hooking {
        ((c.hook_pos.x - c.pos.x) / 400.0).clamp(-1.5, 1.5)
    } else {
        0.0
    });
    out.push(if hooking {
        ((c.hook_pos.y - c.pos.y) / 400.0).clamp(-1.5, 1.5)
    } else {
        0.0
    });
    out.push(c.jumps_left as f32 / 2.0);
    out.push(f32::from(u8::from(c.grounded)));
    out.push(c.direction as f32);
}

fn tile_window(c: &CharacterObservation, f: &MapFields, out: &mut Vec<f32>) {
    let (tx, ty) = ((c.pos.x / 32.0).floor() as i32, (c.pos.y / 32.0).floor() as i32);
    for dy in -TILE_HALF..=TILE_HALF {
        for dx in -TILE_HALF..=TILE_HALF {
            let k = f.at(tx + dx, ty + dy);
            out.push(f32::from(u8::from(k == CLASS_SOLID)));
            out.push(f32::from(u8::from(k == CLASS_FREEZE)));
            out.push(f32::from(u8::from(k == CLASS_DEATH)));
        }
    }
}

/// The critic's input for the decision with observation `obs` (the first of `others` is the opponent; absent = dead).
pub fn critic_features(obs: &Observation, f: &MapFields, clock: &EpisodeClock) -> Vec<f32> {
    let mut x = Vec::with_capacity(INPUT_DIM);
    let me = &obs.self_state;
    tee_features(me, f, &mut x);
    let opp = obs.others.first();
    match opp {
        Some(o) => tee_features(o, f, &mut x),
        None => {
            // A dead opponent: zeros and a "dead" mark in the alive slot (index 6 of the tee block).
            x.extend(std::iter::repeat_n(0.0, TEE_DIM));
        }
    }
    // Relative position and distance.
    match opp {
        Some(o) => {
            let (dx, dy) = ((o.pos.x - me.pos.x) / 400.0, (o.pos.y - me.pos.y) / 400.0);
            x.push(dx.clamp(-2.0, 2.0));
            x.push(dy.clamp(-2.0, 2.0));
            x.push((dx * dx + dy * dy).sqrt().min(3.0));
        }
        None => x.extend([0.0, 0.0, 0.0]),
    }
    let cap = f32::from(FIELD_CAP);
    x.push(opp.map_or(0.0, |o| f32::from(f.exit_distance(o.pos.x, o.pos.y)) / cap));
    x.push(opp.map_or(0.0, |o| f32::from(f.freeze_distance(o.pos.x, o.pos.y)) / cap));
    x.push(f32::from(f.exit_distance(me.pos.x, me.pos.y)) / cap);
    x.push(f32::from(f.freeze_distance(me.pos.x, me.pos.y)) / cap);
    tile_window(me, f, &mut x);
    match opp {
        Some(o) => tile_window(o, f, &mut x),
        None => x.extend(std::iter::repeat_n(
            0.0,
            (2 * TILE_HALF as usize + 1).pow(2) * TILE_CHANNELS,
        )),
    }
    // The clock.
    let tick = obs.tick;
    let (frozen_yet, since, left) = match clock.freeze_tick {
        Some(ft) if tick >= ft => (
            1.0,
            (tick - ft) as f32 / clock.window as f32,
            (ft + clock.window - tick) as f32 / clock.window as f32,
        ),
        _ => (0.0, 0.0, 1.0),
    };
    x.push(frozen_yet);
    x.push(since.clamp(0.0, 1.5));
    x.push(left.clamp(0.0, 1.0));
    x.push((tick as f32 / clock.max_ticks.max(1) as f32).clamp(0.0, 2.0));
    // Dead opponent flag lives in the free alive slot of the opponent block: set it to 1 when present, 0 when dead (already 0).
    debug_assert_eq!(x.len(), INPUT_DIM);
    x
}

/// The potential of a state for the potential-based shaping (task 8.5b): the **hold margin** of the opponent in `[0, 1]`: `1` when it is
/// dead (it never comes back), `0` when it is free, and while it is frozen `0.35 * (freeze time left / 150) + 0.35 * (BFS distance of
/// its cell to the nearest non-freeze cell / 8)`: the time it still needs to thaw, and how deep in the freeze it sits (a tee deep in a pit
/// has a long way to be free even once thawed). A *potential*, not a reward: it only ever enters as a difference.
pub fn hold_potential(obs: &Observation, f: &MapFields) -> f32 {
    match obs.others.first() {
        None => 1.0,
        Some(o) if o.freeze_ticks_remaining > 0 || o.is_frozen => {
            let t = (o.freeze_ticks_remaining as f32 / FREEZE_TICKS).min(1.0);
            let d = f32::from(f.exit_distance(o.pos.x, o.pos.y)) / f32::from(FIELD_CAP);
            0.35 * t + 0.35 * d
        }
        Some(_) => 0.0,
    }
}

// --- The network -------------------------------------------------------------------------------------------------------------

/// A `tanh` MLP `in -> h1 -> h2 -> 1`, all parameters in one flat vector: `[W1 (h1 x in), b1, W2 (h2 x h1), b2, w3 (h2), b3]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Critic {
    pub n_in: usize,
    pub h1: usize,
    pub h2: usize,
    pub params: Vec<f32>,
}

/// The weight blocks of the network: `W1, b1, W2, b2, w3, b3`.
type Split<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a [f32], f32);

/// Activations of one forward pass.
pub struct Acts {
    a1: Vec<f32>,
    a2: Vec<f32>,
    pub value: f32,
}

impl Critic {
    pub fn num_params(n_in: usize, h1: usize, h2: usize) -> usize {
        h1 * n_in + h1 + h2 * h1 + h2 + h2 + 1
    }

    /// Initialised from `seed`: weights `N(0, 1/fan_in)` (the output layer 100 times smaller, so the first values are about zero).
    pub fn new(n_in: usize, h1: usize, h2: usize, seed: u64) -> Critic {
        let mut rng = SplitMix64::new(seed ^ 0xC217_1C00);
        let mut params = vec![0.0f32; Self::num_params(n_in, h1, h2)];
        let (w1, rest) = params.split_at_mut(h1 * n_in);
        for w in w1 {
            *w = rng.next_gaussian() / (n_in as f32).sqrt();
        }
        let (_b1, rest) = rest.split_at_mut(h1);
        let (w2, rest) = rest.split_at_mut(h2 * h1);
        for w in w2 {
            *w = rng.next_gaussian() / (h1 as f32).sqrt();
        }
        let (_b2, rest) = rest.split_at_mut(h2);
        let (w3, _b3) = rest.split_at_mut(h2);
        for w in w3 {
            *w = 0.01 * rng.next_gaussian() / (h2 as f32).sqrt();
        }
        Critic { n_in, h1, h2, params }
    }

    fn split(&self) -> Split<'_> {
        let (w1, rest) = self.params.split_at(self.h1 * self.n_in);
        let (b1, rest) = rest.split_at(self.h1);
        let (w2, rest) = rest.split_at(self.h2 * self.h1);
        let (b2, rest) = rest.split_at(self.h2);
        let (w3, b3) = rest.split_at(self.h2);
        (w1, b1, w2, b2, w3, b3[0])
    }

    pub fn forward(&self, x: &[f32]) -> Acts {
        assert_eq!(x.len(), self.n_in);
        let (w1, b1, w2, b2, w3, b3) = self.split();
        let a1: Vec<f32> = (0..self.h1)
            .map(|j| {
                let row = &w1[j * self.n_in..(j + 1) * self.n_in];
                (b1[j] + row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>()).tanh()
            })
            .collect();
        let a2: Vec<f32> = (0..self.h2)
            .map(|j| {
                let row = &w2[j * self.h1..(j + 1) * self.h1];
                (b2[j] + row.iter().zip(&a1).map(|(w, v)| w * v).sum::<f32>()).tanh()
            })
            .collect();
        let value = b3 + w3.iter().zip(&a2).map(|(w, v)| w * v).sum::<f32>();
        Acts { a1, a2, value }
    }

    pub fn value(&self, x: &[f32]) -> f32 {
        self.forward(x).value
    }

    /// Adds `dloss/dparams` of one sample with `dloss/dvalue = dv` into `grad`.
    pub fn backward(&self, x: &[f32], acts: &Acts, dv: f32, grad: &mut [f32]) {
        let (_w1, _b1, w2, _b2, w3, _b3) = self.split();
        let (g_w1, rest) = grad.split_at_mut(self.h1 * self.n_in);
        let (g_b1, rest) = rest.split_at_mut(self.h1);
        let (g_w2, rest) = rest.split_at_mut(self.h2 * self.h1);
        let (g_b2, rest) = rest.split_at_mut(self.h2);
        let (g_w3, g_b3) = rest.split_at_mut(self.h2);
        g_b3[0] += dv;
        let mut d2 = vec![0.0f32; self.h2];
        for j in 0..self.h2 {
            g_w3[j] += dv * acts.a2[j];
            d2[j] = dv * w3[j] * (1.0 - acts.a2[j] * acts.a2[j]);
            g_b2[j] += d2[j];
            for k in 0..self.h1 {
                g_w2[j * self.h1 + k] += d2[j] * acts.a1[k];
            }
        }
        for k in 0..self.h1 {
            let mut s = 0.0f32;
            for j in 0..self.h2 {
                s += d2[j] * w2[j * self.h1 + k];
            }
            let d1 = s * (1.0 - acts.a1[k] * acts.a1[k]);
            g_b1[k] += d1;
            for (g, &xi) in g_w1[k * self.n_in..(k + 1) * self.n_in].iter_mut().zip(x) {
                *g += d1 * xi;
            }
        }
    }

    /// The mean squared error `0.5 (V - R)^2` over `samples` and its gradient with respect to the parameters, summed in fixed chunks of
    /// 64 samples (deterministic whatever the thread count). Returns `(mean loss, mean gradient)`.
    pub fn mse_grad(&self, xs: &[&[f32]], targets: &[f32]) -> (f32, Vec<f32>) {
        assert_eq!(xs.len(), targets.len());
        let n = xs.len().max(1) as f32;
        let parts: Vec<(f64, Vec<f32>)> = xs
            .par_chunks(64)
            .zip(targets.par_chunks(64))
            .map(|(cx, ct)| {
                let mut g = vec![0.0f32; self.params.len()];
                let mut loss = 0.0f64;
                for (x, &t) in cx.iter().zip(ct) {
                    let a = self.forward(x);
                    let e = a.value - t;
                    loss += 0.5 * f64::from(e) * f64::from(e);
                    self.backward(x, &a, e / n, &mut g);
                }
                (loss, g)
            })
            .collect();
        let mut grad = vec![0.0f32; self.params.len()];
        let mut loss = 0.0f64;
        for (l, g) in parts {
            loss += l;
            for (a, b) in grad.iter_mut().zip(&g) {
                *a += b;
            }
        }
        (loss as f32 / n, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gradient_matches_finite_differences() {
        let c = Critic::new(12, 7, 5, 3);
        let mut c = c;
        // Larger output weights so the output depends on everything.
        let n = c.params.len();
        for (i, p) in c.params.iter_mut().enumerate() {
            *p += 0.05 * ((i * 7919 % 13) as f32 - 6.0) / 6.0;
        }
        let mut rng = SplitMix64::new(9);
        let xs: Vec<Vec<f32>> = (0..5).map(|_| (0..12).map(|_| rng.next_gaussian()).collect()).collect();
        let ts = [0.3f32, -0.5, 1.0, 0.0, 0.7];
        let refs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
        let (_, g) = c.mse_grad(&refs, &ts);
        let loss = |c: &Critic| c.mse_grad(&refs, &ts).0 as f64;
        for i in (0..n).step_by(5) {
            let eps = 1e-3f32;
            let (mut a, mut b) = (c.clone(), c.clone());
            a.params[i] += eps;
            b.params[i] -= eps;
            let numeric = ((loss(&a) - loss(&b)) / (2.0 * f64::from(eps))) as f32;
            assert!(
                (numeric - g[i]).abs() < 2e-3 * (1.0 + numeric.abs()),
                "param {i}: {} vs {numeric}",
                g[i]
            );
        }
    }

    #[test]
    fn the_critic_fits_a_simple_function() {
        let mut c = Critic::new(4, 16, 16, 1);
        let mut adam = crate::trainer::Adam::new(c.params.len());
        let mut rng = SplitMix64::new(2);
        let xs: Vec<Vec<f32>> = (0..256)
            .map(|_| (0..4).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect())
            .collect();
        let ts: Vec<f32> = xs.iter().map(|x| 0.8 * x[0] - 0.5 * x[1] * x[2] + 0.2).collect();
        let refs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
        let lrs = vec![3e-3f32; c.params.len()];
        let first = c.mse_grad(&refs, &ts).0;
        for _ in 0..600 {
            let (_, g) = c.mse_grad(&refs, &ts);
            let mut p = c.params.clone();
            assert!(adam.step(&mut p, &g, &lrs, 1.0));
            c.params = p;
        }
        let last = c.mse_grad(&refs, &ts).0;
        assert!(last < 0.05 * first, "{first} -> {last}");
    }

    fn map_with_pit() -> MapData {
        use ddai_physics::map::Tile;
        let (w, h) = (12usize, 10usize);
        let mut game = vec![Tile::default(); w * h];
        let set = |game: &mut Vec<Tile>, x: usize, y: usize, i: u8| game[y * w + x].index = i;
        for x in 0..w {
            set(&mut game, x, 9, TILE_SOLID);
        }
        for y in 6..9 {
            for x in 4..8 {
                set(&mut game, x, y, TILE_FREEZE);
            }
        }
        MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn the_exit_distance_grows_with_the_depth_in_the_freeze() {
        let f = MapFields::new(&map_with_pit());
        // Row 6 is the top of the pit, row 8 the bottom (solid below): deeper = farther from a non-freeze cell.
        let d = |tx: i32, ty: i32| f.exit_distance(tx as f32 * 32.0 + 16.0, ty as f32 * 32.0 + 16.0);
        assert_eq!(d(1, 5), 0, "open air is not freeze");
        assert_eq!(d(4, 6), 1, "the rim of the pit");
        assert!(d(5, 8) >= 2, "{}", d(5, 8));
        assert!(d(5, 8) > d(4, 6));
        assert_eq!(f.freeze_distance(5.0 * 32.0 + 16.0, 7.0 * 32.0 + 16.0), 0);
        assert!(f.freeze_distance(1.0 * 32.0, 3.0 * 32.0) >= 2);
    }

    #[test]
    fn features_have_the_declared_size_and_the_potential_is_zero_free_and_one_dead() {
        use std::sync::Arc;
        let map = Arc::new(map_with_pit());
        let f = MapFields::new(&map);
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(40.0, 150.0);
        let mut opp = CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(5.0 * 32.0 + 16.0, 8.0 * 32.0 + 16.0);
        opp.freeze_ticks_remaining = 150;
        opp.is_frozen = true;
        let mut obs = Observation {
            map,
            tick: 100,
            self_state: me,
            others: vec![opp],
            target_id: Some(1),
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        let clock = EpisodeClock {
            freeze_tick: Some(90),
            window: 250,
            max_ticks: 1500,
        };
        assert_eq!(critic_features(&obs, &f, &clock).len(), INPUT_DIM);
        let frozen = hold_potential(&obs, &f);
        assert!(frozen > 0.35 && frozen < 1.0, "{frozen}");
        obs.others[0].freeze_ticks_remaining = 0;
        obs.others[0].is_frozen = false;
        assert_eq!(hold_potential(&obs, &f), 0.0, "a free opponent");
        obs.others.clear();
        assert_eq!(hold_potential(&obs, &f), 1.0, "a dead opponent");
        assert_eq!(critic_features(&obs, &f, &clock).len(), INPUT_DIM);
    }
}
