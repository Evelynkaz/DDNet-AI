//! Event extraction over a reconstructed timeline: freeze entries, hook episodes, hammer hits and
//! the D-030 "last toucher" attribution. Everything technique detection and the skill signals
//! need is derived here once, from [`FrameRec`]s only (no nicknames, no demo bytes), so the same
//! code runs on synthetic trajectories in the tests.

use std::collections::HashMap;

use ddai_physics::collision::Collision;
use ddai_physics::map::{MapData, TILE_DFREEZE, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID};

use crate::config::Config;
use crate::ingest::KillEvent;
use crate::store::{FrameReader, FrameStore, Item};
use crate::types::{CharRec, FrameRec, char_flags};

pub const HOOK_GRABBED: i8 = 5;
pub const HOOK_FLYING: i8 = 4;
pub const WEAPON_HAMMER: i8 = 0;

/// Tile queries in world pixels.
pub struct Tiles<'a> {
    map: &'a MapData,
}

impl<'a> Tiles<'a> {
    pub fn new(map: &'a MapData) -> Self {
        Tiles { map }
    }

    /// Tile coordinates of a pixel position (clamped to the grid).
    pub fn cell(&self, x: f32, y: f32) -> (i32, i32) {
        let tx = ((x / 32.0).floor() as i32).clamp(0, self.map.width as i32 - 1);
        let ty = ((y / 32.0).floor() as i32).clamp(0, self.map.height as i32 - 1);
        (tx, ty)
    }

    fn idx(&self, tx: i32, ty: i32) -> Option<usize> {
        if tx < 0 || ty < 0 || tx >= self.map.width as i32 || ty >= self.map.height as i32 {
            return None;
        }
        Some(ty as usize * self.map.width as usize + tx as usize)
    }

    /// Game-layer tile id at a cell, `TILE_SOLID` outside the map (like the collision does).
    pub fn game(&self, tx: i32, ty: i32) -> u8 {
        match self.idx(tx, ty) {
            Some(i) => self.map.game[i].index,
            None => TILE_SOLID,
        }
    }

    pub fn front(&self, tx: i32, ty: i32) -> u8 {
        match (self.idx(tx, ty), &self.map.front) {
            (Some(i), Some(f)) => f[i].index,
            _ => 0,
        }
    }

    pub fn is_freeze(&self, tx: i32, ty: i32) -> bool {
        let g = self.game(tx, ty);
        let f = self.front(tx, ty);
        g == TILE_FREEZE || g == TILE_DFREEZE || f == TILE_FREEZE || f == TILE_DFREEZE
    }

    pub fn is_solid(&self, tx: i32, ty: i32) -> bool {
        let g = self.game(tx, ty);
        g == TILE_SOLID || g == TILE_NOHOOK
    }

    pub fn is_hookable(&self, tx: i32, ty: i32) -> bool {
        self.game(tx, ty) == TILE_SOLID
    }

    /// The nearest freeze tile centre within `radius` px (Chebyshev over tile centres), as an
    /// offset from `(x, y)`; ties broken by scan order (row-major), so the result is deterministic.
    pub fn nearest_freeze(&self, x: f32, y: f32, radius: f32) -> Option<(f32, f32)> {
        let (cx, cy) = self.cell(x, y);
        let r = (radius / 32.0).ceil() as i32;
        let mut best: Option<(f32, (f32, f32))> = None;
        for ty in (cy - r)..=(cy + r) {
            for tx in (cx - r)..=(cx + r) {
                if !self.is_freeze(tx, ty) {
                    continue;
                }
                let dx = tx as f32 * 32.0 + 16.0 - x;
                let dy = ty as f32 * 32.0 + 16.0 - y;
                let d = dx.abs().max(dy.abs());
                if d <= radius && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, (dx, dy)));
                }
            }
        }
        best.map(|(_, o)| o)
    }

    /// Scans straight down from `(x, y)` (the column of `x` and its two neighbours) and returns the
    /// distance from `y` to the top edge of the first freeze tile, if no solid tile is hit before
    /// it and the distance is at most `max_px`.
    pub fn freeze_below(&self, x: f32, y: f32, max_px: f32) -> Option<f32> {
        self.scan_column(x, y, max_px, 1)
    }

    /// Same upwards (a freeze ceiling): distance to the tile's bottom edge.
    pub fn freeze_above(&self, x: f32, y: f32, max_px: f32) -> Option<f32> {
        self.scan_column(x, y, max_px, -1)
    }

    fn scan_column(&self, x: f32, y: f32, max_px: f32, dir: i32) -> Option<f32> {
        let (cx, cy) = self.cell(x, y);
        let steps = (max_px / 32.0).ceil() as i32 + 1;
        for s in 0..=steps {
            let ty = cy + dir * s;
            // Distance to the near edge of row `ty` (0 while inside it).
            let edge = if dir > 0 {
                ty as f32 * 32.0 - y
            } else {
                y - (ty + 1) as f32 * 32.0
            };
            let dist = edge.max(0.0);
            if dist > max_px {
                break;
            }
            for tx in [cx, cx - 1, cx + 1] {
                if self.is_freeze(tx, ty) {
                    return Some(dist);
                }
            }
            if self.is_solid(cx, ty) {
                return None;
            }
        }
        None
    }

    /// Scans sideways from `(x, y)` along the rows within +-1 tile of `y` in direction `dir_x`
    /// (`+1`/`-1`) for a freeze tile before a solid one; returns the distance to its near edge if
    /// it is at most `max_px`.
    pub fn freeze_side(&self, x: f32, y: f32, max_px: f32, dir_x: i32) -> Option<f32> {
        let (cx, cy) = self.cell(x, y);
        let steps = (max_px / 32.0).ceil() as i32 + 1;
        for s in 0..=steps {
            let tx = cx + dir_x * s;
            let edge = if dir_x > 0 {
                tx as f32 * 32.0 - x
            } else {
                x - (tx + 1) as f32 * 32.0
            };
            let dist = edge.max(0.0);
            if dist > max_px {
                break;
            }
            for ty in [cy, cy - 1, cy + 1] {
                if self.is_freeze(tx, ty) {
                    return Some(dist);
                }
            }
            if self.is_solid(tx, cy) {
                return None;
            }
        }
        None
    }

    /// True when a solid, hookable tile is at or within 4 px of `(x, y)` (a hook attached to
    /// terrain ends exactly on/inside such a tile).
    pub fn hook_on_terrain(&self, x: f32, y: f32) -> bool {
        for (ox, oy) in [(0.0, 0.0), (4.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0)] {
            let (tx, ty) = self.cell(x + ox, y + oy);
            if self.is_hookable(tx, ty) {
                return true;
            }
        }
        false
    }
}

/// A character entering freeze.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreezeEntry {
    /// Frame index of the first frozen snapshot.
    pub k: usize,
    pub tick: i32,
    pub player: u16,
    pub id: u8,
    /// Frame index of the first snapshot after the entry where the character is no longer
    /// frozen (or absent); `None` if it stayed frozen until the end of the demo.
    pub exit_k: Option<usize>,
    /// Length of the frozen run seen, in ticks.
    pub frozen_ticks: i32,
    /// The character vanished (absent from the next snapshot) while still frozen.
    pub vanished: bool,
}

/// A run of consecutive snapshots in which `actor` had its hook attached to `victim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HookEpisode {
    pub actor: u16,
    pub victim: u16,
    pub start_k: usize,
    pub end_k: usize,
}

/// A hammer hit (fire event with the hammer whose swing point covers a target that then got an
/// impulse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub actor: u16,
    pub victim: u16,
    /// Frame index *after* the hit (the interval `(k-1, k]` contains it).
    pub k: usize,
    pub tick: i32,
    pub victim_frozen_before: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchKind {
    Hook,
    Hammer,
}

/// A freeze entry with its D-030 attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attributed {
    pub entry: FreezeEntry,
    /// Last toucher (hook or hammer) within the attribution window, if any.
    pub toucher: Option<(u16, TouchKind)>,
    /// Credited freeze that became a block (victim stayed frozen long enough or died frozen).
    pub block: bool,
    /// Killed while frozen (a kill message for the victim during the frozen run).
    pub killed: bool,
}

/// The timeline all extraction runs over. The frames live in a [`FrameStore`] (spilled to a
/// temporary file, task 8.4d) and are read through a small page cache: every method below touches
/// a frame neighbourhood, never the whole timeline, so the resident size does not depend on the
/// length of the demo.
pub struct Timeline<'a> {
    pub cfg: &'a Config,
    pub kills: &'a [KillEvent],
    pub tiles: Tiles<'a>,
    map: &'a MapData,
    /// The map's collision, built on first use (only the ballistic counterfactuals need it).
    collision: std::sync::OnceLock<Collision<f32>>,
    store: &'a FrameStore,
    reader: FrameReader<'a>,
}

impl<'a> Timeline<'a> {
    pub fn new(cfg: &'a Config, store: &'a FrameStore, kills: &'a [KillEvent], map: &'a MapData) -> Self {
        Timeline {
            cfg,
            kills,
            tiles: Tiles::new(map),
            map,
            collision: std::sync::OnceLock::new(),
            store,
            reader: store.reader(),
        }
    }

    /// Like [`Timeline::new`] with a custom page-cache size (tests).
    pub fn with_cache(
        cfg: &'a Config,
        store: &'a FrameStore,
        kills: &'a [KillEvent],
        map: &'a MapData,
        cache_pages: usize,
    ) -> Self {
        let mut tl = Self::new(cfg, store, kills, map);
        tl.reader = store.reader_with_cache(cache_pages);
        tl
    }

    /// Pages decoded from the spill file so far (cache misses).
    pub fn page_loads(&self) -> u64 {
        self.reader.page_loads()
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Frame `k` (keeps its page alive while held).
    pub fn frame(&self, k: usize) -> Item<FrameRec> {
        self.reader.get(k)
    }

    /// The map's collision (built once, on first use).
    pub fn collision(&self) -> &Collision<f32> {
        self.collision.get_or_init(|| Collision::new(self.map))
    }

    pub fn slot(&self, k: usize, label: u16) -> Option<usize> {
        if k >= self.len() {
            return None;
        }
        self.frame(k).slot_of(label)
    }

    /// The character `label` in frame `k`, by value (`CharRec` is `Copy`).
    pub fn ch(&self, k: usize, label: u16) -> Option<CharRec> {
        if k >= self.len() {
            return None;
        }
        self.frame(k).by_label(label).copied()
    }

    pub fn tick(&self, k: usize) -> i32 {
        self.store.tick(k)
    }

    /// Frames `k` and `k + 1` are one decision step apart.
    pub fn contiguous(&self, k: usize) -> bool {
        k + 1 < self.len() && self.cfg.is_step(self.tick(k + 1) - self.tick(k))
    }

    /// Label of the character with client id `id` in frame `k`.
    pub fn label_of_id(&self, k: usize, id: i32) -> Option<u16> {
        if k >= self.len() {
            return None;
        }
        self.frame(k)
            .chars
            .iter()
            .find(|c| i32::from(c.id) == id)
            .map(|c| c.player)
    }

    /// Frame index of the last frame with `tick <= t`.
    pub fn frame_at_or_before(&self, t: i32) -> Option<usize> {
        let p = self.store.ticks().partition_point(|&f| f <= t);
        p.checked_sub(1)
    }

    /// First freeze entry of `label` with `from_tick < tick <= from_tick + window`.
    pub fn freeze_entry_within(
        &self,
        entries: &[FreezeEntry],
        label: u16,
        from_tick: i32,
        window: i32,
    ) -> Option<FreezeEntry> {
        entries
            .iter()
            .find(|e| e.player == label && e.tick > from_tick && e.tick <= from_tick + window)
            .copied()
    }

    /// Whether `label` is frozen in any frame with `from_tick < tick <= from_tick + window`
    /// (binary search for the frame range: detectors call this once per event).
    pub fn frozen_within(&self, label: u16, from_tick: i32, window: i32) -> bool {
        let ticks = self.store.ticks();
        let lo = ticks.partition_point(|&f| f <= from_tick);
        let hi = ticks.partition_point(|&f| f <= from_tick + window);
        (lo..hi).any(|k| self.ch(k, label).is_some_and(|c| c.frozen()))
    }
}

/// All freeze entries: frozen at `k`, present and not frozen at `k - 1` (consecutive frames).
pub fn freeze_entries(tl: &Timeline<'_>) -> Vec<FreezeEntry> {
    let mut out = Vec::new();
    for k in 1..tl.len() {
        if !tl.contiguous(k - 1) {
            continue;
        }
        let fk = tl.frame(k);
        let fprev = tl.frame(k - 1);
        for c in &fk.chars {
            if !c.frozen() {
                continue;
            }
            match fprev.by_label(c.player) {
                Some(p) if !p.frozen() => {}
                _ => continue,
            }
            // Follow the frozen run.
            let mut exit_k = None;
            let mut vanished = false;
            let mut last_frozen_tick = tl.tick(k);
            for j in (k + 1)..tl.len() {
                if !tl.contiguous(j - 1) {
                    break;
                }
                match tl.ch(j, c.player) {
                    Some(n) if n.frozen() => last_frozen_tick = tl.tick(j),
                    Some(_) => {
                        exit_k = Some(j);
                        break;
                    }
                    None => {
                        vanished = true;
                        exit_k = Some(j);
                        break;
                    }
                }
            }
            out.push(FreezeEntry {
                k,
                tick: tl.tick(k),
                player: c.player,
                id: c.id,
                exit_k,
                frozen_ticks: last_frozen_tick - tl.tick(k) + tl.cfg.decision_ticks,
                vanished,
            });
        }
    }
    out
}

/// Hook episodes: actor hooked to a victim (`hook_state == GRABBED`, `hooked_player` = victim id).
pub fn hook_episodes(tl: &Timeline<'_>) -> Vec<HookEpisode> {
    let mut out = Vec::new();
    // open[(actor, victim)] = (start_k, last_k)
    let mut open: HashMap<(u16, u16), (usize, usize)> = HashMap::new();
    for k in 0..tl.len() {
        let mut seen: Vec<(u16, u16)> = Vec::new();
        let fk = tl.frame(k);
        for c in &fk.chars {
            if c.hook_state != HOOK_GRABBED || c.hooked_player < 0 {
                continue;
            }
            let Some(v) = fk
                .chars
                .iter()
                .find(|o| i32::from(o.id) == i32::from(c.hooked_player))
                .map(|o| o.player)
            else {
                continue;
            };
            if v == c.player {
                continue;
            }
            seen.push((c.player, v));
        }
        let contiguous_prev = k > 0 && tl.contiguous(k - 1);
        // Close episodes that did not continue.
        let mut closed: Vec<(u16, u16)> = open
            .keys()
            .filter(|key| !(contiguous_prev && seen.contains(key)))
            .copied()
            .collect();
        closed.sort_unstable();
        for key in closed {
            let (s, e) = open.remove(&key).expect("key came from the map");
            out.push(HookEpisode {
                actor: key.0,
                victim: key.1,
                start_k: s,
                end_k: e,
            });
        }
        for key in seen {
            open.entry(key).and_modify(|(_, last)| *last = k).or_insert((k, k));
        }
    }
    let mut rest: Vec<((u16, u16), (usize, usize))> = open.into_iter().collect();
    rest.sort_unstable_by_key(|(key, _)| *key);
    for ((a, v), (s, e)) in rest {
        out.push(HookEpisode {
            actor: a,
            victim: v,
            start_k: s,
            end_k: e,
        });
    }
    out.sort_unstable_by_key(|e| (e.start_k, e.actor, e.victim));
    out
}

fn unit(v: [i32; 2]) -> (f32, f32) {
    let (x, y) = (v[0] as f32, v[1] as f32);
    let n = (x * x + y * y).sqrt();
    if n < 1e-3 { (0.0, -1.0) } else { (x / n, y / n) }
}

/// Hammer hits. `actor` fired the hammer in `(k-1, k]` and
/// - was free (not frozen) at `k-1` and at `k` (a frozen tee cannot fire);
/// - had the swing point (21 px along the aim) at `k-1` within `hammer_reach_px` of the victim;
/// - the victim's velocity changed by at least 4 px/tick (a hammer impulse is ~8-11) **in the
///   direction DDNet's hammer pushes**: `normalize(unit(victim - attacker) + (0, -1.1))`
///   (`character.cpp` `CCharacter::FireWeapon`, WEAPON_HAMMER: sideways and up, never down), with
///   cosine >= 0.6, which is "along the attack, away from the attacker";
/// - and did not jump in the interval (jump button, air-jump bit or jump counter changed), since a
///   jump adds an upward impulse of its own.
pub fn hammer_hits(tl: &Timeline<'_>) -> Vec<Hit> {
    let tc = &tl.cfg.technique;
    let mut out = Vec::new();
    for k in 1..tl.len() {
        if !tl.contiguous(k - 1) {
            continue;
        }
        let fk = tl.frame(k);
        let fprev = tl.frame(k - 1);
        for a in &fk.chars {
            if !a.has(char_flags::FIRED) || a.weapon != WEAPON_HAMMER || a.frozen() {
                continue;
            }
            let Some(a0) = fprev.by_label(a.player) else { continue };
            if a0.frozen() {
                continue;
            }
            let (ux, uy) = unit(a.aim);
            let (hx, hy) = (a0.pos[0] + 21.0 * ux, a0.pos[1] + 21.0 * uy);
            let mut best: Option<(f32, &CharRec, &CharRec)> = None;
            for v in &fk.chars {
                if v.player == a.player {
                    continue;
                }
                let Some(v0) = fprev.by_label(v.player) else { continue };
                let d = ((v0.pos[0] - hx).powi(2) + (v0.pos[1] - hy).powi(2)).sqrt();
                if d > tc.hammer_reach_px || best.is_some_and(|(bd, _, _)| d >= bd) {
                    continue;
                }
                let (dvx, dvy) = (v.vel[0] - v0.vel[0], v.vel[1] - v0.vel[1]);
                let dv = (dvx * dvx + dvy * dvy).sqrt();
                if dv < 4.0 {
                    continue;
                }
                // Direction of the DDNet hammer impulse.
                let (mut ex, mut ey) = (v0.pos[0] - a0.pos[0], v0.pos[1] - a0.pos[1]);
                let en = (ex * ex + ey * ey).sqrt();
                if en < 1e-3 {
                    (ex, ey) = (0.0, 0.0);
                } else {
                    (ex, ey) = (ex / en, ey / en);
                }
                ey -= 1.1;
                let n = (ex * ex + ey * ey).sqrt();
                if (dvx * ex + dvy * ey) / (dv * n) < 0.6 {
                    continue;
                }
                let jumped = (v.has(char_flags::JUMP_HELD) && !v0.has(char_flags::JUMP_HELD))
                    || (v.has(char_flags::AIR_JUMP_USED) && !v0.has(char_flags::AIR_JUMP_USED))
                    || v.jumps_used != v0.jumps_used;
                if jumped {
                    continue;
                }
                best = Some((d, v, v0));
            }
            if let Some((_, v, v0)) = best {
                out.push(Hit {
                    actor: a.player,
                    victim: v.player,
                    k,
                    tick: tl.tick(k),
                    victim_frozen_before: v0.frozen(),
                });
            }
        }
    }
    out
}

/// D-030 attribution of every freeze entry: the last toucher by hook or hammer within
/// `attribution_ticks` before the entry; plus block detection (credited and the victim stays
/// frozen `block_hold_ticks` or dies frozen).
pub fn attribute(tl: &Timeline<'_>, entries: &[FreezeEntry], hooks: &[HookEpisode], hits: &[Hit]) -> Vec<Attributed> {
    let cfg = tl.cfg;
    entries
        .iter()
        .map(|e| {
            let mut best: Option<(i32, u16, TouchKind)> = None; // (touch tick, actor, kind)
            for h in hooks.iter().filter(|h| h.victim == e.player) {
                // The hook touches the victim on every frame of the episode.
                let last = tl.tick(h.end_k).min(e.tick);
                let first = tl.tick(h.start_k);
                if first <= e.tick && e.tick - last <= cfg.attribution_ticks && best.is_none_or(|(t, _, _)| last > t) {
                    best = Some((last, h.actor, TouchKind::Hook));
                }
            }
            for h in hits.iter().filter(|h| h.victim == e.player) {
                if h.tick <= e.tick
                    && e.tick - h.tick <= cfg.attribution_ticks
                    && best.is_none_or(|(t, _, _)| h.tick > t)
                {
                    best = Some((h.tick, h.actor, TouchKind::Hammer));
                }
            }
            let toucher = best.map(|(_, a, kind)| (a, kind));
            let end_tick = e.tick + e.frozen_ticks;
            let killed = tl.kills.iter().any(|kl| {
                i32::from(e.id) == kl.victim && kl.tick >= e.tick && kl.tick <= end_tick + cfg.decision_ticks * 2
            });
            let block = toucher.is_some() && (e.frozen_ticks >= cfg.block_hold_ticks || killed);
            Attributed {
                entry: *e,
                toucher,
                block,
                killed,
            }
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Frame builders for synthetic trajectories.
    use super::*;

    /// A store of `frames` that lives for the rest of the test process (a `Timeline` borrows its
    /// store; leaking one small temporary store per test keeps the tests readable).
    pub fn leak_store(frames: &[FrameRec]) -> &'static FrameStore {
        Box::leak(Box::new(FrameStore::from_frames(frames).expect("spill file")))
    }

    /// Like [`leak_store`] with tiny pages, so that neighbouring frames are often on different pages.
    pub fn leak_store_paged(frames: &[FrameRec], page_len: usize) -> &'static FrameStore {
        Box::leak(Box::new(
            FrameStore::from_frames_paged(frames, page_len).expect("spill file"),
        ))
    }

    pub fn ch(id: u8, player: u16, x: f32, y: f32) -> CharRec {
        CharRec {
            id,
            player,
            team: 0,
            pos: [x, y],
            vel: [0.0, 0.0],
            hook_state: 0,
            hook_pos: [x, y],
            hooked_player: -1,
            flags: char_flags::FRESH,
            freeze_ticks: 0,
            jumps_left: 2,
            jumps_used: 0,
            weapon: 0,
            direction: 0,
            aim: [1, 0],
        }
    }

    pub fn frame(tick: i32, chars: Vec<CharRec>) -> FrameRec {
        FrameRec { tick, chars }
    }

    pub fn frozen(mut c: CharRec) -> CharRec {
        c.flags |= char_flags::FROZEN;
        c.freeze_ticks = 100;
        c
    }

    pub fn grounded(mut c: CharRec) -> CharRec {
        c.flags |= char_flags::GROUNDED;
        c
    }

    pub fn hooking(mut c: CharRec, victim_id: i16) -> CharRec {
        c.hook_state = HOOK_GRABBED;
        c.hooked_player = victim_id;
        c
    }

    pub fn with_vel(mut c: CharRec, vx: f32, vy: f32) -> CharRec {
        c.vel = [vx, vy];
        c
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::testutil::arena_with_pit;

    fn tl_of<'a>(cfg: &'a Config, frames: &[FrameRec], kills: &'a [KillEvent], map: &'a MapData) -> Timeline<'a> {
        Timeline::new(cfg, leak_store(frames), kills, map)
    }

    #[test]
    fn tiles_find_a_pit_below_and_at_the_side() {
        // 20 x 10 arena, freeze pit in the floor at columns 8..12 (x 256..384), floor row y=9.
        let map = arena_with_pit(20, 10, (8, 12));
        let t = Tiles::new(&map);
        // A point above the pit, 3 tiles up: the pit is straight below (floor row centre y = 304).
        assert!(t.freeze_below(320.0, 200.0, 160.0).is_some());
        // Above solid floor there is nothing.
        assert!(t.freeze_below(100.0, 200.0, 160.0).is_none());
        // From the left of the pit, sideways scan on the floor row finds it to the right.
        assert!(t.freeze_side(200.0, 270.0, 160.0, 1).is_some());
        assert!(t.freeze_side(200.0, 270.0, 160.0, -1).is_none());
        assert!(t.nearest_freeze(250.0, 280.0, 64.0).is_some());
        assert!(t.nearest_freeze(50.0, 100.0, 64.0).is_none());
    }

    #[test]
    fn freeze_entry_is_the_transition_only() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let frames = vec![
            frame(10, vec![ch(0, 1, 100.0, 100.0)]),
            frame(12, vec![ch(0, 1, 100.0, 100.0)]),
            frame(14, vec![frozen(ch(0, 1, 100.0, 100.0))]),
            frame(16, vec![frozen(ch(0, 1, 100.0, 100.0))]),
            frame(18, vec![ch(0, 1, 100.0, 100.0)]),
        ];
        let tl = tl_of(&cfg, &frames, &[], &map);
        let e = freeze_entries(&tl);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].tick, 14);
        assert_eq!(e[0].exit_k, Some(4));
        assert_eq!(e[0].frozen_ticks, 4);
        assert!(!e[0].vanished);
    }

    #[test]
    fn a_character_first_seen_frozen_is_not_an_entry() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let frames = vec![
            frame(10, vec![frozen(ch(0, 1, 100.0, 100.0))]),
            frame(12, vec![frozen(ch(0, 1, 100.0, 100.0))]),
        ];
        assert!(freeze_entries(&tl_of(&cfg, &frames, &[], &map)).is_empty());
    }

    #[test]
    fn a_gap_in_the_snapshots_is_not_an_entry() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let frames = vec![
            frame(10, vec![ch(0, 1, 100.0, 100.0)]),
            frame(20, vec![frozen(ch(0, 1, 100.0, 100.0))]),
        ];
        assert!(freeze_entries(&tl_of(&cfg, &frames, &[], &map)).is_empty());
    }

    #[test]
    fn vanishing_while_frozen_is_flagged() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let frames = vec![
            frame(10, vec![ch(0, 1, 100.0, 100.0), ch(1, 2, 500.0, 100.0)]),
            frame(12, vec![frozen(ch(0, 1, 100.0, 100.0)), ch(1, 2, 500.0, 100.0)]),
            frame(14, vec![ch(1, 2, 500.0, 100.0)]),
        ];
        let e = freeze_entries(&tl_of(&cfg, &frames, &[], &map));
        assert_eq!(e.len(), 1);
        assert!(e[0].vanished);
        assert_eq!(e[0].exit_k, Some(2));
    }

    #[test]
    fn hook_episode_spans_consecutive_frames_and_splits_on_gaps() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let a = |x| hooking(ch(0, 1, x, 100.0), 1);
        let v = ch(1, 2, 200.0, 100.0);
        let frames = vec![
            frame(10, vec![ch(0, 1, 100.0, 100.0), v]),
            frame(12, vec![a(100.0), v]),
            frame(14, vec![a(100.0), v]),
            frame(16, vec![ch(0, 1, 100.0, 100.0), v]),
            frame(18, vec![a(100.0), v]),
        ];
        let eps = hook_episodes(&tl_of(&cfg, &frames, &[], &map));
        assert_eq!(eps.len(), 2);
        assert_eq!(
            (eps[0].actor, eps[0].victim, eps[0].start_k, eps[0].end_k),
            (1, 2, 1, 2)
        );
        assert_eq!((eps[1].start_k, eps[1].end_k), (4, 4));
    }

    #[test]
    fn hooking_yourself_or_nobody_is_not_an_episode() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let frames = vec![
            frame(10, vec![hooking(ch(0, 1, 100.0, 100.0), 0)]),
            frame(12, vec![hooking(ch(0, 1, 100.0, 100.0), -1)]),
        ];
        assert!(hook_episodes(&tl_of(&cfg, &frames, &[], &map)).is_empty());
    }

    #[test]
    fn hammer_hit_needs_reach_and_an_impulse() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let mut a1 = ch(0, 1, 100.0, 100.0);
        a1.flags |= char_flags::FIRED;
        a1.aim = [1, 0];
        // Victim 40 px in front, gets thrown (dv ~ 9).
        let v0 = ch(1, 2, 140.0, 100.0);
        let v1 = with_vel(ch(1, 2, 146.0, 92.0), 6.7, -8.4);
        let frames = vec![frame(10, vec![ch(0, 1, 100.0, 100.0), v0]), frame(12, vec![a1, v1])];
        let hits = hammer_hits(&tl_of(&cfg, &frames, &[], &map));
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].actor, hits[0].victim, hits[0].k), (1, 2, 1));
        // Same swing but the victim did not move: a miss.
        let frames_miss = vec![frame(10, vec![ch(0, 1, 100.0, 100.0), v0]), frame(12, vec![a1, v0])];
        assert!(hammer_hits(&tl_of(&cfg, &frames_miss, &[], &map)).is_empty());
        // Victim out of reach.
        let far0 = ch(1, 2, 300.0, 100.0);
        let far1 = with_vel(ch(1, 2, 306.0, 92.0), 6.7, -8.4);
        let frames_far = vec![frame(10, vec![ch(0, 1, 100.0, 100.0), far0]), frame(12, vec![a1, far1])];
        assert!(hammer_hits(&tl_of(&cfg, &frames_far, &[], &map)).is_empty());
    }

    #[test]
    fn hammer_hits_need_a_free_attacker_a_hammer_direction_push_and_no_victim_jump() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let mut swing = ch(0, 1, 100.0, 100.0);
        swing.flags |= char_flags::FIRED;
        swing.aim = [1, 0];
        let v0 = ch(1, 2, 140.0, 100.0);
        let thrown = with_vel(ch(1, 2, 146.0, 92.0), 6.7, -8.4);
        let hits = |a1: CharRec, a0: CharRec, v1: CharRec| {
            let frames = vec![frame(10, vec![a0, v0]), frame(12, vec![a1, v1])];
            hammer_hits(&tl_of(&cfg, &frames, &[], &map)).len()
        };
        assert_eq!(hits(swing, ch(0, 1, 100.0, 100.0), thrown), 1);
        // A frozen attacker cannot fire: at k or at k-1.
        assert_eq!(hits(frozen(swing), ch(0, 1, 100.0, 100.0), thrown), 0);
        assert_eq!(hits(swing, frozen(ch(0, 1, 100.0, 100.0)), thrown), 0);
        // A push that points down, or back at the attacker, is not a hammer impulse.
        assert_eq!(
            hits(
                swing,
                ch(0, 1, 100.0, 100.0),
                with_vel(ch(1, 2, 146.0, 108.0), 6.0, 8.0)
            ),
            0
        );
        assert_eq!(
            hits(
                swing,
                ch(0, 1, 100.0, 100.0),
                with_vel(ch(1, 2, 134.0, 92.0), -6.7, -8.4)
            ),
            0
        );
        // The victim's own jump (button, air jump or jump counter) explains an upward velocity change.
        let mut jump_held = with_vel(ch(1, 2, 146.0, 92.0), 0.0, -12.0);
        jump_held.flags |= char_flags::JUMP_HELD;
        assert_eq!(hits(swing, ch(0, 1, 100.0, 100.0), jump_held), 0);
        let mut air = thrown;
        air.flags |= char_flags::AIR_JUMP_USED;
        assert_eq!(hits(swing, ch(0, 1, 100.0, 100.0), air), 0);
        let mut counted = thrown;
        counted.jumps_used = 1;
        assert_eq!(hits(swing, ch(0, 1, 100.0, 100.0), counted), 0);
    }

    #[test]
    fn attribution_uses_the_last_toucher_within_the_window() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let v = |x, frozen_| {
            let c = ch(1, 2, x, 250.0);
            if frozen_ { frozen(c) } else { c }
        };
        let a = |hook: bool| {
            let c = ch(0, 1, 100.0, 250.0);
            if hook { hooking(c, 1) } else { c }
        };
        let frames = vec![
            frame(10, vec![a(false), v(300.0, false)]),
            frame(12, vec![a(true), v(300.0, false)]),
            frame(14, vec![a(true), v(300.0, false)]),
            frame(16, vec![a(false), v(300.0, false)]),
            frame(18, vec![a(false), v(300.0, true)]),
            frame(20, vec![a(false), v(300.0, true)]),
        ];
        let tl = tl_of(&cfg, &frames, &[], &map);
        let entries = freeze_entries(&tl);
        let hooks = hook_episodes(&tl);
        let at = attribute(&tl, &entries, &hooks, &[]);
        assert_eq!(at.len(), 1);
        assert_eq!(at[0].toucher, Some((1, TouchKind::Hook)));
        assert!(
            !at[0].block,
            "frozen 4 ticks only: below the 50-tick hold, no kill message"
        );
    }

    #[test]
    fn attribution_expires_after_the_window() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config {
            attribution_ticks: 4,
            ..Config::default()
        };
        let a = |hook: bool| {
            let c = ch(0, 1, 100.0, 250.0);
            if hook { hooking(c, 1) } else { c }
        };
        let frames = vec![
            frame(10, vec![a(true), ch(1, 2, 300.0, 250.0)]),
            frame(12, vec![a(false), ch(1, 2, 300.0, 250.0)]),
            frame(14, vec![a(false), ch(1, 2, 300.0, 250.0)]),
            frame(16, vec![a(false), ch(1, 2, 300.0, 250.0)]),
            frame(18, vec![a(false), frozen(ch(1, 2, 300.0, 250.0))]),
        ];
        let tl = tl_of(&cfg, &frames, &[], &map);
        let at = attribute(&tl, &freeze_entries(&tl), &hook_episodes(&tl), &[]);
        assert_eq!(at.len(), 1);
        assert_eq!(at[0].toucher, None, "last touch was 8 ticks ago, window is 4");
    }

    #[test]
    fn block_needs_credit_and_a_long_freeze_or_a_kill() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let mut frames = vec![
            frame(10, vec![hooking(ch(0, 1, 100.0, 250.0), 1), ch(1, 2, 300.0, 250.0)]),
            frame(12, vec![hooking(ch(0, 1, 100.0, 250.0), 1), ch(1, 2, 300.0, 250.0)]),
        ];
        // Frozen for 30 snapshots = 60 ticks >= 50.
        for i in 0..30 {
            frames.push(frame(
                14 + 2 * i,
                vec![ch(0, 1, 100.0, 250.0), frozen(ch(1, 2, 300.0, 250.0))],
            ));
        }
        let tl = tl_of(&cfg, &frames, &[], &map);
        let at = attribute(&tl, &freeze_entries(&tl), &hook_episodes(&tl), &[]);
        assert!(at[0].block);
        // Short freeze but a kill message for the victim id (1) during it.
        let short = vec![
            frame(10, vec![hooking(ch(0, 1, 100.0, 250.0), 1), ch(1, 2, 300.0, 250.0)]),
            frame(12, vec![ch(0, 1, 100.0, 250.0), frozen(ch(1, 2, 300.0, 250.0))]),
            frame(14, vec![ch(0, 1, 100.0, 250.0)]),
        ];
        let kills = [KillEvent {
            tick: 13,
            killer: 0,
            victim: 1,
            weapon: -1,
        }];
        let tl2 = tl_of(&cfg, &short, &kills, &map);
        let at2 = attribute(&tl2, &freeze_entries(&tl2), &hook_episodes(&tl2), &[]);
        assert!(at2[0].killed && at2[0].block);
    }
}
