//! Geometric hook anchors (task 3.5, D-048): the hookable solid points around a tee that a hook
//! thrown from where it stands can actually reach, found by ray-casting the same
//! `intersect_line_hook` the physics uses (so no-hook tiles and hook-through tiles are respected
//! exactly). Walls, ceilings and platform edges become escape, trajectory-change and swing plans
//! in [`crate::hybrid::techniques`].
//!
//! **Bounded and cached.** The full ring of rays is cast once per tile (from the tile centre) and
//! the hookable points cached per tile; a query then re-derives angle and distance from the exact
//! position, keeps at most `limit` anchors chosen for *angular diversity* (farthest-point
//! selection, so eight anchors cover eight different directions instead of eight points of one
//! wall) and re-verifies only those with a ray from the exact position. Ray casts are counted
//! (`AnchorCache::rays`) as work for the D-045 accounting.

use std::collections::HashMap;

use crate::plan_world::{CFLAG_NOHOOK, CFLAG_SOLID, PlanCollision};
use crate::tuning::HOOK_LENGTH;
use crate::vmath::{Vec2, vdistance};
use ddai_jsmath as js;

/// Rays per full turn (10 degrees apart: about 66 px between neighbouring rays at full length).
pub const RAYS: usize = 36;

/// The cache is wiped when it holds this many tiles.
pub const MAX_CACHED_TILES: usize = 4096;

/// Anchors closer than this are not worth hooking (the rope would be shorter than the tee).
const MIN_DIST_PX: f64 = 48.0;

/// Where on the anchor surface the hook lands, as seen from the tee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorKind {
    /// The surface faces down: a ceiling (or the underside of a platform).
    Ceiling,
    /// The surface faces up: a floor or a platform top.
    Floor,
    /// A vertical wall.
    Wall,
}

/// One hookable point, from a given origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    /// The hit position (world px).
    pub point: Vec2,
    /// Absolute angle from the origin to the point (`atan2(dy, dx)`, y down).
    pub angle: f64,
    pub dist: f64,
    pub kind: AnchorKind,
}

/// Per-tile cache of the hookable points visible from the tile centre.
#[derive(Debug, Default)]
pub struct AnchorCache {
    tiles: HashMap<(i32, i32), Vec<Vec2>>,
    /// Ray casts done so far (work counter).
    pub rays: u64,
    pub hits: u64,
    pub misses: u64,
}

fn kind_of(before: Vec2, at: Vec2) -> AnchorKind {
    let (nx, ny) = (before.x - at.x, before.y - at.y);
    if js::abs(ny) > js::abs(nx) {
        if ny > 0.0 {
            AnchorKind::Ceiling
        } else {
            AnchorKind::Floor
        }
    } else {
        AnchorKind::Wall
    }
}

fn hookable(hit: &crate::plan_world::LineHit) -> bool {
    (hit.collision & CFLAG_SOLID) != 0 && (hit.collision & CFLAG_NOHOOK) == 0
}

/// Smallest absolute difference between two angles, in `[0, pi]`.
fn angle_gap(a: f64, b: f64) -> f64 {
    let mut d = js::abs(a - b);
    while d > 2.0 * js::PI {
        d -= 2.0 * js::PI;
    }
    if d > js::PI { 2.0 * js::PI - d } else { d }
}

impl AnchorCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The hookable points visible from the centre of `tile`, from cache or by casting the ring.
    fn tile_points(&mut self, col: &impl PlanCollision, tile: (i32, i32)) -> &[Vec2] {
        if !self.tiles.contains_key(&tile) {
            self.misses += 1;
            // Bounded memory over a long session: forget everything rather than grow without limit
            // (a map has a few thousand walkable tiles; a wipe costs one ring per tile visited).
            if self.tiles.len() >= MAX_CACHED_TILES {
                self.tiles.clear();
            }
            let centre = Vec2 {
                x: f64::from(tile.0) * 32.0 + 16.0,
                y: f64::from(tile.1) * 32.0 + 16.0,
            };
            let mut pts: Vec<Vec2> = Vec::new();
            let mut seen: Vec<(i32, i32)> = Vec::new();
            for k in 0..RAYS {
                let a = (k as f64) * 2.0 * js::PI / (RAYS as f64) - js::PI;
                let to = Vec2 {
                    x: centre.x + js::cos(a) * *HOOK_LENGTH,
                    y: centre.y + js::sin(a) * *HOOK_LENGTH,
                };
                self.rays += 1;
                let hit = col.intersect_line_hook(centre, to);
                if !hookable(&hit) {
                    continue;
                }
                let key = (
                    js::floor(hit.out_pos.x / 32.0) as i32,
                    js::floor(hit.out_pos.y / 32.0) as i32,
                );
                if !seen.contains(&key) {
                    seen.push(key);
                    pts.push(hit.out_pos);
                }
            }
            self.tiles.insert(tile, pts);
        } else {
            self.hits += 1;
        }
        &self.tiles[&tile]
    }

    /// At most `limit` verified anchors from `origin`, most diverse in direction first. Empty
    /// when nothing hookable is in reach.
    pub fn select(&mut self, col: &impl PlanCollision, origin: Vec2, limit: usize) -> Vec<Anchor> {
        let tile = (js::floor(origin.x / 32.0) as i32, js::floor(origin.y / 32.0) as i32);
        let reach = *HOOK_LENGTH - 2.0;
        let mut cands: Vec<(f64, f64, Vec2)> = self
            .tile_points(col, tile)
            .iter()
            .filter_map(|&p| {
                let d = vdistance(origin, p);
                (d >= MIN_DIST_PX && d <= reach).then(|| (js::atan2(p.y - origin.y, p.x - origin.x), d, p))
            })
            .collect();
        // Nearest first: the seed of the farthest-point selection and the tie-break.
        cands.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut chosen: Vec<(f64, f64, Vec2)> = Vec::new();
        let mut out: Vec<Anchor> = Vec::new();
        let mut spent = 0usize;
        while out.len() < limit && !cands.is_empty() && spent < cands.len() + limit {
            spent += 1;
            let pick = if chosen.is_empty() {
                0
            } else {
                let mut best = 0usize;
                let mut best_gap = -1.0;
                for (i, c) in cands.iter().enumerate() {
                    let gap = chosen
                        .iter()
                        .map(|ch| angle_gap(c.0, ch.0))
                        .fold(f64::INFINITY, f64::min);
                    if gap > best_gap + 1e-9 {
                        best_gap = gap;
                        best = i;
                    }
                }
                best
            };
            let c = cands.remove(pick);
            // Re-verify from the exact origin: the cached point came from the tile centre.
            let dir_len = vdistance(origin, c.2);
            let to = Vec2 {
                x: origin.x + (c.2.x - origin.x) / dir_len * (dir_len + 6.0),
                y: origin.y + (c.2.y - origin.y) / dir_len * (dir_len + 6.0),
            };
            self.rays += 1;
            let hit = col.intersect_line_hook(origin, to);
            if !hookable(&hit) || vdistance(hit.out_pos, c.2) > 24.0 {
                continue;
            }
            chosen.push(c);
            out.push(Anchor {
                point: hit.out_pos,
                angle: js::atan2(hit.out_pos.y - origin.y, hit.out_pos.x - origin.x),
                dist: vdistance(origin, hit.out_pos),
                kind: kind_of(hit.out_before_pos, hit.out_pos),
            });
        }
        out
    }

    pub fn cached_tiles(&self) -> usize {
        self.tiles.len()
    }
}

/// The shield's anchor-aimed hook escapes (task 3.5b): up to `max_anchors` of `anchors` -- walls first
/// (a sideways hook), then ceilings (a hook that hauls us up), nearest first -- each with and without
/// a jump. The input walks towards a wall anchor (no direction for a ceiling), holds the hook and aims
/// at the anchor point relative to the tee, as the ordinary escapes aim (a vector of length 300).
pub fn hook_escapes(anchors: &[Anchor], me: Vec2, max_anchors: usize) -> Vec<crate::types::PlayerInput> {
    let mut walls: Vec<&Anchor> = anchors.iter().filter(|a| a.kind == AnchorKind::Wall).collect();
    let mut ceilings: Vec<&Anchor> = anchors.iter().filter(|a| a.kind == AnchorKind::Ceiling).collect();
    walls.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    ceilings.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    let take_walls = walls.len().min(max_anchors.saturating_sub(1).max(1));
    let picked: Vec<&Anchor> = walls
        .into_iter()
        .take(take_walls)
        .chain(ceilings)
        .take(max_anchors)
        .collect();
    let mut out = Vec::with_capacity(2 * picked.len());
    for a in picked {
        let (dx, dy) = (a.point.x - me.x, a.point.y - me.y);
        let len = js::max(1.0, js::sqrt(dx * dx + dy * dy));
        let dir = if a.kind == AnchorKind::Wall {
            if dx > 0.0 { 1 } else { -1 }
        } else {
            0
        };
        for jump in [0, 1] {
            let mut e = crate::types::empty_input();
            e.direction = dir;
            e.jump = jump;
            e.hook = 1;
            e.target_x = js::round(dx / len * 300.0);
            e.target_y = js::round(dy / len * 300.0);
            out.push(e);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_world::LineHit;

    /// A grid world: `#` solid, `n` no-hook, `.` air; 32 px tiles.
    struct Grid {
        rows: Vec<Vec<u8>>,
    }

    impl Grid {
        fn new(rows: &[&str]) -> Grid {
            Grid {
                rows: rows.iter().map(|r| r.bytes().collect()).collect(),
            }
        }
        fn at(&self, x: f64, y: f64) -> u8 {
            let (tx, ty) = (js::floor(x / 32.0) as i32, js::floor(y / 32.0) as i32);
            if ty < 0 || tx < 0 || ty as usize >= self.rows.len() || tx as usize >= self.rows[0].len() {
                return b'#';
            }
            self.rows[ty as usize][tx as usize]
        }
    }

    impl PlanCollision for Grid {
        fn identity(&self) -> u64 {
            0
        }
        fn width(&self) -> i32 {
            self.rows[0].len() as i32
        }
        fn height(&self) -> i32 {
            self.rows.len() as i32
        }
        fn game_tile(&self, tx: i32, ty: i32) -> u8 {
            match self.at(f64::from(tx) * 32.0 + 1.0, f64::from(ty) * 32.0 + 1.0) {
                b'#' => 1,
                b'n' => 3,
                _ => 0,
            }
        }
        fn is_solid(&self, x: f64, y: f64) -> bool {
            matches!(self.at(x, y), b'#' | b'n')
        }
        fn is_death(&self, _: f64, _: f64) -> bool {
            false
        }
        fn is_freeze(&self, _: f64, _: f64) -> bool {
            false
        }
        fn is_un_freeze(&self, _: f64, _: f64) -> bool {
            false
        }
        fn is_no_hook(&self, x: f64, y: f64) -> bool {
            self.at(x, y) == b'n'
        }
        fn test_box(&self, _: Vec2, _: Vec2) -> bool {
            false
        }
        fn intersect_line(&self, a: Vec2, b: Vec2) -> LineHit {
            self.intersect_line_hook(a, b)
        }
        fn intersect_line_hook(&self, a: Vec2, b: Vec2) -> LineHit {
            let n = (vdistance(a, b) + 1.0) as i32;
            let mut last = a;
            for i in 0..=n {
                let t = f64::from(i) / f64::from(n.max(1));
                let p = Vec2 {
                    x: a.x + (b.x - a.x) * t,
                    y: a.y + (b.y - a.y) * t,
                };
                match self.at(p.x, p.y) {
                    b'#' => {
                        return LineHit {
                            collision: CFLAG_SOLID,
                            out_pos: p,
                            out_before_pos: last,
                        };
                    }
                    b'n' => {
                        return LineHit {
                            collision: CFLAG_SOLID | CFLAG_NOHOOK,
                            out_pos: p,
                            out_before_pos: last,
                        };
                    }
                    _ => {}
                }
                last = p;
            }
            LineHit {
                collision: 0,
                out_pos: b,
                out_before_pos: b,
            }
        }
        fn has_tele(&self) -> bool {
            false
        }
        fn tele_at(&self, _: f64, _: f64) -> (i32, i32) {
            (0, 0)
        }
        fn tele_outs_for(&self, _: i32) -> Vec<Vec2> {
            Vec::new()
        }
    }

    fn room() -> Grid {
        Grid::new(&[
            "##########", //
            "#........#", //
            "#........#", //
            "#........#", //
            "#........#", //
            "##########",
        ])
    }

    #[test]
    fn a_closed_room_offers_anchors_in_every_direction_class() {
        let col = room();
        let mut cache = AnchorCache::new();
        let origin = Vec2 { x: 160.0, y: 96.0 };
        let a = cache.select(&col, origin, 8);
        assert_eq!(a.len(), 8);
        assert!(a.iter().any(|x| x.kind == AnchorKind::Ceiling));
        assert!(a.iter().any(|x| x.kind == AnchorKind::Floor));
        assert!(a.iter().any(|x| x.kind == AnchorKind::Wall));
        for x in &a {
            assert!(x.dist >= MIN_DIST_PX && x.dist <= *HOOK_LENGTH);
        }
        // Angular diversity: no two of eight anchors are within 20 degrees of each other.
        for i in 0..a.len() {
            for j in (i + 1)..a.len() {
                assert!(angle_gap(a[i].angle, a[j].angle) > 20f64.to_radians(), "{i} {j}");
            }
        }
    }

    #[test]
    fn no_hook_tiles_are_never_anchors() {
        let col = Grid::new(&[
            "nnnnnnnnnn", //
            "n........n", //
            "n........n", //
            "n........n", //
            "n........n", //
            "nnnnnnnnnn",
        ]);
        let mut cache = AnchorCache::new();
        assert!(cache.select(&col, Vec2 { x: 160.0, y: 96.0 }, 8).is_empty());
    }

    #[test]
    fn anchors_are_cached_per_tile_and_rays_counted() {
        let col = room();
        let mut cache = AnchorCache::new();
        let o = Vec2 { x: 150.0, y: 90.0 };
        cache.select(&col, o, 4);
        let rays_after_first = cache.rays;
        assert_eq!(cache.misses, 1);
        assert!(rays_after_first >= RAYS as u64);
        // Same tile, slightly different position: no new ring, only the verification rays.
        cache.select(&col, Vec2 { x: 155.0, y: 92.0 }, 4);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 1);
        assert!(
            cache.rays - rays_after_first <= 8,
            "only verification rays: {}",
            cache.rays
        );
        assert_eq!(cache.cached_tiles(), 1);
    }

    #[test]
    fn the_cache_is_bounded() {
        let col = room();
        let mut cache = AnchorCache::new();
        for i in 0..(MAX_CACHED_TILES + 50) {
            // Far outside the room: every query is a new tile and finds no anchors.
            cache.select(
                &col,
                Vec2 {
                    x: 32.0 * i as f64 + 16.0,
                    y: 5000.0,
                },
                2,
            );
        }
        assert!(cache.cached_tiles() <= MAX_CACHED_TILES);
    }

    #[test]
    fn limit_bounds_the_number_of_anchors() {
        let col = room();
        let mut cache = AnchorCache::new();
        assert_eq!(cache.select(&col, Vec2 { x: 160.0, y: 96.0 }, 3).len(), 3);
        assert!(cache.select(&col, Vec2 { x: 160.0, y: 96.0 }, 0).is_empty());
    }

    #[test]
    fn a_ceiling_anchor_above_is_classified_ceiling() {
        let col = room();
        let mut cache = AnchorCache::new();
        let a = cache.select(&col, Vec2 { x: 160.0, y: 100.0 }, 36);
        let up = a
            .iter()
            .min_by(|x, y| angle_gap(x.angle, -js::PI / 2.0).total_cmp(&angle_gap(y.angle, -js::PI / 2.0)))
            .unwrap();
        assert_eq!(up.kind, AnchorKind::Ceiling);
        assert!(up.point.y <= 32.0 + 1.0);
    }
}
