//! Skill and outcome signals (acceptance criterion 3) from the D-030 attribution, and the
//! ranking rule that turns them into skill buckets.
//!
//! Per demo and anonymous player: credited freezes, *blocks* (credited freeze that lasted
//! `block_hold_ticks` or ended in a kill), unforced self-freezes (no enemy touched the player
//! within the attribution window), times blocked, time frozen, time visible, deaths.
//!
//! **Ranking rule** ([`assign_buckets`]). A player is ranked when visible for at least
//! `min_ranked_seconds`. The score is
//! `(blocks - self_freeze_weight * self_freezes) / visible_minutes`; players are sorted by score
//! (ties: more blocks, then demo and label, so the order is deterministic). The best
//! `top_fraction` of the ranked players are `Top` - but only with at least one block, else `Mid` -
//! the next `mid_fraction` are `Mid`, the rest `Low`; players below the minimum visible time are
//! `Unranked`. "Good players" are `Top` (and, if more data is wanted, `Mid`).
//! Clients only see what is on their screen, so blocks made off-screen are not counted: the score
//! is a lower bound and favours players who fight near the recorder.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::analysis::{Attributed, Timeline};
use crate::config::Config;
use crate::types::SkillBucket;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlayerSkill {
    pub player: u16,
    pub present_ticks: i32,
    pub frozen_ticks: i32,
    /// Freeze entries of *other* players credited to this player (hook/hammer within the window).
    pub credited_freezes: u32,
    /// Credited freezes that became blocks.
    pub blocks: u32,
    /// Kill messages naming this player as the killer (the block mod credits the last toucher).
    pub kills_credited: u32,
    /// Own freeze entries with no enemy toucher within the window.
    pub self_freezes: u32,
    /// Own freeze entries with an enemy toucher (someone else was credited).
    pub blocked: u32,
    /// Kill messages naming this player as the victim.
    pub deaths: u32,
    pub blocks_per_min: f32,
    /// Mean visible seconds per life (`visible / (deaths + 1)`).
    pub survival_s: f32,
    pub score: f32,
}

impl PlayerSkill {
    pub fn present_seconds(&self, tick_rate: f32) -> f32 {
        self.present_ticks as f32 / tick_rate
    }
}

/// Server tick rate of the demos (DDNet: 50 Hz).
pub const TICK_RATE: f32 = 50.0;

pub fn compute(tl: &Timeline<'_>, attributed: &[Attributed]) -> BTreeMap<u16, PlayerSkill> {
    let cfg = tl.cfg;
    let mut out: BTreeMap<u16, PlayerSkill> = BTreeMap::new();
    for f in tl.frames {
        for c in &f.chars {
            let p = out.entry(c.player).or_insert_with(|| PlayerSkill {
                player: c.player,
                ..Default::default()
            });
            p.present_ticks += cfg.decision_ticks;
            if c.frozen() {
                p.frozen_ticks += cfg.decision_ticks;
            }
        }
    }
    for a in attributed {
        let victim = a.entry.player;
        match a.toucher {
            Some((actor, _)) => {
                if let Some(p) = out.get_mut(&victim) {
                    p.blocked += 1;
                }
                if let Some(p) = out.get_mut(&actor) {
                    p.credited_freezes += 1;
                    if a.block {
                        p.blocks += 1;
                    }
                }
            }
            None => {
                if let Some(p) = out.get_mut(&victim) {
                    p.self_freezes += 1;
                }
            }
        }
    }
    for kl in tl.kills {
        let Some(k) = tl.frame_at_or_before(kl.tick) else {
            continue;
        };
        if let Some(v) = tl.label_of_id(k, kl.victim)
            && let Some(p) = out.get_mut(&v)
        {
            p.deaths += 1;
        }
        if kl.killer >= 0
            && kl.killer != kl.victim
            && let Some(a) = tl.label_of_id(k, kl.killer)
            && let Some(p) = out.get_mut(&a)
        {
            p.kills_credited += 1;
        }
    }
    for p in out.values_mut() {
        finish(p, cfg);
    }
    out
}

/// Fills the derived rates of a [`PlayerSkill`].
pub fn finish(p: &mut PlayerSkill, cfg: &Config) {
    let minutes = (p.present_ticks as f32 / TICK_RATE / 60.0).max(1e-3);
    p.blocks_per_min = p.blocks as f32 / minutes;
    p.survival_s = p.present_ticks as f32 / TICK_RATE / (p.deaths as f32 + 1.0);
    p.score = (p.blocks as f32 - cfg.self_freeze_weight * p.self_freezes as f32) / minutes;
}

/// One row of the global ranking table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankRow {
    pub demo: u32,
    pub skill: PlayerSkill,
    pub bucket: SkillBucket,
    /// 1-based rank among the ranked players, 0 = unranked.
    pub rank: u32,
}

/// Ranks all players and assigns buckets (see the module docs). `rows` may come in any order; the
/// result is sorted by rank (unranked last, by demo and label).
pub fn assign_buckets(mut rows: Vec<RankRow>, cfg: &Config) -> Vec<RankRow> {
    let ranked = |r: &RankRow| r.skill.present_seconds(TICK_RATE) >= cfg.min_ranked_seconds;
    rows.sort_by(|a, b| {
        ranked(b)
            .cmp(&ranked(a))
            .then(b.skill.score.total_cmp(&a.skill.score))
            .then(b.skill.blocks.cmp(&a.skill.blocks))
            .then(a.demo.cmp(&b.demo))
            .then(a.skill.player.cmp(&b.skill.player))
    });
    let n = rows.iter().filter(|r| ranked(r)).count();
    let top_n = (cfg.top_fraction * n as f32).ceil() as usize;
    let mid_n = top_n + (cfg.mid_fraction * n as f32).ceil() as usize;
    for (i, r) in rows.iter_mut().enumerate() {
        if i >= n {
            r.bucket = SkillBucket::Unranked;
            r.rank = 0;
            continue;
        }
        r.rank = i as u32 + 1;
        r.bucket = if i < top_n && r.skill.blocks >= 1 {
            SkillBucket::Top
        } else if i < mid_n {
            SkillBucket::Mid
        } else {
            SkillBucket::Low
        };
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::fixtures::*;
    use crate::analysis::{Timeline, attribute, freeze_entries, hook_episodes};
    use crate::ingest::KillEvent;
    use crate::testutil::arena_with_pit;

    fn row(demo: u32, player: u16, secs: f32, blocks: u32, self_freezes: u32) -> RankRow {
        let cfg = Config::default();
        let mut s = PlayerSkill {
            player,
            present_ticks: (secs * TICK_RATE) as i32,
            blocks,
            self_freezes,
            ..Default::default()
        };
        finish(&mut s, &cfg);
        RankRow {
            demo,
            skill: s,
            bucket: SkillBucket::Unranked,
            rank: 0,
        }
    }

    #[test]
    fn signals_from_a_credited_block_and_a_self_freeze() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let mut frames = vec![
            frame(10, vec![hooking(ch(0, 1, 100.0, 250.0), 1), ch(1, 2, 300.0, 250.0)]),
            frame(12, vec![hooking(ch(0, 1, 100.0, 250.0), 1), ch(1, 2, 300.0, 250.0)]),
        ];
        // Victim (label 2) is frozen for 60 ticks; later the hooker (label 1) freezes by itself.
        for i in 0..30 {
            frames.push(frame(
                14 + 2 * i,
                vec![ch(0, 1, 100.0, 250.0), frozen(ch(1, 2, 300.0, 250.0))],
            ));
        }
        frames.push(frame(74, vec![ch(0, 1, 100.0, 250.0), ch(1, 2, 300.0, 250.0)]));
        frames.push(frame(76, vec![frozen(ch(0, 1, 100.0, 250.0)), ch(1, 2, 300.0, 250.0)]));
        let kills = [KillEvent {
            tick: 60,
            killer: 0,
            victim: 1,
            weapon: -1,
        }];
        let tl = Timeline::new(&cfg, &frames, &kills, &map);
        let at = attribute(&tl, &freeze_entries(&tl), &hook_episodes(&tl), &[]);
        let sk = compute(&tl, &at);
        let a = &sk[&1];
        let v = &sk[&2];
        assert_eq!(a.blocks, 1);
        assert_eq!(a.credited_freezes, 1);
        assert_eq!(a.self_freezes, 1);
        assert_eq!(a.kills_credited, 1);
        assert_eq!(v.blocked, 1);
        assert_eq!(v.deaths, 1);
        assert_eq!(v.self_freezes, 0);
        assert_eq!(v.frozen_ticks, 60);
        assert!(a.score < a.blocks_per_min, "the self freeze is penalised");
    }

    #[test]
    fn buckets_follow_the_documented_rule() {
        let cfg = Config::default();
        let rows = vec![
            row(0, 0, 600.0, 5, 0), // best
            row(0, 1, 600.0, 4, 1),
            row(0, 2, 600.0, 3, 0),
            row(0, 3, 600.0, 2, 0),
            row(0, 4, 600.0, 1, 0),
            row(0, 5, 600.0, 0, 3), // worst
            row(1, 6, 10.0, 9, 0),  // too short: unranked
            row(1, 7, 600.0, 0, 0),
            row(1, 8, 600.0, 0, 1),
            row(1, 9, 600.0, 2, 1),
        ];
        let out = assign_buckets(rows, &cfg);
        let by = |d: u32, p: u16| out.iter().find(|r| r.demo == d && r.skill.player == p).unwrap();
        assert_eq!(by(1, 6).bucket, SkillBucket::Unranked);
        assert_eq!(by(1, 6).rank, 0);
        assert_eq!(by(0, 0).rank, 1);
        assert_eq!(by(0, 0).bucket, SkillBucket::Top);
        assert_eq!(by(0, 5).bucket, SkillBucket::Low);
        // 9 ranked players: ceil(0.2 * 9) = 2 Top, ceil(0.4 * 9) = 4 Mid.
        assert_eq!(out.iter().filter(|r| r.bucket == SkillBucket::Top).count(), 2);
        assert_eq!(out.iter().filter(|r| r.bucket == SkillBucket::Mid).count(), 4);
        assert_eq!(out.iter().filter(|r| r.bucket == SkillBucket::Unranked).count(), 1);
        // Sorted by rank with unranked last.
        assert!(
            out.windows(2).all(
                |w| w[0].rank == 0 && w[1].rank == 0 || w[0].rank != 0 && (w[1].rank == 0 || w[0].rank < w[1].rank)
            )
        );
    }

    #[test]
    fn a_top_slot_without_blocks_becomes_mid() {
        let cfg = Config::default();
        let out = assign_buckets(vec![row(0, 0, 600.0, 0, 0), row(0, 1, 600.0, 0, 2)], &cfg);
        assert!(out.iter().all(|r| r.bucket != SkillBucket::Top));
    }

    #[test]
    fn ranking_is_deterministic_under_input_order() {
        let cfg = Config::default();
        let rows: Vec<RankRow> = (0..12)
            .map(|i| row(i / 4, (i % 4) as u16, 300.0 + i as f32, i % 3, i % 2))
            .collect();
        let mut rev = rows.clone();
        rev.reverse();
        assert_eq!(assign_buckets(rows, &cfg), assign_buckets(rev, &cfg));
    }
}
