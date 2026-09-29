//! One demo end to end: anonymised frames + map -> frames, `(Observation, Action)` samples with
//! tags and quality flags, skill signals and technique events.

use std::collections::BTreeMap;
use std::sync::Arc;

use ddai_physics::map::MapData;
use serde::{Deserialize, Serialize};

use crate::analysis::{self, Timeline};
use crate::config::Config;
use crate::ingest::Ingested;
use crate::pipeline::{self, BuildCounters};
use crate::replay::ReplayStats;
use crate::skill::{self, PlayerSkill};
use crate::tags::signal;
use crate::technique::{self, Ctx, TechniqueEvent};
use crate::types::{FrameRec, SampleRec, quality};

/// Everything one demo contributes to the dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DemoOutput {
    pub frames: Vec<FrameRec>,
    pub samples: Vec<SampleRec>,
    pub players: Vec<PlayerSkill>,
    pub replay: BTreeMap<u16, ReplayStats>,
    pub events: Vec<TechniqueEvent>,
    pub counters: BuildCounters,
    pub freeze_entries: u32,
    pub credited_freezes: u32,
    pub blocks: u32,
    pub hook_episodes: u32,
    pub hammer_hits: u32,
    /// Frames with non-default tuning (see `Ingested::tune_nondefault_frames`).
    pub tune_nondefault_frames: u32,
    /// Frames with movement-relevant non-default tuning.
    pub tune_movement_frames: u32,
}

/// The `target_id` rule (documented in `docs/formats.md` §20):
///
/// 1. the character this one is hooking (a human who hooks someone is busy with that target);
/// 2. else the nearest character hooking this one (the threat);
/// 3. else the nearest *free* (not frozen) other within `target_range` (the opponent to fight);
/// 4. else the nearest free other at any distance;
/// 5. else the nearest other (frozen);
/// 6. else none (`-1`).
///
/// Distances are between centres; ties go to the lower client id.
pub fn choose_target(frame: &FrameRec, slot: usize, target_range: f32) -> i16 {
    let me = &frame.chars[slot];
    let dist = |o: &crate::types::CharRec| ((o.pos[0] - me.pos[0]).powi(2) + (o.pos[1] - me.pos[1]).powi(2)).sqrt();
    if me.hooked_player >= 0
        && me.hook_state == analysis::HOOK_GRABBED
        && frame
            .chars
            .iter()
            .any(|o| i16::from(o.id) == me.hooked_player && o.id != me.id)
    {
        return me.hooked_player;
    }
    let nearest = |pred: &dyn Fn(&crate::types::CharRec) -> bool| -> Option<i16> {
        frame
            .chars
            .iter()
            .filter(|o| o.id != me.id && pred(o))
            .min_by(|a, b| dist(a).total_cmp(&dist(b)).then(a.id.cmp(&b.id)))
            .map(|o| i16::from(o.id))
    };
    nearest(&|o| o.hook_state == analysis::HOOK_GRABBED && o.hooked_player == i16::from(me.id))
        .or_else(|| nearest(&|o| !o.frozen() && dist(o) <= target_range))
        .or_else(|| nearest(&|o| !o.frozen()))
        .or_else(|| nearest(&|_| true))
        .unwrap_or(-1)
}

/// Runs the whole per-demo pipeline. The skill bucket of the samples is left at `Unranked`; it is
/// assigned after the global ranking.
pub fn process(cfg: &Config, map: &Arc<MapData>, ing: &Ingested) -> DemoOutput {
    let built = pipeline::build(cfg, map, ing);
    let tl = Timeline::new(cfg, &built.frames, &ing.kills, map);
    let entries = analysis::freeze_entries(&tl);
    let hooks = analysis::hook_episodes(&tl);
    let hits = analysis::hammer_hits(&tl);
    let attr = analysis::attribute(&tl, &entries, &hooks, &hits);
    let skills = skill::compute(&tl, &attr);
    let ctx = Ctx {
        tl: &tl,
        entries: &entries,
        hooks: &hooks,
        hits: &hits,
        attr: &attr,
        tc: &cfg.technique,
    };
    let events = technique::detect(&ctx);

    // Tag bits per (frame, slot).
    let mut tag_bits: Vec<Vec<u32>> = built.frames.iter().map(|f| vec![0u32; f.chars.len()]).collect();
    let mut tag_window = |label: u16, from: i32, to: i32, bit: u32| {
        let lo = built.frames.partition_point(|f| f.tick < from);
        let hi = built.frames.partition_point(|f| f.tick <= to);
        for (k, bits) in tag_bits.iter_mut().enumerate().take(hi).skip(lo) {
            if let Some(s) = tl.slot(k, label) {
                bits[s] |= bit;
            }
        }
    };
    for e in events.iter().filter(|e| e.active) {
        tag_window(e.actor, e.start_tick, e.end_tick, e.technique.bit());
    }
    for a in &attr {
        let t = a.entry.tick;
        let lead = cfg.signal_lead_ticks;
        match a.toucher {
            Some((actor, _)) => {
                if a.block {
                    tag_window(actor, t - lead, t, signal::LEADS_TO_BLOCK);
                }
                tag_window(a.entry.player, t - lead, t, signal::LEADS_TO_BLOCKED);
            }
            None => tag_window(a.entry.player, t - lead, t, signal::LEADS_TO_SELF_FREEZE),
        }
    }

    let mut samples = Vec::new();
    for (k, frame) in built.frames.iter().enumerate() {
        for (slot, step) in built.steps[k].iter().enumerate() {
            let Some(st) = step else { continue };
            let mut q = st.replay as u8 & quality::REPLAY_MASK;
            if st.next_fresh {
                q |= quality::NEXT_FRESH;
            }
            if st.active {
                q |= quality::ACTIVE;
            }
            samples.push(SampleRec {
                frame: k as u32,
                slot: slot as u8,
                action: st.action,
                target: choose_target(frame, slot, cfg.target_range),
                tags: tag_bits[k][slot],
                q,
                skill: 0,
            });
        }
    }

    DemoOutput {
        samples,
        players: skills.into_values().collect(),
        replay: built.replay,
        events,
        counters: built.counters,
        freeze_entries: entries.len() as u32,
        credited_freezes: attr.iter().filter(|a| a.toucher.is_some()).count() as u32,
        blocks: attr.iter().filter(|a| a.block).count() as u32,
        hook_episodes: hooks.len() as u32,
        hammer_hits: hits.len() as u32,
        tune_nondefault_frames: ing.tune_nondefault_frames as u32,
        tune_movement_frames: ing.tune_movement_frames as u32,
        frames: built.frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::fixtures::*;

    #[test]
    fn target_rule_priorities() {
        // Self (id 0) at origin; id 1 frozen and close; id 2 free at 300; id 3 free at 200 but
        // 4 hooks me from far away.
        let me = ch(0, 1, 0.0, 0.0);
        let f = |chars: Vec<crate::types::CharRec>| frame(10, chars);
        let frame_a = f(vec![
            me,
            frozen(ch(1, 2, 30.0, 0.0)),
            ch(2, 3, 300.0, 0.0),
            ch(3, 4, 200.0, 0.0),
        ]);
        assert_eq!(
            choose_target(&frame_a, 0, 380.0),
            3,
            "nearest free opponent, not the frozen one"
        );
        // A hooker outranks a nearer free opponent.
        let hooker = hooking(ch(4, 5, 350.0, 0.0), 0);
        let frame_b = f(vec![me, ch(3, 4, 100.0, 0.0), hooker]);
        assert_eq!(choose_target(&frame_b, 0, 380.0), 4);
        // Whoever I hook comes first of all.
        let frame_c = f(vec![hooking(me, 2), ch(2, 3, 300.0, 0.0), ch(3, 4, 50.0, 0.0), hooker]);
        assert_eq!(choose_target(&frame_c, 0, 380.0), 2);
        // Only frozen others: nearest frozen. Alone: none.
        let frame_d = f(vec![me, frozen(ch(1, 2, 60.0, 0.0)), frozen(ch(2, 3, 30.0, 0.0))]);
        assert_eq!(choose_target(&frame_d, 0, 380.0), 2);
        assert_eq!(choose_target(&f(vec![me]), 0, 380.0), -1);
        // Out of range free opponent is still preferred over a frozen one (rule 4 before 5).
        let frame_e = f(vec![me, frozen(ch(1, 2, 10.0, 0.0)), ch(2, 3, 900.0, 0.0)]);
        assert_eq!(choose_target(&frame_e, 0, 380.0), 2);
    }

    #[test]
    fn hook_to_a_departed_character_does_not_pick_a_ghost() {
        let me = hooking(ch(0, 1, 0.0, 0.0), 7);
        let f = frame(10, vec![me, ch(1, 2, 100.0, 0.0)]);
        assert_eq!(choose_target(&f, 0, 380.0), 1);
    }
}
