//! One demo end to end: anonymised frames + map -> frames, `(Observation, Action)` samples with
//! tags and quality flags, skill signals and technique events.
//!
//! Memory is bounded (task 8.4d): the frames stream through [`pipeline::build_stream`] into two
//! [`crate::store`] page stores (frames, and the samples of every frame without their tags), the
//! analyses read the frames back through a small page cache, and the tag bits, which depend on the
//! analyses' events, are applied when the dataset chunks are written ([`DemoOutput::tag_sweeper`]).
//! What stays in memory is the events, the players and the counters: a few thousand small records.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use ddai_physics::map::MapData;

use crate::analysis::{self, Timeline};
use crate::config::Config;
use crate::humaninput::TrueTable;
use crate::ingest::{FrameSource, Ingested, IngestedFrame, KillEvent};
use crate::pipeline::{self, BuildCounters, FrameOut};
use crate::replay::ReplayStats;
use crate::skill::{self, PlayerSkill};
use crate::store::{FrameStore, FrameStoreWriter, PagedStore, PagedWriter, Spill};
use crate::tags::signal;
use crate::technique::{self, Ctx, TechniqueEvent};
use crate::types::{FrameRec, SampleRec, quality};

/// A range of ticks in which every frame's character `label` carries the tag bit `bit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagWindow {
    pub label: u16,
    pub from: i32,
    pub to: i32,
    pub bit: u32,
}

/// Everything one demo contributes to the dataset. The frames and the tag-less samples live in
/// the spill file of the demo's stores (see the module docs) and are read by the dataset writer.
#[derive(Debug)]
pub struct DemoOutput {
    /// All frames, in order.
    pub frames: FrameStore,
    /// The samples of each frame (index = frame index), without tags and skill bucket.
    pub samples: PagedStore<Vec<SampleRec>>,
    pub frame_count: usize,
    pub sample_count: usize,
    /// Ticks of the first and last frame.
    pub first_tick: i32,
    pub last_tick: i32,
    /// Tag windows, sorted by start tick.
    pub tag_windows: Vec<TagWindow>,
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
    /// The demo ended early on a decode error.
    pub decode_error: bool,
    /// Facts about the input reconstruction pass (fire events, the longest registration delay,
    /// re-registered `(client id, tick)` pairs).
    pub recon: pipeline::ReconStats,
    /// Frame ticks were not ascending somewhere (windows by tick assume they are).
    pub ticks_ascending: bool,
    /// Bytes the demo's pages take in the spill file.
    pub spill_bytes: u64,
}

impl DemoOutput {
    /// Applies the tag windows to frames visited in order.
    pub fn tag_sweeper(&self) -> TagSweeper<'_> {
        TagSweeper {
            windows: &self.tag_windows,
            next: 0,
            active: Vec::new(),
        }
    }
}

/// Computes the tag bits of each character of the frames it is fed in order (tick ascending):
/// the OR of the bits of every window `[from, to]` that contains the frame's tick and names a
/// label present in the frame.
pub struct TagSweeper<'w> {
    windows: &'w [TagWindow],
    next: usize,
    active: Vec<TagWindow>,
}

impl TagSweeper<'_> {
    pub fn bits(&mut self, frame: &FrameRec) -> Vec<u32> {
        while self.next < self.windows.len() && self.windows[self.next].from <= frame.tick {
            self.active.push(self.windows[self.next]);
            self.next += 1;
        }
        self.active.retain(|w| w.to >= frame.tick);
        let mut bits = vec![0u32; frame.chars.len()];
        for w in &self.active {
            if let Some(slot) = frame.slot_of(w.label) {
                bits[slot] |= w.bit;
            }
        }
        bits
    }
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

/// Runs the whole per-demo pipeline on an in-memory demo (tests and small inputs); the skill
/// bucket of the samples is left at `Unranked`, it is assigned after the global ranking.
pub fn process(cfg: &Config, map: &Arc<MapData>, ing: &Ingested) -> io::Result<DemoOutput> {
    let frames = || ing.frames_with_tunes();
    process_frames(cfg, map, None, frames, None, |_| Tail {
        kills: ing.kills.clone(),
        tune_nondefault_frames: ing.tune_nondefault_frames as u32,
        tune_movement_frames: ing.tune_movement_frames as u32,
        decode_error: ing.decode_error.is_some(),
    })
}

/// Runs the whole per-demo pipeline on a demo in bounded memory: the frames are streamed out of
/// the demo twice ([`FrameSource`]; see [`pipeline::recon_table`]) and spilled to a temporary file
/// in `spill_dir` (the system temporary directory when `None`).
pub fn process_source(
    cfg: &Config,
    map: &Arc<MapData>,
    demo: &ddai_demo::Demo<'_>,
    spill_dir: Option<&Path>,
) -> io::Result<DemoOutput> {
    process_source_real(cfg, map, demo, spill_dir, None)
}

/// [`process_source`] with the demo's real inputs (`Sv_PreInput`, task 3.24): where a player's
/// track has an input in force the sample carries it (flagged [`quality::REAL_INPUT`]); the rest
/// stays reconstructed. `None` = exactly [`process_source`].
pub fn process_source_real(
    cfg: &Config,
    map: &Arc<MapData>,
    demo: &ddai_demo::Demo<'_>,
    spill_dir: Option<&Path>,
    real: Option<&TrueTable>,
) -> io::Result<DemoOutput> {
    process_frames(
        cfg,
        map,
        real,
        || FrameSource::new(demo).min_spacing(cfg.min_frame_spacing),
        spill_dir,
        |src| Tail {
            kills: src.kills,
            tune_nondefault_frames: src.tune_nondefault_frames as u32,
            tune_movement_frames: src.tune_movement_frames as u32,
            decode_error: src.decode_error.is_some(),
        },
    )
}

/// What is only known once the frames have all been read.
pub struct Tail {
    pub kills: Vec<KillEvent>,
    pub tune_nondefault_frames: u32,
    pub tune_movement_frames: u32,
    pub decode_error: bool,
}

/// `frames` is called twice and must yield the same frames each time; `tail` gets the iterator of
/// the second pass after it has been drained.
fn process_frames<I: Iterator<Item = IngestedFrame>>(
    cfg: &Config,
    map: &Arc<MapData>,
    real: Option<&TrueTable>,
    mut frames: impl FnMut() -> I,
    spill_dir: Option<&Path>,
    tail: impl FnOnce(I) -> Tail,
) -> io::Result<DemoOutput> {
    // --- pass 0: the whole-demo facts of the input reconstruction ---
    let table = pipeline::recon_table(frames());

    // --- pass 1: stream the demo through the pipeline into the spilled stores ---
    let spill = Spill::create(spill_dir)?;
    let mut frame_w = FrameStoreWriter::new(Arc::clone(&spill));
    let mut sample_w: PagedWriter<Vec<SampleRec>> = PagedWriter::new(Arc::clone(&spill));
    let mut sample_count = 0usize;
    let mut k = 0u32;
    let mut frames = frames();
    let summary = pipeline::build_stream_real(
        cfg,
        map,
        &table,
        real,
        frames.by_ref(),
        |out: FrameOut| -> io::Result<()> {
            let samples = samples_of(cfg, k, &out);
            sample_count += samples.len();
            frame_w.push(out.frame)?;
            sample_w.push(samples)?;
            k += 1;
            Ok(())
        },
    )?;
    let tail = tail(frames);
    let store = frame_w.finish()?;
    let sample_store = sample_w.finish()?;
    let (first_tick, last_tick) = match (store.ticks().first(), store.ticks().last()) {
        (Some(&a), Some(&b)) => (a, b),
        _ => (0, 0),
    };
    let ticks_ascending = store.ticks().windows(2).all(|w| w[0] <= w[1]);

    // --- pass 2: analyses over the stored timeline ---
    let tl = Timeline::new(cfg, &store, &tail.kills, map);
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

    // Tag windows; the bits are applied when the chunks are written.
    let mut tag_windows: Vec<TagWindow> = Vec::new();
    for e in events.iter().filter(|e| e.active) {
        tag_windows.push(TagWindow {
            label: e.actor,
            from: e.start_tick,
            to: e.end_tick,
            bit: e.technique.bit(),
        });
    }
    for a in &attr {
        let t = a.entry.tick;
        let lead = cfg.signal_lead_ticks;
        let mut window = |label: u16, bit: u32| {
            tag_windows.push(TagWindow {
                label,
                from: t - lead,
                to: t,
                bit,
            });
        };
        match a.toucher {
            Some((actor, _)) => {
                if a.block {
                    window(actor, signal::LEADS_TO_BLOCK);
                }
                window(a.entry.player, signal::LEADS_TO_BLOCKED);
            }
            None => window(a.entry.player, signal::LEADS_TO_SELF_FREEZE),
        }
    }
    tag_windows.sort_by_key(|w| w.from);

    let spill_bytes = store.stored_bytes() + sample_store.stored_bytes();
    let frame_count = store.len();
    drop(tl);
    Ok(DemoOutput {
        frames: store,
        samples: sample_store,
        frame_count,
        sample_count,
        first_tick,
        last_tick,
        tag_windows,
        players: skills.into_values().collect(),
        replay: summary.replay,
        events,
        counters: summary.counters,
        freeze_entries: entries.len() as u32,
        credited_freezes: attr.iter().filter(|a| a.toucher.is_some()).count() as u32,
        blocks: attr.iter().filter(|a| a.block).count() as u32,
        hook_episodes: hooks.len() as u32,
        hammer_hits: hits.len() as u32,
        tune_nondefault_frames: tail.tune_nondefault_frames,
        tune_movement_frames: tail.tune_movement_frames,
        decode_error: tail.decode_error,
        recon: summary.recon,
        ticks_ascending,
        spill_bytes,
    })
}

/// The samples of one finished frame, without tags and skill bucket.
fn samples_of(cfg: &Config, k: u32, out: &FrameOut) -> Vec<SampleRec> {
    let mut samples = Vec::new();
    for (slot, step) in out.steps.iter().enumerate() {
        let Some(st) = step else { continue };
        let mut q = st.replay as u8 & quality::REPLAY_MASK;
        if st.next_fresh {
            q |= quality::NEXT_FRESH;
        }
        if st.active {
            q |= quality::ACTIVE;
        }
        if st.real {
            q |= quality::REAL_INPUT;
        }
        samples.push(SampleRec {
            frame: k,
            slot: slot as u8,
            action: st.action,
            target: choose_target(&out.frame, slot, cfg.target_range),
            tags: 0,
            q,
            skill: 0,
        });
    }
    samples
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
