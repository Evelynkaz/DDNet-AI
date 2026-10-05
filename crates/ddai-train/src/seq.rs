//! Sequences and training windows.
//!
//! A [`Seq`] is one contiguous run of decisions of one player (a teacher-game episode, or a run
//! of consecutive snapshots of one human player) in compact form; a [`Window`] is a slice of it
//! turned into the plain-data observations and per-step targets a learner consumes.
//!
//! **Windows.** Truncated BPTT over `len` decisions (32-64). A window starts at `anchor - r`
//! with `r` uniform in `[burn_in, len - 1]`; the first `burn_in` decisions run without loss so a
//! recurrent state has settled from its rest start (the fly's time constants are a few
//! decisions), except when the window starts at the sequence's first decision, where the rest
//! state is exact and everything is scored. Anchors are sampled proportionally to step weight
//! ([`Corpus::sample_anchor`]), so technique-tagged human steps and later DAgger rounds are drawn
//! more often; inside a window every scored step then counts equally.

use std::sync::{Arc, OnceLock};

use ddai_brain::{CharacterObservation, Observation, mirror_map_data};
use ddai_dataset::types::{ActionRec, CharRec};
use ddai_fly::bc::{HeadMask, SoftTargets, StepTargets};
use ddai_fly::rng::SplitMix64;
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;

use crate::types::{SoftRec, ring_angle_of_target};

/// A map and (lazily) its X mirror.
pub struct MapEntry {
    pub map: Arc<MapData>,
    mirrored: OnceLock<Arc<MapData>>,
}

impl MapEntry {
    pub fn new(map: Arc<MapData>) -> Arc<Self> {
        Arc::new(MapEntry {
            map,
            mirrored: OnceLock::new(),
        })
    }
    pub fn width_px(&self) -> f32 {
        self.map.width as f32 * 32.0
    }
    pub fn mirrored(&self) -> &Arc<MapData> {
        self.mirrored.get_or_init(|| Arc::new(mirror_map_data(&self.map)))
    }
}

/// Where a sequence came from (for weighting rules and reports).
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    Teacher { arena: String, round: u32 },
    Human { demo: u32 },
}

/// One decision of a sequence, compact.
#[derive(Debug, Clone)]
pub struct SeqStep {
    pub tick: i32,
    pub me: CharRec,
    pub others: Vec<CharRec>,
    pub target: i16,
    pub label: ActionRec,
    pub soft: Option<SoftRec>,
    /// Sampling weight; `0` = the step is context only (never scored, never an anchor).
    pub weight: f32,
    pub mask: HeadMask,
}

/// A contiguous run of decisions.
pub struct Seq {
    pub map: Arc<MapEntry>,
    pub steps: Vec<SeqStep>,
    pub source: Source,
}

/// A slice of a sequence as a learner sees it.
pub struct Window {
    pub observations: Vec<Observation>,
    pub targets: Vec<StepTargets>,
    /// Index (in the source sequence) of the first decision.
    pub start: usize,
    pub mirrored: bool,
}

impl Window {
    pub fn len(&self) -> usize {
        self.observations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
    /// The same window with every observation's own hook state hidden (the hook head's view under
    /// `HookView::MaskedForHookHead`); targets are unchanged.
    pub fn with_own_hook_masked(&self) -> Window {
        Window {
            observations: self.observations.iter().map(ddai_fly::bc::mask_own_hook).collect(),
            targets: self.targets.clone(),
            start: self.start,
            mirrored: self.mirrored,
        }
    }

    /// Number of decisions that carry a loss.
    pub fn scored(&self) -> usize {
        self.targets.iter().filter(|t| t.weight > 0.0).count()
    }
}

fn mirror_char(c: &CharacterObservation, width_px: f32) -> CharacterObservation {
    c.mirror_x(width_px)
}

/// The observation of `step` (mirrored through the map's centre line when `mirror`).
pub fn observation_of(step: &SeqStep, map: &MapEntry, mirror: bool) -> Observation {
    let width = map.width_px();
    let conv = |c: &CharRec| {
        let o = c.to_observation();
        if mirror { mirror_char(&o, width) } else { o }
    };
    Observation {
        map: if mirror {
            map.mirrored().clone()
        } else {
            map.map.clone()
        },
        tick: step.tick,
        self_state: conv(&step.me),
        others: step.others.iter().map(&conv).collect(),
        target_id: (step.target >= 0).then_some(i32::from(step.target)),
        tuning: TuningParams::default(),
    }
}

/// `pi - a`, wrapped to `(-pi, pi]`: the ring angle of an X-mirrored aim.
pub fn mirror_ring_angle(a: f32) -> f32 {
    let m = std::f32::consts::PI - a;
    m.sin().atan2(m.cos())
}

/// The targets of `step` for a learner. `weight` scales the step's loss; `mirror` flips the
/// direction, the left/right soft fractions and the aim angle to match a mirrored observation.
pub fn targets_of(step: &SeqStep, weight: f32, mirror: bool) -> StepTargets {
    let dir_hard = (i32::from(step.label.direction) + 1).clamp(0, 2) as u8;
    let dir_hard = if mirror { 2 - dir_hard } else { dir_hard };
    let ring = ring_angle_of_target(step.label.aim);
    let aim = if mirror { mirror_ring_angle(ring) } else { ring };
    let soft = step.soft.map(|s| SoftTargets {
        dir: if mirror {
            [s.right, s.stop, s.left]
        } else {
            [s.left, s.stop, s.right]
        },
        jump: s.jump,
        hook: s.hook,
        fire: s.fire,
    });
    StepTargets {
        dir: dir_hard,
        jump: step.label.jump,
        hook: step.label.hook,
        fire: step.label.fire,
        aim,
        soft,
        mask: step.mask,
        weight,
        hook_scale: 1.0,
    }
}

/// Builds the window of `len` decisions that ends `r` decisions after `start`'s anchor, per the
/// module docs. `burn_in` steps at the front (unless the window starts at step 0) carry no loss.
pub fn make_window(seq: &Seq, anchor: usize, len: usize, burn_in: usize, rng: &mut SplitMix64, mirror: bool) -> Window {
    let n = seq.steps.len();
    let len = len.max(1).min(n);
    let burn_in = burn_in.min(len.saturating_sub(1));
    let r = burn_in + (rng.next_u64() % (len - burn_in) as u64) as usize;
    let mut start = anchor.saturating_sub(r);
    let starts_at_beginning = start == 0;
    start = start.min(n - len);
    let starts_at_beginning = starts_at_beginning && start == 0;
    let mut observations = Vec::with_capacity(len);
    let mut targets = Vec::with_capacity(len);
    for (k, step) in seq.steps[start..start + len].iter().enumerate() {
        observations.push(observation_of(step, &seq.map, mirror));
        let scored = step.weight > 0.0 && (starts_at_beginning || k >= burn_in);
        targets.push(if scored {
            targets_of(step, 1.0, mirror)
        } else {
            StepTargets::burn_in()
        });
    }
    Window {
        observations,
        targets,
        start,
        mirrored: mirror,
    }
}

/// A set of sequences with weighted anchor sampling.
pub struct Corpus {
    pub seqs: Vec<Seq>,
    anchors: Vec<(u32, u32)>,
    cumulative: Vec<f64>,
}

impl Corpus {
    /// Builds the anchor table: every step with positive weight is an anchor.
    pub fn new(seqs: Vec<Seq>) -> Self {
        let mut anchors = Vec::new();
        let mut cumulative = Vec::new();
        let mut total = 0.0f64;
        for (si, s) in seqs.iter().enumerate() {
            for (ti, st) in s.steps.iter().enumerate() {
                if st.weight > 0.0 {
                    total += f64::from(st.weight);
                    anchors.push((si as u32, ti as u32));
                    cumulative.push(total);
                }
            }
        }
        Corpus {
            seqs,
            anchors,
            cumulative,
        }
    }

    /// A corpus for evaluation: every positive weight becomes `1`, so windows follow the natural
    /// distribution of the data rather than the training emphasis.
    pub fn uniform(mut seqs: Vec<Seq>) -> Self {
        for s in &mut seqs {
            for st in &mut s.steps {
                if st.weight > 0.0 {
                    st.weight = 1.0;
                }
            }
        }
        Corpus::new(seqs)
    }

    /// Adds sequences (their positively weighted steps become anchors) without touching the existing ones:
    /// a DAgger round grows the teacher corpus by its own episodes instead of rebuilding it from the data
    /// of every round so far.
    pub fn append(&mut self, more: Vec<Seq>) {
        let mut total = self.cumulative.last().copied().unwrap_or(0.0);
        let base = self.seqs.len();
        for (k, s) in more.iter().enumerate() {
            for (ti, st) in s.steps.iter().enumerate() {
                if st.weight > 0.0 {
                    total += f64::from(st.weight);
                    self.anchors.push(((base + k) as u32, ti as u32));
                    self.cumulative.push(total);
                }
            }
        }
        self.seqs.extend(more);
    }

    /// [`Corpus::append`] for an evaluation corpus: every positive weight becomes `1`.
    pub fn append_uniform(&mut self, mut more: Vec<Seq>) {
        for s in &mut more {
            for st in &mut s.steps {
                if st.weight > 0.0 {
                    st.weight = 1.0;
                }
            }
        }
        self.append(more);
    }

    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Number of decisions with positive weight.
    pub fn scored_steps(&self) -> usize {
        self.anchors.len()
    }

    pub fn total_weight(&self) -> f64 {
        self.cumulative.last().copied().unwrap_or(0.0)
    }

    /// An anchor drawn proportionally to step weight. Panics on an empty corpus.
    pub fn sample_anchor(&self, rng: &mut SplitMix64) -> (usize, usize) {
        let x = f64::from(rng.next_f32_unit()) * self.total_weight();
        let i = self.cumulative.partition_point(|&c| c <= x).min(self.anchors.len() - 1);
        let (s, t) = self.anchors[i];
        (s as usize, t as usize)
    }

    /// A training window around a sampled anchor.
    pub fn sample_window(&self, rng: &mut SplitMix64, len: usize, burn_in: usize, mirror: bool) -> Window {
        let (s, t) = self.sample_anchor(rng);
        make_window(&self.seqs[s], t, len, burn_in, rng, mirror)
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;
    use ddai_brain::CharacterObservation;
    use ddai_physics::map::{MapData, TILE_SOLID, Tile};

    pub fn room_map() -> Arc<MapEntry> {
        let (w, h) = (30usize, 20usize);
        let mut game = vec![Tile::default(); w * h];
        for y in 0..h {
            for x in 0..w {
                if y >= 14 || x == 0 || x == w - 1 || y == 0 || (x == 6 && y < 14 && y > 8) {
                    game[y * w + x] = Tile {
                        index: TILE_SOLID,
                        ..Tile::default()
                    };
                }
            }
        }
        MapEntry::new(Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }))
    }

    pub fn step(t: i32, weight: f32) -> SeqStep {
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(200.0 + t as f32, 400.0);
        me.vel = ddai_physics::vmath::Vec2::new(3.0, -1.0);
        let mut opp = CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(500.0, 380.0);
        SeqStep {
            tick: t * 2,
            me: crate::types::char_rec(&me),
            others: vec![crate::types::char_rec(&opp)],
            target: 1,
            label: ActionRec {
                direction: 1,
                jump: t % 2 == 0,
                hook: t % 3 == 0,
                fire: false,
                aim: [300, -100],
            },
            soft: Some(SoftRec {
                left: 0.1,
                stop: 0.2,
                right: 0.7,
                jump: 0.5,
                hook: 0.4,
                fire: 0.0,
                aim_mean: 0.3,
                aim_spread: 0.2,
            }),
            weight,
            mask: HeadMask::ALL,
        }
    }

    pub fn seq(n: usize) -> Seq {
        Seq {
            map: room_map(),
            steps: (0..n as i32).map(|t| step(t, 1.0)).collect(),
            source: Source::Teacher {
                arena: "room".to_string(),
                round: 0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    #[test]
    fn windows_have_the_requested_length_and_burn_in_steps_carry_no_loss() {
        let s = seq(100);
        let mut rng = SplitMix64::new(1);
        for anchor in [0usize, 5, 50, 99] {
            let w = make_window(&s, anchor, 32, 6, &mut rng, false);
            assert_eq!(w.len(), 32);
            assert!(w.start + 32 <= 100);
            if w.start > 0 {
                assert!(
                    w.targets[..6].iter().all(|t| t.weight == 0.0),
                    "burn-in unscored (anchor {anchor})"
                );
                assert!(w.targets[6..].iter().all(|t| t.weight > 0.0));
            } else {
                assert!(
                    w.targets.iter().all(|t| t.weight > 0.0),
                    "a window at the start scores everything"
                );
            }
        }
    }

    #[test]
    fn a_short_sequence_gives_a_short_window_and_zero_weight_steps_are_never_scored() {
        let mut s = seq(10);
        s.steps[3].weight = 0.0;
        let mut rng = SplitMix64::new(2);
        let w = make_window(&s, 5, 32, 6, &mut rng, false);
        assert_eq!((w.len(), w.start), (10, 0));
        assert_eq!(w.targets[3].weight, 0.0);
        assert_eq!(w.scored(), 9);
    }

    #[test]
    fn appending_sequences_equals_building_the_corpus_from_all_of_them() {
        let all: Vec<Seq> = vec![testutil::seq(20), testutil::seq(35), testutil::seq(12)];
        let rebuilt = Corpus::new(vec![testutil::seq(20), testutil::seq(35), testutil::seq(12)]);
        let mut grown = Corpus::new(vec![testutil::seq(20)]);
        grown.append(all.into_iter().skip(1).collect());
        assert_eq!(
            (grown.scored_steps(), grown.seqs.len()),
            (rebuilt.scored_steps(), rebuilt.seqs.len())
        );
        assert!((grown.total_weight() - rebuilt.total_weight()).abs() < 1e-9);
        let (mut a, mut b) = (SplitMix64::new(5), SplitMix64::new(5));
        for _ in 0..200 {
            assert_eq!(grown.sample_anchor(&mut a), rebuilt.sample_anchor(&mut b));
        }
        let mut ev = Corpus::uniform(vec![testutil::seq(10)]);
        let mut heavy = testutil::seq(10);
        heavy.steps.iter_mut().for_each(|s| s.weight = 7.0);
        ev.append_uniform(vec![heavy]);
        assert_eq!(ev.total_weight(), 20.0, "appended evaluation sequences are uniform too");
    }

    #[test]
    fn anchors_follow_the_step_weights() {
        let mut s = seq(4);
        s.steps[0].weight = 0.0;
        s.steps[1].weight = 1.0;
        s.steps[2].weight = 3.0;
        s.steps[3].weight = 0.0;
        let c = Corpus::new(vec![s]);
        assert_eq!(c.scored_steps(), 2);
        assert!((c.total_weight() - 4.0).abs() < 1e-9);
        let mut rng = SplitMix64::new(3);
        let mut counts = [0usize; 4];
        for _ in 0..8000 {
            counts[c.sample_anchor(&mut rng).1] += 1;
        }
        assert_eq!((counts[0], counts[3]), (0, 0), "zero-weight steps are never anchors");
        let ratio = counts[2] as f64 / counts[1] as f64;
        assert!((ratio - 3.0).abs() < 0.5, "ratio {ratio}");
    }

    #[test]
    fn sampling_is_deterministic_in_the_rng_seed() {
        let c = Corpus::new(vec![seq(60), seq(40)]);
        let run = |seed| {
            let mut rng = SplitMix64::new(seed);
            (0..5)
                .map(|_| c.sample_window(&mut rng, 16, 4, false).start)
                .collect::<Vec<_>>()
        };
        assert_eq!(run(9), run(9));
    }

    #[test]
    fn mirroring_flips_direction_aim_and_soft_fractions_and_is_an_involution() {
        let s = seq(3);
        let st = &s.steps[1];
        let plain = targets_of(st, 1.0, false);
        let m = targets_of(st, 1.0, true);
        assert_eq!((plain.dir, m.dir), (2, 0));
        let (sp, sm) = (plain.soft.unwrap(), m.soft.unwrap());
        assert_eq!((sp.dir[0], sp.dir[2]), (sm.dir[2], sm.dir[0]));
        assert!((mirror_ring_angle(plain.aim) - m.aim).abs() < 1e-6);
        assert!((mirror_ring_angle(mirror_ring_angle(plain.aim)) - plain.aim).abs() < 1e-5);
        assert_eq!((plain.jump, plain.hook, plain.fire), (m.jump, m.hook, m.fire));

        // The mirrored observation is the observation mirrored through the map's centre line.
        let o = observation_of(st, &s.map, false);
        let om = observation_of(st, &s.map, true);
        assert_eq!(om.self_state.pos.x, s.map.width_px() - o.self_state.pos.x);
        assert_eq!(om.self_state.vel.x, -o.self_state.vel.x);
        assert_eq!(om.others[0].pos.y, o.others[0].pos.y);
        // A mirrored aim vector's ring angle is `pi - a`.
        let aim = st.label.aim;
        let ring = ring_angle_of_target(aim);
        let mirrored_target = [-aim[0], aim[1]];
        assert!((ring_angle_of_target(mirrored_target) - mirror_ring_angle(ring)).abs() < 1e-5);
    }
}
