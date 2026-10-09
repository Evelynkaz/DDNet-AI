//! Samples from arena games and live clips, their input vectors and their labels.
//!
//! A **sample** is a decision tick `T`: the arena's `T % decide_every == 0` (with the game's own lag as the window), a clip's duel frame (window
//! [`CorpusCfg::clip_lag`]) or a human game's duel frame (task 3.24, the same window). Its input is built by [`assemble`], the same function the live predictor
//! uses. The label says what the opponent did at `T ..`: the arena and the human games know every applied input (a human game's are the real ones from the
//! demo's pre-inputs), a clip only what the next snapshots show (see [`crate::clipdata`]), so labels carry a mask per head.

use rayon::prelude::*;

use super::data::GameRec;
use super::feature::{
    FD, HORIZON, IF_DIM, IF_SLOTS, INPUT_DIM, K_HIST, KnownTick, Label, STRIDE, assemble, frame_features,
    inflight_features, label_tick,
};
use crate::clipdata::{ClipGame, labels_at};
use crate::frame::{InputRec, TeeFrame};
use crate::humandata::HumanGame;

#[derive(Debug, Clone)]
pub struct CorpusCfg {
    /// The window length of clip samples (the live window).
    pub clip_lag: usize,
    /// Probabilities of `n = 0 ..= 4` known ticks in an arena sample (the pre-input simulation); must sum to 1.
    pub known_p: [f32; 5],
    /// Ablation: only the newest `hist_keep` history frames are read, the older slots repeat the oldest kept one (default [`K_HIST`]).
    pub hist_keep: usize,
    /// Arena fire label: a real **swing** (the opponent's `attack_age` is 1 in the frame after the step: the weapon went off) instead of a fresh press of the fire
    /// counter (a press during the reload does nothing). Clips can only show swings, so this makes the two sources teach the same thing (3.21 review F3).
    pub swing_label: bool,
}

impl Default for CorpusCfg {
    fn default() -> Self {
        CorpusCfg {
            clip_lag: 2,
            known_p: [1.0, 0.0, 0.0, 0.0, 0.0],
            hist_keep: K_HIST,
            swing_label: false,
        }
    }
}

/// A sample's place: source (0 arena, 1 clip, 2 human game), game and tick index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRef {
    pub src: u8,
    pub game: u32,
    pub idx: u32,
}

pub struct Corpus {
    pub arena: Vec<GameRec>,
    pub clips: Vec<ClipGame>,
    /// Games of real humans from demos (source 2); see [`Corpus::with_humans`].
    pub humans: Vec<HumanGame>,
    af: Vec<Vec<[f32; FD]>>,
    cf: Vec<Vec<[f32; FD]>>,
    hf: Vec<Vec<[f32; FD]>>,
    pub cfg: CorpusCfg,
}

/// Our input applied in the step into world tick `tick`, for the human-game sample at frame `i`: the real one when the demo has it, else what the frame that
/// ends the step shows of our own tee (direction, jump key as the jump bit, hook key as a hook above idle, aim as its angle; no fire).
fn human_us_input(g: &HumanGame, i: usize, tick: i32) -> InputRec {
    if let Some(r) = g.input_at(0, tick) {
        return r;
    }
    let m = (i..g.ticks.len())
        .find(|&m| g.ticks[m].tick >= tick)
        .unwrap_or(g.ticks.len() - 1);
    let f = &g.ticks[m].frames[0];
    let a = f64::from(f.angle);
    InputRec {
        direction: f.direction,
        jump: f.jumped & 1 != 0,
        hook: f.hook_state > 0,
        fire: 0,
        target_x: (ddai_jsmath::cos(a) * 1000.0) as i16,
        target_y: (ddai_jsmath::sin(a) * 1000.0) as i16,
    }
}

fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ c.wrapping_mul(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Corpus {
    pub fn new(arena: Vec<GameRec>, clips: Vec<ClipGame>, cfg: CorpusCfg) -> Corpus {
        let af = arena
            .par_iter()
            .map(|g| {
                g.ticks
                    .iter()
                    .map(|t| {
                        let mut f = [0.0; FD];
                        frame_features(&t.frames[0], &t.frames[1], &mut f);
                        f
                    })
                    .collect()
            })
            .collect();
        let cf = clips
            .par_iter()
            .map(|g| {
                g.ticks
                    .iter()
                    .map(|t| {
                        let mut f = [0.0; FD];
                        frame_features(&t.frames[0], &t.frames[1], &mut f);
                        f
                    })
                    .collect()
            })
            .collect();
        Corpus {
            arena,
            clips,
            humans: Vec::new(),
            af,
            cf,
            hf: Vec::new(),
            cfg,
        }
    }

    /// Adds games of real humans (source 2) to the corpus.
    pub fn with_humans(mut self, humans: Vec<HumanGame>) -> Corpus {
        self.hf = humans
            .par_iter()
            .map(|g| {
                g.ticks
                    .iter()
                    .map(|t| {
                        let mut f = [0.0; FD];
                        frame_features(&t.frames[0], &t.frames[1], &mut f);
                        f
                    })
                    .collect()
            })
            .collect();
        self.humans = humans;
        self
    }

    /// Every usable decision tick.
    pub fn samples(&self) -> Vec<SampleRef> {
        let mut out = Vec::new();
        for (gi, g) in self.arena.iter().enumerate() {
            let lag = usize::from(g.lag[0]);
            if lag == 0 || lag > super::feature::MAX_LAG {
                continue;
            }
            let de = i32::from(g.decide_every.max(1));
            for i in 0..g.ticks.len() {
                let t = g.tick0 + i as i32;
                if t % de == 0 && i + lag < g.ticks.len() && i + 1 < g.ticks.len() {
                    out.push(SampleRef {
                        src: 0,
                        game: gi as u32,
                        idx: i as u32,
                    });
                }
            }
        }
        for (gi, g) in self.clips.iter().enumerate() {
            for i in 0..g.ticks.len() {
                if self.clip_sample_ok(g, i) {
                    out.push(SampleRef {
                        src: 1,
                        game: gi as u32,
                        idx: i as u32,
                    });
                }
            }
        }
        for (gi, g) in self.humans.iter().enumerate() {
            for i in 0..g.ticks.len() {
                if self.human_sample_ok(g, i) {
                    out.push(SampleRef {
                        src: 2,
                        game: gi as u32,
                        idx: i as u32,
                    });
                }
            }
        }
        out
    }

    /// A human-game frame is a sample when it is a duel frame, both tees are alive and free, the next frame follows and the opponent's real input just before
    /// it is known. Our own inputs for the window need not be: the player who recorded a demo has none in it (the server forwards the others' only), so
    /// they are then read off the frames ([`human_us_input`]).
    fn human_sample_ok(&self, g: &HumanGame, i: usize) -> bool {
        let t = &g.ticks[i];
        if !t.duel
            || !t.frames[0].alive
            || !t.frames[1].alive
            || t.frames[0].freeze_left > 0
            || t.frames[1].freeze_left > 0
        {
            return false;
        }
        if !g.consecutive(i, 0, 1) || !g.ticks[i + 1].frames[1].alive {
            return false;
        }
        g.input_at(1, t.tick).is_some()
    }

    /// A clip frame is a sample when it is a duel frame, both tees are alive and free, the next frame follows, and our inputs for the window are known.
    fn clip_sample_ok(&self, g: &ClipGame, i: usize) -> bool {
        let t = &g.ticks[i];
        if !t.duel
            || !t.frames[0].alive
            || !t.frames[1].alive
            || t.frames[0].freeze_left > 0
            || t.frames[1].freeze_left > 0
        {
            return false;
        }
        if !g.consecutive(i, 0, 1) || !g.ticks[i + 1].frames[1].alive {
            return false;
        }
        (1..=self.cfg.clip_lag as i32).all(|k| g.sent_at(i, t.tick + k).is_some())
    }

    /// How many ticks of the window are known (pre-inputs) in this sample at this epoch: always 0 for a clip.
    pub fn known_n(&self, r: SampleRef, salt: u64) -> usize {
        if r.src == 1 {
            return 0;
        }
        let u = (mix(u64::from(r.game), u64::from(r.idx), salt) >> 40) as f32 / (1u64 << 24) as f32;
        let mut acc = 0.0;
        for (n, p) in self.cfg.known_p.iter().enumerate() {
            acc += p;
            if u < acc {
                return n;
            }
        }
        0
    }

    /// The window length of a sample.
    pub fn lag_of(&self, r: SampleRef) -> usize {
        if r.src == 0 {
            usize::from(self.arena[r.game as usize].lag[0])
        } else {
            self.cfg.clip_lag
        }
    }

    /// Builds the input and the label (masked by the known ticks) of a sample for the epoch `salt`.
    pub fn make(&self, r: SampleRef, salt: u64, x: &mut [f32; INPUT_DIM]) -> Label {
        let lag = self.lag_of(r);
        let n_if = lag.min(IF_SLOTS);
        let i = r.idx as usize;
        let mut inflight = [[0.0f32; IF_DIM]; IF_SLOTS];
        let mut known = [None; HORIZON];
        let mut label;
        if r.src == 0 {
            let g = &self.arena[r.game as usize];
            let feats = &self.af[r.game as usize];
            for (k, f) in inflight.iter_mut().enumerate().take(n_if) {
                inflight_features(&g.ticks[i + 1 + k].applied[0], f);
            }
            let n = self.known_n(r, salt);
            for (k, slot) in known.iter_mut().enumerate().take(n) {
                if let Some(t) = g.ticks.get(i + 1 + k) {
                    *slot = Some(KnownTick::from_rec(&t.applied[1]));
                }
            }
            assemble(
                x,
                |j| &feats[i.saturating_sub(j.min(self.cfg.hist_keep.max(1) - 1) * STRIDE)],
                &g.ticks[i].rays[1],
                &g.ticks[i].rays[0],
                &inflight[..n_if],
                lag,
                &known,
            );
            label = self.label(r);
            label.mask_known(n);
        } else if r.src == 2 {
            let g = &self.humans[r.game as usize];
            let feats = &self.hf[r.game as usize];
            let t = &g.ticks[i];
            for (k, f) in inflight.iter_mut().enumerate().take(n_if) {
                let rec = human_us_input(g, i, t.tick + 1 + k as i32);
                inflight_features(&rec, f);
            }
            let n = self.known_n(r, salt);
            for (k, slot) in known.iter_mut().enumerate().take(n) {
                *slot = g
                    .input_at(1, t.tick + 1 + k as i32)
                    .map(|rec| KnownTick::from_rec(&rec));
            }
            assemble(
                x,
                |j| &feats[i.saturating_sub(j.min(self.cfg.hist_keep.max(1) - 1))],
                &t.rays[1],
                &t.rays[0],
                &inflight[..n_if],
                lag,
                &known,
            );
            label = self.label(r);
            label.mask_known(n);
        } else {
            let g = &self.clips[r.game as usize];
            let feats = &self.cf[r.game as usize];
            let t = &g.ticks[i];
            for (k, f) in inflight.iter_mut().enumerate().take(n_if) {
                let tag = t.tick + 1 + k as i32;
                let rec = g.sent_at(i, tag).unwrap_or_default();
                inflight_features(&rec, f);
            }
            assemble(
                x,
                |j| &feats[i.saturating_sub(j.min(self.cfg.hist_keep.max(1) - 1))],
                &t.rays[1],
                &t.rays[0],
                &inflight[..n_if],
                lag,
                &known,
            );
            label = self.label(r);
        }
        label
    }

    /// The label alone (not masked by known ticks).
    pub fn label(&self, r: SampleRef) -> Label {
        let i = r.idx as usize;
        let mut label = Label::default();
        if r.src == 0 {
            let g = &self.arena[r.game as usize];
            let base = f64::from(g.ticks[i].frames[1].angle);
            let mut prev = g.ticks[i].applied[1];
            for k in 0..HORIZON {
                let Some(next) = g.ticks.get(i + 1 + k) else { break };
                let cur = next.applied[1];
                let f = &next.frames[1];
                // A frozen or dead opponent's inputs do nothing: no label.
                if f.alive && f.freeze_left == 0 {
                    label_tick(&mut label, k, &prev, &cur, base);
                    if self.cfg.swing_label {
                        // The step `T + k` swung when the frame it led to shows the weapon used one tick ago.
                        label.press &= !(1 << k);
                        // (not in the first 30 ticks: a tee that never swung has `attack_tick` 0, so the first tick of a game reads as a use)
                        label.press |= u8::from(f.attack_age == 1 && g.tick0 + (i + 1 + k) as i32 >= 30) << k;
                    }
                }
                prev = cur;
            }
        } else if r.src == 2 {
            return self.human_label(&self.humans[r.game as usize], i);
        } else {
            let g = &self.clips[r.game as usize];
            let l = labels_at(g, i, HORIZON);
            label.v_dir = l.v_obs;
            label.v_hook = l.v_obs;
            label.v_aim = l.v_obs;
            label.v_press = l.v_press;
            label.hook = l.hook;
            label.press = l.press;
            for k in 0..HORIZON {
                label.dir[k] = (i32::from(l.dir[k]) + 1).clamp(0, 2) as u8;
                label.aim_delta[k] = l.aim_delta[k];
            }
        }
        label
    }

    /// The label of the human-game frame `i`: window tick `k` is the real input applied in the step into tick `T + 1 + k` (`prev` the one before it),
    /// valid where the opponent is alive and free in the frame that ends the step and both inputs are known; the fire label is a real **swing** (the
    /// weapon went off: the frames show its `attack_tick`), as for clips and the `swing_label` arena.
    fn human_label(&self, g: &HumanGame, i: usize) -> Label {
        let mut label = Label::default();
        let t0 = g.ticks[i].tick;
        let base = f64::from(g.ticks[i].frames[1].angle);
        for k in 0..HORIZON {
            // the frame that ends the step into T + 1 + k: the first one at or after it
            let j = i + (k + 2) / 2;
            if j >= g.ticks.len() || !g.consecutive(i, 0, j - i) {
                break;
            }
            let f = &g.ticks[j].frames[1];
            if !f.alive || f.freeze_left > 0 || !g.ticks[j].duel {
                break;
            }
            let (Some(prev), Some(cur)) = (g.input_at(1, t0 + k as i32), g.input_at(1, t0 + 1 + k as i32)) else {
                continue;
            };
            label_tick(&mut label, k, &prev, &cur, base);
            // a swing in the step T + k -> T + k + 1 leaves `attack_tick == T + k` in the frames that follow
            let swung = (i + 1..=j).any(|m| {
                let f = &g.ticks[m].frames[1];
                f.attack_age < 120 && g.ticks[m].tick - i32::from(f.attack_age) == t0 + k as i32
            });
            label.press &= !(1 << k);
            label.press |= u8::from(swung) << k;
        }
        label
    }

    /// The opponent's frame at the sample tick, and (arena and human games) the input it applied just before it.
    pub fn snapshot_of(&self, r: SampleRef) -> (Option<InputRec>, TeeFrame) {
        if r.src == 0 {
            let t = &self.arena[r.game as usize].ticks[r.idx as usize];
            (Some(t.applied[1]), t.frames[1])
        } else if r.src == 2 {
            let g = &self.humans[r.game as usize];
            let t = &g.ticks[r.idx as usize];
            (g.input_at(1, t.tick), t.frames[1])
        } else {
            (None, self.clips[r.game as usize].ticks[r.idx as usize].frames[1])
        }
    }

    /// Mean of every frame feature over the sample frames of one source (a diagnostic: compare the arena's and the clips').
    pub fn mean_features(&self, src: u8) -> Vec<f64> {
        let mut sum = vec![0.0f64; FD];
        let mut n = 0u64;
        let mut add = |f: &[f32; FD]| {
            n += 1;
            for (s, x) in sum.iter_mut().zip(f) {
                *s += f64::from(*x);
            }
        };
        if src == 0 {
            for (g, f) in self.arena.iter().zip(&self.af) {
                for (t, v) in g.ticks.iter().zip(f) {
                    if t.frames[1].alive {
                        add(v);
                    }
                }
            }
        } else {
            for (g, f) in self.clips.iter().zip(&self.cf) {
                for (t, v) in g.ticks.iter().zip(f) {
                    if t.duel {
                        add(v);
                    }
                }
            }
        }
        sum.iter().map(|s| s / n.max(1) as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::data::TickRec;
    use super::*;
    use crate::clipdata::ClipTick;
    use crate::frame::N_RAYS;

    fn tee(dir: i8, angle: f32) -> TeeFrame {
        TeeFrame {
            alive: true,
            direction: dir,
            angle,
            ..TeeFrame::default()
        }
    }

    fn arena_game(n: usize, lag: u8) -> GameRec {
        let ticks = (0..n)
            .map(|t| TickRec {
                frames: [tee(0, 0.0), tee(((t / 3) % 3) as i8 - 1, 0.0)],
                applied: [
                    InputRec {
                        direction: 1,
                        ..InputRec::default()
                    },
                    InputRec {
                        direction: ((t / 3) % 3) as i8 - 1,
                        fire: if t == 6 { 1 } else { 0 },
                        ..InputRec::default()
                    },
                ],
                rays: [[1.0; N_RAYS]; 2],
            })
            .collect();
        GameRec {
            arena: "t".into(),
            seed: 1,
            lag: [lag, 0],
            swap: false,
            decide_every: 2,
            tick0: 0,
            ticks,
        }
    }

    fn clip_game(n: usize) -> ClipGame {
        let ticks = (0..n)
            .map(|j| ClipTick {
                tick: 100 + 2 * j as i32,
                frames: [tee(0, 0.0), tee(1, 0.5)],
                rays: [[1.0; N_RAYS]; 2],
                sent: [Some(InputRec::default()), Some(InputRec::default())],
                opp_attack_tick: 0,
                opp_weapon: 0,
                duel: true,
            })
            .collect();
        ClipGame {
            source: "t".into(),
            session: 0,
            ticks,
        }
    }

    #[test]
    fn samples_cover_arena_decision_ticks_and_clip_duel_frames() {
        let c = Corpus::new(vec![arena_game(20, 2)], vec![clip_game(6)], CorpusCfg::default());
        let s = c.samples();
        let arena: Vec<u32> = s.iter().filter(|r| r.src == 0).map(|r| r.idx).collect();
        assert_eq!(
            arena,
            vec![0, 2, 4, 6, 8, 10, 12, 14, 16],
            "even ticks with room for the window"
        );
        let clips: Vec<u32> = s.iter().filter(|r| r.src == 1).map(|r| r.idx).collect();
        // The last frame has no future, and the window-2 inputs of frame i are tagged in frame i + 1's list: the first five frames.
        assert_eq!(clips, vec![0, 1, 2, 3, 4]);
        // A lag above the model's, or no lag, is skipped.
        let c = Corpus::new(vec![arena_game(20, 5), arena_game(20, 0)], vec![], CorpusCfg::default());
        assert!(c.samples().is_empty());
    }

    #[test]
    fn arena_labels_are_the_applied_inputs_and_known_ticks_are_masked() {
        let cfg = CorpusCfg {
            known_p: [0.0, 0.0, 1.0, 0.0, 0.0],
            ..CorpusCfg::default()
        };
        let c = Corpus::new(vec![arena_game(20, 2)], vec![], cfg);
        let r = SampleRef {
            src: 0,
            game: 0,
            idx: 4,
        };
        let mut x = Box::new([0.0f32; INPUT_DIM]);
        let l = c.make(r, 7, &mut x);
        // Two ticks known: only k = 2, 3 are learned.
        assert_eq!(l.v_dir, 0b1100);
        let raw = c.label(r);
        assert_eq!(raw.v_dir, 0b1111);
        assert_eq!(raw.dir[0], 1, "applied[1] at tick 5: (5/3)%3 - 1 = 0 -> class 1");
        // The known block holds the real inputs of ticks 0 and 1 and nothing for 2, 3.
        let k0 = INPUT_DIM - HORIZON * super::super::feature::KNOWN_DIM;
        assert_eq!(x[k0], 1.0);
        assert_eq!(x[k0 + super::super::feature::KNOWN_DIM], 1.0);
        assert_eq!(x[k0 + 2 * super::super::feature::KNOWN_DIM], 0.0);
        // A swing at tick 6: the press label of window tick 1 of the sample at tick 4... applied index 6 = step 5 -> k = 1 of T = 4 is tick 6.
        let r = SampleRef {
            src: 0,
            game: 0,
            idx: 4,
        };
        assert_eq!(c.label(r).press, 0b0010);
    }

    #[test]
    fn a_swing_label_counts_weapon_uses_not_presses_during_the_reload() {
        let mut g = arena_game(20, 2);
        g.tick0 = 100;
        // Presses (fire 0 -> 1) at ticks[6] and ticks[10]; only the first one swung: the frame of ticks[6] shows the weapon used one tick ago, the one of ticks[10] does not.
        for t in 0..20 {
            g.ticks[t].applied[1].fire = if t == 6 || t == 10 { 1 } else { 0 };
            g.ticks[t].frames[1].attack_age = if t == 6 { 1 } else { 50 };
        }
        let by_press = Corpus::new(vec![g.clone()], vec![], CorpusCfg::default());
        let by_swing = Corpus::new(
            vec![g],
            vec![],
            CorpusCfg {
                swing_label: true,
                ..CorpusCfg::default()
            },
        );
        // The sample at index 4 reads ticks[5..=8] as window ticks 0..3: the swing is window tick 1.
        let r = SampleRef {
            src: 0,
            game: 0,
            idx: 4,
        };
        assert_eq!(by_press.label(r).press, 0b0010);
        assert_eq!(by_swing.label(r).press, 0b0010);
        // The sample at index 8 reads ticks[9..=12]: the press of ticks[10] (window tick 1) is not a swing.
        let r = SampleRef {
            src: 0,
            game: 0,
            idx: 8,
        };
        assert_eq!(by_press.label(r).press, 0b0010);
        assert_eq!(by_swing.label(r).press, 0);
    }

    #[test]
    fn the_known_count_is_deterministic_and_follows_the_probabilities() {
        let cfg = CorpusCfg {
            known_p: [0.5, 0.25, 0.25, 0.0, 0.0],
            ..CorpusCfg::default()
        };
        let c = Corpus::new(vec![arena_game(400, 2)], vec![], cfg);
        let mut hist = [0usize; 5];
        for idx in 0..200u32 {
            let r = SampleRef { src: 0, game: 0, idx };
            let n = c.known_n(r, 3);
            assert_eq!(n, c.known_n(r, 3));
            hist[n] += 1;
        }
        assert!(hist[0] > 70 && hist[0] < 130, "{hist:?}");
        assert!(hist[1] > 25 && hist[2] > 25 && hist[3] == 0, "{hist:?}");
        assert_ne!(
            (0..200u32)
                .map(|i| c.known_n(
                    SampleRef {
                        src: 0,
                        game: 0,
                        idx: i
                    },
                    3
                ))
                .collect::<Vec<_>>(),
            (0..200u32)
                .map(|i| c.known_n(
                    SampleRef {
                        src: 0,
                        game: 0,
                        idx: i
                    },
                    4
                ))
                .collect::<Vec<_>>(),
            "another epoch draws again"
        );
    }

    #[test]
    fn clip_labels_mask_what_a_snapshot_does_not_show() {
        let mut g = clip_game(4);
        g.ticks[1].frames[1].direction = -1;
        g.ticks[1].opp_attack_tick = 101;
        g.ticks[2].opp_attack_tick = 101;
        g.ticks[3].opp_attack_tick = 101;
        let c = Corpus::new(vec![], vec![g], CorpusCfg::default());
        let r = SampleRef {
            src: 1,
            game: 0,
            idx: 0,
        };
        let l = c.label(r);
        assert_eq!(l.v_dir & 1, 0, "k = 0 is not shown");
        assert_eq!(l.v_dir & 2, 2, "k = 1 is");
        assert_eq!(l.dir[1], 0, "direction -1 -> class 0");
        assert_eq!(l.v_jump, 0, "the jump level is never shown");
        assert_eq!(l.press, 0b0010, "the swing at tick 101 is window tick 1");
        let mut x = Box::new([0.0f32; INPUT_DIM]);
        let made = c.make(r, 0, &mut x);
        assert_eq!(made, l, "a clip sample has no known ticks");
        let lag_slot = INPUT_DIM - HORIZON * super::super::feature::KNOWN_DIM - super::super::feature::LAG_SLOTS;
        assert_eq!(x[lag_slot + 2], 1.0, "the live window");
    }

    /// Frames every 2 ticks from tick 100; the opponent (slot 1) walks right (+1) from tick 100 on, jumps at 105, hooks at 106-107 and swings its
    /// weapon at tick 106 (the frames from 108 on show `attack_tick` 106); we walk left throughout.
    fn human_game() -> HumanGame {
        let n = 6usize;
        let ticks = (0..n)
            .map(|j| {
                let tick = 100 + 2 * j as i32;
                let mut opp = tee(1, 0.5);
                opp.attack_age = if tick >= 108 { (tick - 106) as i16 } else { 120 };
                ClipTick {
                    tick,
                    frames: [tee(-1, 0.0), opp],
                    rays: [[1.0; N_RAYS]; 2],
                    sent: [None, None],
                    opp_attack_tick: 0,
                    opp_weapon: 0,
                    duel: true,
                }
            })
            .collect();
        let inputs = (99..99 + 14)
            .map(|t| {
                let me = InputRec {
                    direction: -1,
                    ..InputRec::default()
                };
                let opp = InputRec {
                    direction: if t >= 100 { 1 } else { 0 },
                    jump: t == 105,
                    hook: t == 106 || t == 107,
                    fire: if t == 106 {
                        1
                    } else if t >= 107 {
                        2
                    } else {
                        0
                    },
                    ..InputRec::default()
                };
                [Some(me), Some(opp)]
            })
            .collect();
        HumanGame {
            source: "h0-1-2p1".into(),
            session: 0,
            ticks,
            first: 99,
            inputs,
        }
    }

    #[test]
    fn human_labels_are_the_real_inputs_with_a_swing_from_the_frames() {
        let c = Corpus::new(vec![], vec![], CorpusCfg::default()).with_humans(vec![human_game()]);
        let s = c.samples();
        let idx: Vec<u32> = s.iter().map(|r| r.idx).collect();
        assert_eq!(
            idx,
            vec![0, 1, 2, 3, 4],
            "every frame with a successor and the inputs it needs"
        );
        // sample 3 is the frame at tick 106: window ticks 0..3 are the inputs applied in the steps into 107..110
        let r = SampleRef {
            src: 2,
            game: 0,
            idx: 3,
        };
        assert_eq!(
            c.snapshot_of(r).0.map(|i| i.hook),
            Some(true),
            "the true previous input is the one of tick 106"
        );
        // sample 2 is the frame at tick 104: k = 0 is the step into 105 (the jump), k = 1 the step into 106 (hook on, press), k = 2 into 107
        let r = SampleRef {
            src: 2,
            game: 0,
            idx: 2,
        };
        let l = c.label(r);
        assert_eq!(l.v_dir & 0b0011, 0b0011);
        assert_eq!(l.jump & 0b0001, 0b0001, "jump key down at tick 105");
        assert_eq!(l.hook & 0b0011, 0b0010, "hook at 106, not at 105");
        assert_eq!(l.dir[0], 2, "direction +1 -> class 2");
        // the weapon went off in the step 106 -> 107? No: a press at world tick 106 is applied in the step into tick 106, i.e. `attack_tick`
        // 105... the frames say 106, so the swing is the step T + k -> T + k + 1 = 106 -> 107 for T = 104: k = 2
        assert_eq!(l.press, 0b0100, "swing in window tick 2 (attack tick 106 = T + 2)");
        // our in-flight inputs are the real ones, and known ticks come from the opponent's real inputs
        let mut x = Box::new([0.0f32; INPUT_DIM]);
        let l2 = c.make(r, 0, &mut x);
        assert_eq!(l2, l, "no known ticks by default");
        let cfg = CorpusCfg {
            known_p: [0.0, 0.0, 1.0, 0.0, 0.0],
            ..CorpusCfg::default()
        };
        let c = Corpus::new(vec![], vec![], cfg).with_humans(vec![human_game()]);
        let l3 = c.make(r, 0, &mut x);
        assert_eq!(l3.v_dir, l.v_dir & 0b1100, "two known ticks are not learned");
        let k0 = INPUT_DIM - HORIZON * super::super::feature::KNOWN_DIM;
        assert_eq!(x[k0], 1.0);
        assert_eq!(x[k0 + 2 * super::super::feature::KNOWN_DIM], 0.0);
    }

    #[test]
    fn a_human_sample_needs_the_inputs_and_a_free_alive_pair() {
        let mut g = human_game();
        // our own inputs are not in the demo (the recording player's): the samples stay and read them off the frames
        for i in &mut g.inputs {
            i[0] = None;
        }
        g.ticks[1].frames[0].direction = 1;
        g.ticks[1].frames[0].jumped = 1;
        let c = Corpus::new(vec![], vec![], CorpusCfg::default()).with_humans(vec![g.clone()]);
        let idx: Vec<u32> = c.samples().iter().map(|r| r.idx).collect();
        assert_eq!(idx, vec![0, 1, 2, 3, 4]);
        let rec = human_us_input(&g, 0, 101);
        assert_eq!(
            (rec.direction, rec.jump),
            (1, true),
            "the frame that ends the step into tick 101 is the one at tick 102"
        );
        let mut x = Box::new([0.0f32; INPUT_DIM]);
        let r = SampleRef {
            src: 2,
            game: 0,
            idx: 0,
        };
        c.make(r, 0, &mut x);
        let if0 = INPUT_DIM
            - HORIZON * super::super::feature::KNOWN_DIM
            - super::super::feature::LAG_SLOTS
            - IF_SLOTS * IF_DIM;
        assert_eq!(x[if0], 1.0, "the in-flight direction comes from the frame");
        // the opponent's previous input unknown: no sample
        let mut g = human_game();
        g.inputs[1][1] = None; // tick 100
        let c = Corpus::new(vec![], vec![], CorpusCfg::default()).with_humans(vec![g]);
        let idx: Vec<u32> = c.samples().iter().map(|r| r.idx).collect();
        assert!(!idx.contains(&0), "{idx:?}");
        let mut g = human_game();
        g.ticks[2].frames[1].freeze_left = 30;
        let c = Corpus::new(vec![], vec![], CorpusCfg::default()).with_humans(vec![g]);
        let idx: Vec<u32> = c.samples().iter().map(|r| r.idx).collect();
        assert!(!idx.contains(&2), "a frozen opponent is not a sample");
        // and the label stops at the frame where it froze
        let l = c.label(SampleRef {
            src: 2,
            game: 0,
            idx: 0,
        });
        assert_eq!(l.v_dir, 0b0011, "frame 2 is frozen: window ticks 2, 3 end in it");
    }
}
