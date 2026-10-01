//! Batching and data-parallel training (acceptance criterion 4): [`train_step`] runs
//! forward-with-recording + [`crate::backward::backward`] for every [`Sequence`] of a batch on a
//! separate `rayon` worker, then reduces the per-sequence gradients into one [`crate::optim::
//! ParamGradients`] — deliberately *stronger* than the acceptance criterion's "deterministic
//! reduction order so results are reproducible given the same thread count": `rayon`'s
//! `par_iter().map(..).collect::<Vec<_>>()` preserves input order regardless of which worker
//! processed which sequence or in what order they finished, and the reduction over that ordered
//! `Vec` happens single-threaded afterwards — so the summed gradient's floating-point rounding is
//! identical **no matter how many threads ran the parallel part**, not merely across repeats with
//! the same thread count.
//!
//! Nothing here computes a loss: a [`Sequence`] carries `grad_dn` (`dL/d(dn_rates)` per decision)
//! already computed by the caller (a loss lives outside `ddai-fly` — see `docs/formats.md`/this
//! crate's README for why: `backward` is loss-agnostic, matching how `crate::optim` is
//! optimiser-agnostic about *why* a gradient has the value it does). A caller wanting the actual
//! loss value typically runs one plain (non-recording) `step_decision` pass first to get
//! `dn_rates`, computes `loss`/`grad_dn` from that against its targets, *then* builds the
//! `Sequence` `train_step` will forward-with-recording all over again — a second forward pass per
//! sequence per step, but forward is the cheap half of this (backward costs 2-5x forward per the
//! phase-0 benchmark, `docs/research/rust-stack.md` §4 B.1's bwd-vs-fwd columns), so this doesn't
//! change the batch's throughput order of magnitude while keeping `ddai-fly` itself loss-free.

use rayon::prelude::*;

use crate::backward::{BackwardIndex, ExtraRateGrad, backward};
use crate::model::FlyModel;
use crate::optim::ParamGradients;
use crate::recorder::TrajectoryRecorder;
use crate::state::FlyState;

/// One training sequence: `t_decisions = inputs.len() == grad_dn.len()` decisions of
/// `model.config().substeps_per_decision` substeps each, starting from `v_init`.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub v_init: Vec<f32>,
    /// `t_decisions` entries, each `model.num_inputs()` long.
    pub inputs: Vec<Vec<f32>>,
    /// `t_decisions` entries, each `model.num_outputs()` long — `dL/d(dn_rates)` for that
    /// decision (all-zero for a decision with no output-side loss term).
    pub grad_dn: Vec<Vec<f32>>,
    /// Optional auxiliary rate-gradient taps (acceptance criterion 1's "any neuron's rate at any
    /// substep") — e.g. an activity regulariser's contribution
    /// ([`crate::optim::activity_regularizer_rate_grad`]).
    pub extra_taps: Vec<(usize, usize, Vec<f32>)>,
}

impl Sequence {
    pub(crate) fn validate(&self, model: &FlyModel) {
        let t = self.inputs.len();
        assert_eq!(self.grad_dn.len(), t, "Sequence: grad_dn.len() must equal inputs.len()");
        assert_eq!(
            self.v_init.len(),
            model.num_neurons(),
            "Sequence: v_init.len() must equal num_neurons"
        );
        for input_t in &self.inputs {
            assert_eq!(
                input_t.len(),
                model.num_inputs(),
                "Sequence: each inputs[t].len() must equal num_inputs"
            );
        }
        for g in &self.grad_dn {
            assert_eq!(
                g.len(),
                model.num_outputs(),
                "Sequence: each grad_dn[t].len() must equal num_outputs"
            );
        }
    }
}

/// `T * substeps_per_decision * num_neurons * STATE_WORDS * 4` bytes: the memory one
/// [`Sequence`]'s [`crate::recorder::TrajectoryRecorder`] holds (acceptance criterion 4's
/// documented memory estimate). `STATE_WORDS = 3` — `V`, `r`, `V_∞`, [`crate::recorder::
/// TrajectoryRecorder`]'s three per-substep buffers (see that module: `r` could in principle be
/// recomputed from `V` instead of stored, but 7.1 already committed to storing it, and re-deriving
/// that trade-off here would just make this estimate and the recorder's actual allocation diverge).
pub const STATE_WORDS_PER_SUBSTEP: usize = 3;

pub fn sequence_memory_bytes(model: &FlyModel, t_decisions: usize) -> usize {
    let substeps = model.config().substeps_per_decision as usize;
    t_decisions * substeps * model.num_neurons() * STATE_WORDS_PER_SUBSTEP * std::mem::size_of::<f32>()
}

/// Sum of [`sequence_memory_bytes`] over every sequence in the batch — a **conservative upper
/// bound**, not a peak-RSS estimate (review round 1, F7c): `train_step` never actually holds all
/// of these recorders in memory at once. `rayon`'s work-stealing keeps at most one `Sequence`'s
/// `TrajectoryRecorder` alive per active worker at any instant (each worker drops its current
/// one before picking up the next), so real peak trajectory memory is closer to
/// `min(num_threads, sequences.len())` sequences' worth, not `sequences.len()`'s — e.g. the
/// crate README's throughput table measured ~65 MiB peak RSS on the real S graph at `B=64`, 8
/// threads, where this function's estimate is ~4x that. Use [`concurrent_batch_memory_bytes`] for
/// a tighter, concurrency-aware number; this one stays a safe (if pessimistic) bound for a caller
/// that doesn't know or want to think about how many threads will actually run the batch.
pub fn batch_memory_bytes(model: &FlyModel, sequences: &[Sequence]) -> usize {
    sequences
        .iter()
        .map(|s| sequence_memory_bytes(model, s.inputs.len()))
        .sum()
}

/// [`batch_memory_bytes`], but scaled down to reflect actual concurrency (review round 1, F7c):
/// sums [`sequence_memory_bytes`] over only the `num_threads` *largest* sequences in the batch
/// (largest-first, so a batch of unevenly-sized sequences is still bounded correctly — the
/// pessimistic case rayon's work-stealing could actually schedule is every thread happening to
/// pick up one of the biggest sequences at once), which is what's actually resident in memory at
/// any instant during [`train_step`] on a pool of that size. Pass
/// `rayon::current_num_threads()` for "however many threads are in the pool that will actually
/// run this" if calling from inside (or about to install) that pool.
pub fn concurrent_batch_memory_bytes(model: &FlyModel, sequences: &[Sequence], num_threads: usize) -> usize {
    let mut per_sequence: Vec<usize> = sequences
        .iter()
        .map(|s| sequence_memory_bytes(model, s.inputs.len()))
        .collect();
    per_sequence.sort_unstable_by(|a, b| b.cmp(a));
    per_sequence.iter().take(num_threads.max(1)).sum()
}

/// [`train_step`]'s only failure mode: the batch's estimated trajectory memory
/// ([`batch_memory_bytes`]) exceeds the caller's configured cap. `train_step` checks this
/// *before* allocating anything for the batch (so a too-large batch never partially allocates),
/// per acceptance criterion 4's "enforced by a configurable cap".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryCapExceeded {
    pub estimated_bytes: usize,
    pub cap_bytes: usize,
}

impl std::fmt::Display for MemoryCapExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "batch trajectory memory estimate ({} bytes) exceeds the configured cap ({} bytes)",
            self.estimated_bytes, self.cap_bytes
        )
    }
}

impl std::error::Error for MemoryCapExceeded {}

/// The result of one [`train_step`] call: the batch-summed parameter gradient (raw — **not**
/// averaged over the batch, clipped, or regularised; the caller composes those with
/// `crate::optim` afterwards, same "mechanism here, policy at the call site" split as everywhere
/// else in this module) plus, per sequence, `dL/dinput[t][k]` (a hook for the future input
/// encoder, task 7.3, which needs exactly this to train through its own parameters).
#[derive(Debug, Clone)]
pub struct BatchGradients {
    pub grad: ParamGradients,
    /// `sequences.len()` entries; entry `i` is sequence `i`'s `grad_inputs` from
    /// [`crate::backward::BpttGradients`] (`t_decisions` vectors, each `num_inputs` long).
    pub grad_inputs: Vec<Vec<Vec<f32>>>,
}

/// Runs forward (recording) + backward for every sequence in `sequences`, in parallel across
/// `rayon`'s current thread pool (a caller wanting a specific thread count installs its own
/// `rayon::ThreadPool` around this call — `train_step` itself never spawns or sizes a pool, so it
/// composes with whatever pool the caller (a benchmark comparing 1 vs. 8 threads; a live training
/// loop sharing the process-wide default pool) already has), then reduces into one
/// [`BatchGradients`] (see the module doc comment for why this reduction is fully deterministic,
/// not merely "with the same thread count").
///
/// Panics if any `Sequence`'s shapes don't match `model` (a caller-assembled-batch mismatch, not a
/// data error — same convention as `crate::backward::backward`). Returns
/// [`MemoryCapExceeded`] instead of running if `memory_cap_bytes` is `Some` and the batch's
/// estimated trajectory memory ([`batch_memory_bytes`] — a conservative, concurrency-agnostic
/// upper bound; see that function's doc comment, and [`concurrent_batch_memory_bytes`] for a
/// tighter one, if a caller wants to check against that instead before calling this) exceeds it.
pub fn train_step(
    model: &FlyModel,
    index: &BackwardIndex,
    sequences: &[Sequence],
    memory_cap_bytes: Option<usize>,
) -> Result<BatchGradients, MemoryCapExceeded> {
    for seq in sequences {
        seq.validate(model);
    }
    if let Some(cap) = memory_cap_bytes {
        let estimated = batch_memory_bytes(model, sequences);
        if estimated > cap {
            return Err(MemoryCapExceeded {
                estimated_bytes: estimated,
                cap_bytes: cap,
            });
        }
    }

    let substeps = model.config().substeps_per_decision as usize;
    let per_sequence: Vec<(ParamGradients, Vec<Vec<f32>>)> = sequences
        .par_iter()
        .map(|seq| {
            let t = seq.inputs.len();
            let mut state = FlyState::new(model);
            state.set_v(model, &seq.v_init);
            let mut recorder = TrajectoryRecorder::new(model.num_neurons(), t * substeps);
            for input_t in &seq.inputs {
                state.step_decision_recording(model, input_t, &mut recorder);
            }

            let mut scratch = crate::backward::BpttScratch::new(model);
            let grad_dn_refs: Vec<&[f32]> = seq.grad_dn.iter().map(Vec::as_slice).collect();
            let extra_refs: Vec<ExtraRateGrad<'_>> = seq
                .extra_taps
                .iter()
                .map(|(decision, local_substep, grad)| ExtraRateGrad {
                    decision: *decision,
                    local_substep: *local_substep,
                    grad,
                })
                .collect();
            let g = backward(
                model,
                index,
                &recorder,
                &seq.v_init,
                t,
                &grad_dn_refs,
                &extra_refs,
                false,
                &mut scratch,
            );
            (
                ParamGradients {
                    a: g.grad_a,
                    b: g.grad_b,
                    theta: g.grad_theta,
                },
                g.grad_inputs,
            )
        })
        .collect();

    let mut total = ParamGradients::zeros_like(model.params());
    let mut grad_inputs = Vec::with_capacity(sequences.len());
    for (g, gi) in per_sequence {
        total.add_assign(&g);
        grad_inputs.push(gi);
    }

    Ok(BatchGradients {
        grad: total,
        grad_inputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FlyConfig;
    use crate::params::FlyParams;
    use crate::test_fixtures::tiny_chain_flyg;

    fn tiny_model() -> FlyModel {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig {
            substeps_per_decision: 2,
            ..FlyConfig::default()
        };
        let params = FlyParams::init_default(&flyg, &config, 1);
        FlyModel::new(flyg, config, params).unwrap()
    }

    fn seq(_model: &FlyModel, t: usize, seed: f32) -> Sequence {
        Sequence {
            v_init: vec![0.1 * seed, -0.2 * seed, 0.3 * seed],
            inputs: (0..t).map(|i| vec![0.5 * seed + i as f32 * 0.01]).collect(),
            grad_dn: (0..t).map(|i| vec![0.2 * seed - i as f32 * 0.01]).collect(),
            extra_taps: vec![],
        }
    }

    #[test]
    fn sequence_memory_bytes_matches_the_documented_formula() {
        let model = tiny_model();
        let bytes = sequence_memory_bytes(&model, 5);
        let want = 5 * 2 * model.num_neurons() * 3 * 4;
        assert_eq!(bytes, want);
    }

    /// Review round 1, F7c: `concurrent_batch_memory_bytes` must be `<=` the conservative
    /// `batch_memory_bytes` sum whenever there are more sequences than threads (the whole point),
    /// must equal the sum when there are *fewer* sequences than threads (every one of them can be
    /// concurrent, so the "conservative" and "concurrency-aware" estimates coincide), and must
    /// correctly pick the *largest* sequences first when sizes differ.
    #[test]
    fn concurrent_batch_memory_bytes_reflects_thread_count_and_picks_the_largest_sequences() {
        let model = tiny_model();
        let sequences = vec![seq(&model, 2, 1.0), seq(&model, 8, 1.0), seq(&model, 4, 1.0)];
        let total = batch_memory_bytes(&model, &sequences);

        // Fewer threads than sequences: only the two largest (T=8, T=4) should count.
        let with_2_threads = concurrent_batch_memory_bytes(&model, &sequences, 2);
        let want_2 = sequence_memory_bytes(&model, 8) + sequence_memory_bytes(&model, 4);
        assert_eq!(with_2_threads, want_2);
        assert!(with_2_threads < total);

        // At least as many threads as sequences: coincides with the conservative sum.
        let with_10_threads = concurrent_batch_memory_bytes(&model, &sequences, 10);
        assert_eq!(with_10_threads, total);

        // `num_threads == 0` must not divide by zero or panic; treated as "at least 1".
        let with_0_threads = concurrent_batch_memory_bytes(&model, &sequences, 0);
        assert_eq!(with_0_threads, sequence_memory_bytes(&model, 8));
    }

    #[test]
    fn train_step_respects_a_tight_memory_cap() {
        let model = tiny_model();
        let index = BackwardIndex::build(&model);
        let sequences = vec![seq(&model, 4, 1.0)];
        let too_small = batch_memory_bytes(&model, &sequences) - 1;
        let err = train_step(&model, &index, &sequences, Some(too_small)).unwrap_err();
        assert_eq!(err.cap_bytes, too_small);
    }

    #[test]
    fn train_step_succeeds_within_a_sufficient_memory_cap() {
        let model = tiny_model();
        let index = BackwardIndex::build(&model);
        let sequences = vec![seq(&model, 4, 1.0)];
        let enough = batch_memory_bytes(&model, &sequences);
        assert!(train_step(&model, &index, &sequences, Some(enough)).is_ok());
    }

    #[test]
    fn train_step_with_no_cap_always_runs() {
        let model = tiny_model();
        let index = BackwardIndex::build(&model);
        let sequences = vec![seq(&model, 4, 1.0)];
        assert!(train_step(&model, &index, &sequences, None).is_ok());
    }

    #[test]
    fn batch_gradient_equals_the_sum_of_per_sequence_gradients() {
        let model = tiny_model();
        let index = BackwardIndex::build(&model);
        let s1 = seq(&model, 3, 1.0);
        let s2 = seq(&model, 3, -0.7);

        let batch = train_step(&model, &index, &[s1.clone(), s2.clone()], None).unwrap();
        let r1 = train_step(&model, &index, &[s1], None).unwrap();
        let r2 = train_step(&model, &index, &[s2], None).unwrap();

        for i in 0..batch.grad.a.len() {
            let want = r1.grad.a[i] + r2.grad.a[i];
            assert!(
                (batch.grad.a[i] - want).abs() < 1e-9,
                "grad_a[{i}]: batch={} want(sum)={want}",
                batch.grad.a[i]
            );
        }
        for i in 0..batch.grad.b.len() {
            let want = r1.grad.b[i] + r2.grad.b[i];
            assert!((batch.grad.b[i] - want).abs() < 1e-9);
        }
        assert_eq!(batch.grad_inputs.len(), 2);
    }

    /// The exact same batch, run through thread pools of different sizes, must give a
    /// **bit-identical** summed gradient (the module doc comment's determinism claim) — not just
    /// "close within tolerance".
    #[test]
    fn batch_gradient_is_bit_identical_across_thread_counts() {
        let model = tiny_model();
        let index = BackwardIndex::build(&model);
        let sequences: Vec<Sequence> = (0..9).map(|i| seq(&model, 3, 0.1 * i as f32 - 0.4)).collect();

        let run_with = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| train_step(&model, &index, &sequences, None).unwrap())
        };

        let one = run_with(1);
        let four = run_with(4);
        assert_eq!(
            one.grad.a, four.grad.a,
            "grad_a must be bit-identical across thread counts"
        );
        assert_eq!(one.grad.b, four.grad.b);
        assert_eq!(one.grad.theta, four.grad.theta);
    }

    #[test]
    #[should_panic]
    fn train_step_panics_on_a_sequence_shape_mismatch() {
        let model = tiny_model();
        let index = BackwardIndex::build(&model);
        let mut bad = seq(&model, 3, 1.0);
        bad.v_init.push(0.0);
        let _ = train_step(&model, &index, &[bad], None);
    }
}
