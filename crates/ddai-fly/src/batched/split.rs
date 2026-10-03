//! `K` independent sub-batch engines (task 7.2c): the opt-in answer to a loaded host.
//!
//! One engine splits every substep over all its threads and ends the substep with a rendezvous
//! (256 regions per training step on M); when the host does not give all those threads a core at
//! the same moment, a single descheduled worker stalls the rest (8 threads were slower than 3 on
//! the loaded machine of task 7.2b). Here the batch is cut into `K` groups of whole 8-lane
//! cells, each group runs on **its own engine and its own thread pool** of `threads / K`
//! threads, concurrently and without any barrier between the groups: a stalled pool delays only
//! its own group, and the rendezvous inside a pool involves fewer threads. There are never more
//! pools (or threads) than the calling pool has threads: with `K` above that, the groups are
//! dealt round-robin onto `threads` one-thread pools, several groups sharing one (they then run
//! one after the other there). The numerics only know `K`, never the pools. The price is that
//! every edge is read once per group instead of once per batch (the per-edge work is shared by
//! `8 * NB / K` lanes instead of `8 * NB`), about 1.3-2x the CPU time per step at `K = 4..8`
//! (README, "Задача 7.2c").
//!
//! What stays identical to one engine: every per-lane result (`dn_rates`, type means, the input
//! and initial-state gradients) is bitwise the same, since a lane's arithmetic does not depend on
//! the other lanes and the groups are whole cells. The parameter gradients are sums over the
//! batch, now added group by group (each group's sum first, in group order): the same value to
//! f32 summation order, bitwise **independent of the thread count** and of the pools' sizes, but
//! different for a different `K`.

use super::{BatchedEngine, BatchedForwardOptions, BatchedGradients, BatchedSeqGrad, BatchedSeqInput, LANES};
use crate::model::FlyModel;
use crate::train::MemoryCapExceeded;

/// The sub-engines of a split batch and their pools.
pub(super) struct Parts {
    engines: Vec<BatchedEngine>,
    /// `min(K, threads)` pools; group `p` runs on `pools[p % pools.len()]`.
    pools: Vec<rayon::ThreadPool>,
    /// First sequence of every group, and the batch size at the end (`engines.len() + 1` long).
    starts: Vec<usize>,
}

impl Parts {
    pub(super) fn bytes(&self) -> usize {
        self.engines.iter().map(BatchedEngine::memory_bytes).sum()
    }

    pub(super) fn dn_rates(&self, b: usize, t: usize, out: &mut [f32]) {
        let (p, local) = self.locate(b);
        self.engines[p].dn_rates(local, t, out);
    }

    pub(super) fn type_mean_rates(&self, b: usize, t: usize, out: &mut [f32]) {
        let (p, local) = self.locate(b);
        self.engines[p].type_mean_rates(local, t, out);
    }

    /// The group of sequence `b` and its index inside the group.
    fn locate(&self, b: usize) -> (usize, usize) {
        assert!(b < *self.starts.last().unwrap_or(&0), "sequence index out of range");
        let p = self.starts.partition_point(|&s| s <= b) - 1;
        (p, b - self.starts[p])
    }
}

/// Runs `job(group, engine)` for every group concurrently, each on its own pool, and returns the
/// results in group order. A panic in a group is re-raised.
fn run_groups<R: Send>(parts: &mut Parts, job: impl Fn(usize, &mut BatchedEngine) -> R + Sync) -> Vec<R> {
    let job = &job;
    std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .engines
            .iter_mut()
            .enumerate()
            .map(|(p, engine)| {
                let pool = &parts.pools[p % parts.pools.len()];
                scope.spawn(move || pool.install(|| job(p, engine)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

impl BatchedEngine {
    /// Splits the batch over `K` sub-engines when the engine was configured with
    /// [`BatchedEngine::with_subengines`] and the batch has at least two cells; `None` = run as a
    /// single engine. Each group gets `1 / K` of the memory cap.
    pub(super) fn forward_split(
        &mut self,
        model: &FlyModel,
        seqs: &[BatchedSeqInput<'_>],
        opts: &BatchedForwardOptions,
    ) -> Option<Result<(), MemoryCapExceeded>> {
        let batch = seqs.len();
        let nb = batch.div_ceil(LANES);
        let k = self.subengines.min(nb);
        if k < 2 {
            return None;
        }
        // Whole cells per group, the first `rem` groups one more.
        let (base, rem) = (nb / k, nb % k);
        let mut starts = vec![0usize];
        let mut cells = 0;
        for p in 0..k {
            cells += base + usize::from(p < rem);
            starts.push((cells * LANES).min(batch));
        }
        if self.parts.as_ref().is_none_or(|p| p.engines.len() != k) {
            // Pools are sized from the calling pool: at most one pool per thread, `threads / pools`
            // threads each.
            let threads = rayon::current_num_threads().max(1);
            let pool_count = k.min(threads);
            let threads_each = (threads / pool_count).max(1);
            let pools = (0..pool_count)
                .map(|_| rayon::ThreadPoolBuilder::new().num_threads(threads_each).build())
                .collect::<Result<Vec<_>, _>>()
                .expect("sub-engine thread pool");
            let engines = (0..k).map(|_| BatchedEngine::single(self.plan.clone())).collect();
            self.parts = Some(Parts {
                engines,
                pools,
                starts: Vec::new(),
            });
        }
        let parts = self.parts.as_mut().expect("parts");
        parts.starts = starts;
        self.run = None;
        let sub_opts = BatchedForwardOptions {
            memory_cap_bytes: opts.memory_cap_bytes.map(|c| (c / k).max(1)),
            ..*opts
        };
        let starts = parts.starts.clone();
        let results = run_groups(parts, |p, engine| {
            engine.forward(model, &seqs[starts[p]..starts[p + 1]], &sub_opts)
        });
        Some(results.into_iter().collect::<Result<Vec<()>, _>>().map(|_| ()))
    }

    /// The backward pass of a split batch: every group's gradients, summed in group order.
    pub(super) fn backward_split(
        &mut self,
        model: &FlyModel,
        seqs: &[BatchedSeqGrad<'_>],
        want_v_init_grad: bool,
    ) -> BatchedGradients {
        let parts = self.parts.as_mut().expect("backward_split: no split forward pass");
        assert_eq!(
            seqs.len(),
            *parts.starts.last().unwrap_or(&0),
            "BatchedEngine::backward: one BatchedSeqGrad per sequence"
        );
        let starts = parts.starts.clone();
        let results = run_groups(parts, |p, engine| {
            engine.backward(model, &seqs[starts[p]..starts[p + 1]], want_v_init_grad)
        });
        let mut iter = results.into_iter();
        let mut out = iter.next().expect("at least one group");
        for part in iter {
            out.grad.add_assign(&part.grad);
            out.grad_inputs.extend(part.grad_inputs);
            if let (Some(all), Some(more)) = (out.grad_v_init.as_mut(), part.grad_v_init) {
                all.extend(more);
            }
        }
        out
    }
}
