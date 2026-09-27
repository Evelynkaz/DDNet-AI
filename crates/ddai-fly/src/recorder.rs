//! [`TrajectoryRecorder`]: an optional, pre-allocated buffer that
//! [`crate::state::FlyState::step_decision_recording`] fills with each substep's `V`/`r` — the
//! hook the constraints ask for ("access to `r` per substep for a later backward pass") without
//! `step_decision` itself ever allocating. Nothing here implements a backward pass (7.2); this is
//! just storage plus accessors.

/// Records `V`/`f(V)` for up to `capacity_substeps` substeps of one decision, each `num_neurons`
/// long. Allocates once, in [`TrajectoryRecorder::new`]; every subsequent `record` call
/// (invoked by `step_decision_recording`, at most once per substep) is a couple of `copy_from_slice`
/// calls into already-allocated storage.
///
/// **Exact `(V, r)` pairing** (review round 1, F8 — spelled out here because 7.2's backward pass
/// will need to rely on it): substep `k`'s recorded entry is `(V after update k, r that *drove*
/// update k)`, i.e. `r == f(V after update k-1)` (or `f(V)` from before the decision at all, for
/// `k == 0`) — **not** `f(V after update k)`. This is the same `r` `step_decision`'s exponential-
/// Euler update actually multiplied by `w_ij` to compute that substep's `V_∞`, which is what a
/// backward pass differentiating through that multiplication needs. One consequence: the very
/// last substep's *resulting* rate (`f(V)` after the decision's final update — what
/// `DecisionOutput::dn_rates` partly averages in) is deliberately **not** stored anywhere in this
/// recorder; recompute it from `v_at(capacity_substeps - 1)` if a backward pass ever needs it.
#[derive(Debug, Clone)]
pub struct TrajectoryRecorder {
    num_neurons: usize,
    capacity_substeps: usize,
    /// How many substeps have been recorded since the last [`TrajectoryRecorder::reset`].
    cursor: usize,
    v_buf: Vec<f32>,
    r_buf: Vec<f32>,
}

impl TrajectoryRecorder {
    pub fn new(num_neurons: usize, capacity_substeps: usize) -> Self {
        TrajectoryRecorder {
            num_neurons,
            capacity_substeps,
            cursor: 0,
            v_buf: vec![0.0; num_neurons * capacity_substeps],
            r_buf: vec![0.0; num_neurons * capacity_substeps],
        }
    }

    /// Rewinds without freeing the buffers, so recording the next decision's trajectory reuses
    /// the same allocation.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// How many substeps have been recorded since the last [`TrajectoryRecorder::reset`].
    pub fn len(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.cursor == 0
    }

    pub fn capacity_substeps(&self) -> usize {
        self.capacity_substeps
    }

    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Appends one substep's `V`/`r` (each `num_neurons` long; panics if either has the wrong
    /// length). If the recorder is already at `capacity_substeps`, the sample is silently dropped
    /// rather than growing the buffer — `step_decision_recording` must stay allocation-free, so
    /// overflow is the caller's responsibility (size the recorder for at least
    /// `substeps_per_decision`, or call `reset` between decisions).
    pub(crate) fn record(&mut self, v: &[f32], r: &[f32]) {
        assert_eq!(v.len(), self.num_neurons);
        assert_eq!(r.len(), self.num_neurons);
        if self.cursor >= self.capacity_substeps {
            return;
        }
        let start = self.cursor * self.num_neurons;
        let end = start + self.num_neurons;
        self.v_buf[start..end].copy_from_slice(v);
        self.r_buf[start..end].copy_from_slice(r);
        self.cursor += 1;
    }

    /// `V` recorded for substep `substep` (`0`-indexed, `< len()`). Panics out of range.
    pub fn v_at(&self, substep: usize) -> &[f32] {
        assert!(
            substep < self.cursor,
            "v_at({substep}): only {} substeps recorded",
            self.cursor
        );
        let start = substep * self.num_neurons;
        &self.v_buf[start..start + self.num_neurons]
    }

    /// `f(V)` recorded for substep `substep` (`0`-indexed, `< len()`). Panics out of range.
    pub fn r_at(&self, substep: usize) -> &[f32] {
        assert!(
            substep < self.cursor,
            "r_at({substep}): only {} substeps recorded",
            self.cursor
        );
        let start = substep * self.num_neurons;
        &self.r_buf[start..start + self.num_neurons]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_up_to_capacity_then_drops_further_samples() {
        let mut rec = TrajectoryRecorder::new(3, 2);
        rec.record(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3]);
        rec.record(&[4.0, 5.0, 6.0], &[0.4, 0.5, 0.6]);
        assert_eq!(rec.len(), 2);
        rec.record(&[7.0, 8.0, 9.0], &[0.7, 0.8, 0.9]); // dropped: over capacity
        assert_eq!(rec.len(), 2);
        assert_eq!(rec.v_at(0), &[1.0, 2.0, 3.0]);
        assert_eq!(rec.v_at(1), &[4.0, 5.0, 6.0]);
        assert_eq!(rec.r_at(1), &[0.4, 0.5, 0.6]);
    }

    #[test]
    fn reset_rewinds_without_clearing_capacity() {
        let mut rec = TrajectoryRecorder::new(2, 3);
        rec.record(&[1.0, 1.0], &[0.0, 0.0]);
        rec.reset();
        assert_eq!(rec.len(), 0);
        assert!(rec.is_empty());
        assert_eq!(rec.capacity_substeps(), 3);
        rec.record(&[9.0, 9.0], &[0.0, 0.0]);
        assert_eq!(rec.v_at(0), &[9.0, 9.0]);
    }

    #[test]
    #[should_panic]
    fn v_at_out_of_range_panics() {
        let rec = TrajectoryRecorder::new(2, 3);
        rec.v_at(0);
    }

    #[test]
    #[should_panic]
    fn record_with_wrong_length_panics() {
        let mut rec = TrajectoryRecorder::new(3, 1);
        rec.record(&[1.0, 2.0], &[0.1, 0.2]);
    }
}
