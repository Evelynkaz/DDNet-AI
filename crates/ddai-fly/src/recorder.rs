//! [`TrajectoryRecorder`]: an optional, pre-allocated buffer that
//! [`crate::state::FlyState::step_decision_recording`] fills with each substep's `V`/`r`/`V_∞` —
//! the hook 7.1's constraints asked for ("access to `r` per substep for a later backward pass"),
//! extended (task 7.2) with `V_∞` (see [`TrajectoryRecorder::v_inf_at`]'s doc comment for why the
//! backward pass needs it) without `step_decision_recording` itself ever allocating. Nothing here
//! *implements* a backward pass — that's [`crate::backward`]; this module is just storage plus
//! accessors.

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
    /// `V_∞` for the same substep (task 7.2) — see [`TrajectoryRecorder::v_inf_at`].
    v_inf_buf: Vec<f32>,
}

impl TrajectoryRecorder {
    pub fn new(num_neurons: usize, capacity_substeps: usize) -> Self {
        TrajectoryRecorder {
            num_neurons,
            capacity_substeps,
            cursor: 0,
            v_buf: vec![0.0; num_neurons * capacity_substeps],
            r_buf: vec![0.0; num_neurons * capacity_substeps],
            v_inf_buf: vec![0.0; num_neurons * capacity_substeps],
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

    /// Appends one substep's `V`/`r`/`V_∞` (each `num_neurons` long; panics if any has the wrong
    /// length). If the recorder is already at `capacity_substeps`, the sample is silently dropped
    /// rather than growing the buffer — `step_decision_recording` must stay allocation-free, so
    /// overflow is the caller's responsibility (size the recorder for at least
    /// `substeps_per_decision` times however many decisions it will be asked to span without a
    /// `reset`, e.g. a whole truncated-BPTT window — task 7.2's [`crate::backward::backward`]
    /// relies on exactly this: calling `step_decision_recording` `T` times in a row without
    /// resetting in between, so one recorder ends up holding the flattened `T * substeps` step
    /// trajectory of a whole window).
    pub(crate) fn record(&mut self, v: &[f32], r: &[f32], v_inf: &[f32]) {
        assert_eq!(v.len(), self.num_neurons);
        assert_eq!(r.len(), self.num_neurons);
        assert_eq!(v_inf.len(), self.num_neurons);
        if self.cursor >= self.capacity_substeps {
            return;
        }
        let start = self.cursor * self.num_neurons;
        let end = start + self.num_neurons;
        self.v_buf[start..end].copy_from_slice(v);
        self.r_buf[start..end].copy_from_slice(r);
        self.v_inf_buf[start..end].copy_from_slice(v_inf);
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

    /// `V_∞` recorded for substep `substep` (`0`-indexed, `< len()`) — `bias + W·r_prev + input`,
    /// i.e. the exponential-Euler update's target value *before* the `decay` blend with the
    /// previous `V` (task 7.2, added on top of 7.1's `V`/`r` pairing). Panics out of range.
    ///
    /// The backward pass needs this to recover `V_∞ - V_prev` (the factor multiplying `decay` in
    /// `V_new = V_prev + decay·(V_∞ - V_prev)`, needed for `dL/dτ`) **without** dividing `V_new -
    /// V_prev` by `decay`, which can be as small as `~0.01` at `τ` near [`crate::config::
    /// FlyConfig::tau_max_s`] and would amplify `V`'s own rounding error by the same factor —
    /// storing `V_∞` directly sidesteps that entirely.
    pub fn v_inf_at(&self, substep: usize) -> &[f32] {
        assert!(
            substep < self.cursor,
            "v_inf_at({substep}): only {} substeps recorded",
            self.cursor
        );
        let start = substep * self.num_neurons;
        &self.v_inf_buf[start..start + self.num_neurons]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_up_to_capacity_then_drops_further_samples() {
        let mut rec = TrajectoryRecorder::new(3, 2);
        rec.record(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3], &[10.0, 20.0, 30.0]);
        rec.record(&[4.0, 5.0, 6.0], &[0.4, 0.5, 0.6], &[40.0, 50.0, 60.0]);
        assert_eq!(rec.len(), 2);
        rec.record(&[7.0, 8.0, 9.0], &[0.7, 0.8, 0.9], &[70.0, 80.0, 90.0]); // dropped: over capacity
        assert_eq!(rec.len(), 2);
        assert_eq!(rec.v_at(0), &[1.0, 2.0, 3.0]);
        assert_eq!(rec.v_at(1), &[4.0, 5.0, 6.0]);
        assert_eq!(rec.r_at(1), &[0.4, 0.5, 0.6]);
        assert_eq!(rec.v_inf_at(0), &[10.0, 20.0, 30.0]);
        assert_eq!(rec.v_inf_at(1), &[40.0, 50.0, 60.0]);
    }

    #[test]
    fn reset_rewinds_without_clearing_capacity() {
        let mut rec = TrajectoryRecorder::new(2, 3);
        rec.record(&[1.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        rec.reset();
        assert_eq!(rec.len(), 0);
        assert!(rec.is_empty());
        assert_eq!(rec.capacity_substeps(), 3);
        rec.record(&[9.0, 9.0], &[0.0, 0.0], &[5.0, 5.0]);
        assert_eq!(rec.v_at(0), &[9.0, 9.0]);
        assert_eq!(rec.v_inf_at(0), &[5.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn v_at_out_of_range_panics() {
        let rec = TrajectoryRecorder::new(2, 3);
        rec.v_at(0);
    }

    #[test]
    #[should_panic]
    fn v_inf_at_out_of_range_panics() {
        let rec = TrajectoryRecorder::new(2, 3);
        rec.v_inf_at(0);
    }

    #[test]
    #[should_panic]
    fn record_with_wrong_length_panics() {
        let mut rec = TrajectoryRecorder::new(3, 1);
        rec.record(&[1.0, 2.0], &[0.1, 0.2], &[0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn record_with_wrong_v_inf_length_panics() {
        let mut rec = TrajectoryRecorder::new(3, 1);
        rec.record(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3], &[0.0, 0.0]);
    }
}
