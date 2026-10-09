//! The controls of FLY.md section 1.3 for the `Gm` neuron model: the permuted type signs, the
//! degree-preserving rewired graph, and the L/R-tied descriptor keys. (The third control, an
//! MLP/GRU with the same number of parameters, is `ddai-controls`.)

use std::collections::{HashMap, HashSet};

use ddai_flyg::Flyg;

use crate::rng::SplitMix64;

/// Fisher-Yates shuffle of `v` driven by splitmix64 (deterministic from `seed`).
pub(crate) fn shuffle_in_place(v: &mut [f32], seed: u64) {
    let mut rng = SplitMix64::new(seed ^ 0x5149_4E53_5348_5546);
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// A degree-preserving rewiring of a post-major CSR graph by double edge swaps.
///
/// Edges `(a -> b)` and `(c -> d)` become `(a -> d)` and `(c -> b)`, with each edge's synapse
/// count staying with its source. A swap is refused if it would make a self loop or a duplicate
/// edge. Every neuron keeps its in-degree (the row lengths do not change) and its out-degree and
/// the multiset of the synapse counts of its outgoing edges; its presynaptic partners and
/// the in-strength change, which is the point of the control. About `10 * nnz` swaps are attempted.
/// Returns `(pre_index, synapse_count)` in the same post-major layout, each row sorted by source.
pub(crate) fn rewire_degree_preserving(
    row_start: &[u32],
    pre_index: &[u32],
    synapse_count: &[u32],
    seed: u64,
) -> (Vec<u32>, Vec<u32>) {
    let n = row_start.len() - 1;
    let nnz = pre_index.len();
    let src: Vec<u32> = pre_index.to_vec();
    let mut dst: Vec<u32> = vec![0; nnz];
    for post in 0..n {
        for e in row_start[post] as usize..row_start[post + 1] as usize {
            dst[e] = post as u32;
        }
    }
    let count = synapse_count.to_vec();
    let key = |a: u32, b: u32| (u64::from(a) << 32) | u64::from(b);
    let mut present: HashSet<u64> = (0..nnz).map(|e| key(src[e], dst[e])).collect();
    let mut rng = SplitMix64::new(seed ^ 0x5245_5749_5245_4450);
    if nnz >= 2 {
        for _ in 0..10 * nnz {
            let e1 = (rng.next_u64() % nnz as u64) as usize;
            let e2 = (rng.next_u64() % nnz as u64) as usize;
            if e1 == e2 {
                continue;
            }
            let (a, b) = (src[e1], dst[e1]);
            let (c, d) = (src[e2], dst[e2]);
            if b == d || a == d || c == b {
                continue;
            }
            if present.contains(&key(a, d)) || present.contains(&key(c, b)) {
                continue;
            }
            present.remove(&key(a, b));
            present.remove(&key(c, d));
            present.insert(key(a, d));
            present.insert(key(c, b));
            dst[e1] = d;
            dst[e2] = b;
        }
    }
    // Back to post-major rows, each sorted by source.
    let mut order: Vec<usize> = (0..nnz).collect();
    order.sort_by_key(|&e| (dst[e], src[e]));
    let pre_out: Vec<u32> = order.iter().map(|&e| src[e]).collect();
    let cnt_out: Vec<u32> = order.iter().map(|&e| count[e]).collect();
    (pre_out, cnt_out)
}

/// Descriptor keys for [`crate::gm::GmDescriptors::PerNeuronTied`]: neurons with the same
/// `(type, group_id)` share a key (the L/R copies of one cell); a neuron without a `group_id`
/// has a key of its own. Returns `(key per neuron, number of keys)`, keys numbered by first appearance.
pub(crate) fn tied_keys(flyg: &Flyg) -> (Vec<u32>, usize) {
    let mut map: HashMap<(u32, i64), u32> = HashMap::new();
    let mut next = 0u32;
    let mut keys = Vec::with_capacity(flyg.neurons.len());
    for n in &flyg.neurons {
        let k = match n.group_id {
            Some(g) => *map.entry((n.type_index, g)).or_insert_with(|| {
                next += 1;
                next - 1
            }),
            None => {
                next += 1;
                next - 1
            }
        };
        keys.push(k);
    }
    (keys, next as usize)
}
