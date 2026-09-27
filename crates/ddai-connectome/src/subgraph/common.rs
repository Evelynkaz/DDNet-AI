//! Small helpers shared across the `subgraph` submodules.

use ddai_flyg::Side;

use crate::tables::ConnectomeTables;

/// A neuron's [`Side`] (the `.flyg` format's enum), from the source `somaSide` dictionary column.
/// `Side::Unknown` for the rare Traced body with no side annotation at all — deliberately not
/// coerced to `M`, since that would corrupt bilateral-pairing and receptive-field sign logic that
/// specifically distinguish "no side" from "midline".
pub fn neuron_side(tables: &ConnectomeTables, idx: u32) -> Side {
    match tables.neurons.rows[idx as usize]
        .soma_side
        .map(|i| tables.dictionaries.soma_sides[i as usize].as_str())
    {
        Some("L") => Side::L,
        Some("R") => Side::R,
        Some("M") => Side::M,
        _ => Side::Unknown,
    }
}
