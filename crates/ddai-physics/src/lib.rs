//! `ddai-physics` will hold the Rust port of DDNet's server-side physics: character movement,
//! collision against the map's tile layers, tuning parameters (including tune zones), weapons,
//! and world stepping — parameterized over the floating-point scalar (`f32`/`f64`) so it can be
//! checked bit-for-bit against the C++ DDNet reference and against the legacy TypeScript bot.
//!
//! This crate is currently a placeholder: the port lands in later tasks (see `docs/PLAN.md`
//! §1.1). It intentionally has no dependencies yet.

/// Placeholder so `cargo test` exercises this crate before the physics port lands.
///
/// Replace or remove once real physics tests exist.
#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds_and_tests_run() {
        assert_eq!(1 + 1, 2);
    }
}
