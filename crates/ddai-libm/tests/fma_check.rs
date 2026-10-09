//! The fused multiply-add the ports use is correctly rounded on this platform (D-127, review F2).
//!
//! `ddai_libm::fma` is the instruction, the C library's `fma` (only after it agreed with the crate's own integer
//! `fma` on a corner-case self-check) or the integer `fma` itself, depending on the target and the CPU; this test
//! checks whichever it is against an exact reference that needs no fused operation (the rounding error of a
//! product, computed with Veltkamp/Dekker splitting), and reports what the platform's own `f64::mul_add` does.
//! A platform whose `mul_add` fails this still gets the right bits from the ports, because the self-check
//! rejects that `fma`; the report line says so.

fn rng(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Veltkamp split: `a = hi + lo` with both halves fitting in 26 bits.
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a; // 2^27 + 1
    let hi = c - (c - a);
    (hi, a - hi)
}

/// Dekker's exact product: `p = a * b` rounded, `e` the exact error, `p + e == a * b` (barring over/underflow).
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

/// Cases that separate a fused operation from `a * b + c`, and the IEEE special values.
fn corner_cases(fma: impl Fn(f64, f64, f64) -> f64) {
    let eps = f64::EPSILON; // 2^-52
    let a = 1.0 + eps;
    let b = 1.0 - eps / 2.0;
    // a * b = 1 + eps/2 - eps^2/2 rounds to 1.0; the fused result keeps what the plain one loses.
    assert_eq!(a * b - 1.0, 0.0);
    assert_ne!(fma(a, b, -1.0), 0.0, "not fused");
    assert!(fma(0.0, f64::INFINITY, 1.0).is_nan());
    assert_eq!(
        fma(1.0, 0.0, -0.0).to_bits(),
        0.0f64.to_bits(),
        "(+0) + (-0) is +0 in round-to-nearest"
    );
    assert_eq!(fma(-1.0, 0.0, -0.0).to_bits(), (-0.0f64).to_bits(), "(-0) + (-0) is -0");
    assert_eq!(fma(2.0, 3.0, 4.0), 10.0);
    assert!(fma(f64::NAN, 1.0, 1.0).is_nan());
    assert_eq!(
        fma(f64::MAX, 2.0, -f64::MAX),
        f64::MAX,
        "the intermediate product does not overflow"
    );
    assert_eq!(fma(f64::MAX, 2.0, 0.0), f64::INFINITY);
    assert_eq!(
        fma(f64::from_bits(1), 0.5, 0.0).to_bits(),
        0,
        "half the smallest subnormal rounds to even"
    );
    assert_eq!(fma(f64::from_bits(1), 0.75, 0.0).to_bits(), 1);
}

#[test]
fn the_fma_the_ports_use_is_a_single_rounding_on_this_platform() {
    println!("ddai-libm fma: {}", ddai_libm::fma_mode_description());
    let mut s = 0x00f4_a5c4_d4e5_f001u64;
    for i in 0..2_000_000u64 {
        // Operands of moderate magnitude, so that neither the product nor its error leaves the normal range.
        let a = f64::from_bits((1023 - 40 + rng(&mut s) % 80) << 52 | (rng(&mut s) & 0x000f_ffff_ffff_ffff))
            * if rng(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
        let b = f64::from_bits((1023 - 40 + rng(&mut s) % 80) << 52 | (rng(&mut s) & 0x000f_ffff_ffff_ffff))
            * if rng(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
        let (p, e) = two_product(a, b);
        // fma(a, b, -p) is exactly the rounding error of the product.
        let got = ddai_libm::fma(a, b, -p);
        assert_eq!(
            got.to_bits(),
            e.to_bits(),
            "probe {i}: fma({a:e}, {b:e}, {:e}) = {got:e}, exact {e:e}: the fma of ddai-libm is not fused",
            -p
        );
    }
}

#[test]
fn the_fma_the_ports_use_passes_the_corner_cases() {
    corner_cases(ddai_libm::fma);
}

#[test]
fn the_software_fma_passes_the_corner_cases_on_every_platform() {
    corner_cases(ddai_libm::soft_fma);
}

/// Not a pass/fail test of the ports: tells, in the test log, whether this platform's own `mul_add` is correct. On
/// a platform where it is not (mingw's `fma` is known to be wrong in corner cases), `ddai-libm` falls back to its
/// own `fma` by itself, which the tests above prove; this line is for the people who read the log.
#[test]
fn report_the_platform_mul_add() {
    let agrees = (0..200_000u64).all(|i| {
        let mut s = i.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let (a, b, c) = (
            f64::from_bits(rng(&mut s)),
            f64::from_bits(rng(&mut s)),
            f64::from_bits(rng(&mut s)),
        );
        let (x, y) = (a.mul_add(b, c), ddai_libm::soft_fma(a, b, c));
        x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
    });
    println!(
        "platform f64::mul_add agrees with the software fma on 200000 random-bit triples: {agrees}; ddai-libm uses: {}",
        ddai_libm::fma_mode_description()
    );
}
