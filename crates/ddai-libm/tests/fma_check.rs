//! The crate rests on `f64::mul_add` being correctly rounded on every platform (D-127): on a target without the `fma` feature it is a call
//! to the C library's `fma`, and the C library of a platform (UCRT, mingw's msvcrt, ...) could in principle emulate it with a plain
//! multiply and add. This test checks the platform's `mul_add` against an exact reference that needs no fused operation: the rounding
//! error of a product, computed with Veltkamp's splitting. If a platform fails this, every other number of `ddai-libm` is suspect, and the
//! failure message says so.

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

#[test]
fn mul_add_is_a_single_rounding_on_this_platform() {
    let mut s = 0x00f4_a5c4_d4e5_f001u64;
    for i in 0..2_000_000u64 {
        // Operands of moderate magnitude, so that neither the product nor its error leaves the normal range.
        let a = f64::from_bits((1023 - 40 + rng(&mut s) % 80) << 52 | (rng(&mut s) & 0x000f_ffff_ffff_ffff))
            * if rng(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
        let b = f64::from_bits((1023 - 40 + rng(&mut s) % 80) << 52 | (rng(&mut s) & 0x000f_ffff_ffff_ffff))
            * if rng(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
        let (p, e) = two_product(a, b);
        // fma(a, b, -p) is exactly the rounding error of the product.
        let got = a.mul_add(b, -p);
        assert_eq!(
            got.to_bits(),
            e.to_bits(),
            "probe {i}: mul_add({a:e}, {b:e}, {:e}) = {got:e}, exact {e:e}: this platform's fma is not fused, so ddai-libm cannot give glibc's bits here",
            -p
        );
    }
}

#[test]
fn mul_add_cancellation_cases() {
    let eps = f64::EPSILON; // 2^-52
    let a = 1.0 + eps;
    let b = 1.0 - eps / 2.0;
    // a * b = 1 - 2^-105 + ... rounds to 1.0; the fused result keeps what the plain one loses.
    let fused = a.mul_add(b, -1.0);
    let plain = a * b - 1.0;
    assert_eq!(plain, 0.0);
    assert_ne!(
        fused, 0.0,
        "not fused: ddai-libm cannot give glibc's bits on this platform"
    );
    // Signs of zero and the special values keep IEEE semantics.
    assert!(0.0f64.mul_add(f64::INFINITY, 1.0).is_nan());
    assert_eq!(
        1.0f64.mul_add(0.0, -0.0).to_bits(),
        0.0f64.to_bits(),
        "(+0) + (-0) is +0 in round-to-nearest"
    );
    assert_eq!(
        (-1.0f64).mul_add(0.0, -0.0).to_bits(),
        (-0.0f64).to_bits(),
        "(-0) + (-0) is -0"
    );
    assert_eq!(2.0f64.mul_add(3.0, 4.0), 10.0);
    assert!(f64::NAN.mul_add(1.0, 1.0).is_nan());
    assert_eq!(
        f64::MAX.mul_add(2.0, -f64::MAX),
        f64::MAX,
        "the intermediate product does not overflow"
    );
}
