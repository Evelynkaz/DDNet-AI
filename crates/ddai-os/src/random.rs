//! Bytes from the operating system's random generator (`getrandom(2)` on Linux, `ProcessPrng` on Windows): what `/dev/urandom` was.

use std::io;

/// Fills `buf` with random bytes.
pub fn fill(buf: &mut [u8]) -> io::Result<()> {
    getrandom::fill(buf).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_differ_between_calls() {
        let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
        fill(&mut a).unwrap();
        fill(&mut b).unwrap();
        assert_ne!(a, b);
        assert_ne!(a, [0u8; 32]);
    }
}
