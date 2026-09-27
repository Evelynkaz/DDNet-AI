//! Streaming MD5 + SHA-256 (GCS pins objects by MD5; we additionally pin SHA-256 ourselves,
//! see `manifests/connectome.toml`). Both digests are computed in a single pass over the file so
//! a 508 MB download is only read from disk once.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use md5::Md5;
use sha2::{Digest, Sha256};

/// Size, MD5 and SHA-256 of a file, all computed together by [`hash_file`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHashes {
    pub size: u64,
    pub md5_hex: String,
    pub sha256_hex: String,
}

/// Streams `path` once, feeding every chunk to both hashers.
pub fn hash_file(path: &Path) -> io::Result<FileHashes> {
    let file = File::open(path)?;
    hash_reader(BufReader::with_capacity(1 << 20, file))
}

fn hash_reader<R: Read>(mut reader: R) -> io::Result<FileHashes> {
    let mut md5 = Md5::new();
    let mut sha256 = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    let mut size = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        md5.update(&buf[..n]);
        sha256.update(&buf[..n]);
        size += n as u64;
    }
    Ok(FileHashes {
        size,
        md5_hex: to_hex(&md5.finalize()),
        sha256_hex: to_hex(&sha256.finalize()),
    })
}

/// Lowercase hex encoding without pulling in a `hex` crate dependency for this one use.
pub(crate) fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX_DIGITS[(b >> 4) as usize]);
        out.push(HEX_DIGITS[(b & 0x0f) as usize]);
    }
    out
}

const HEX_DIGITS: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn to_hex_matches_known_bytes() {
        assert_eq!(to_hex(&[0x00, 0xff, 0x10, 0xab]), "00ff10ab");
        assert_eq!(to_hex(&[]), "");
    }

    #[test]
    fn hash_reader_matches_known_vectors_for_empty_input() {
        // md5("") and sha256("") are well-known test vectors.
        let hashes = hash_reader(Cursor::new(Vec::<u8>::new())).unwrap();
        assert_eq!(hashes.size, 0);
        assert_eq!(hashes.md5_hex, "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(
            hashes.sha256_hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn hash_reader_matches_known_vectors_for_abc() {
        let hashes = hash_reader(Cursor::new(b"abc".to_vec())).unwrap();
        assert_eq!(hashes.size, 3);
        assert_eq!(hashes.md5_hex, "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            hashes.sha256_hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hash_reader_handles_input_larger_than_one_buffer() {
        // Buffer is 1 MiB; make sure multi-chunk reads accumulate correctly rather than only
        // hashing the first chunk.
        let data = vec![0x42u8; (1 << 20) + 12345];
        let hashes = hash_reader(Cursor::new(data.clone())).unwrap();
        assert_eq!(hashes.size, data.len() as u64);

        // Cross-check against a single-shot hash of the same bytes.
        let mut md5 = Md5::new();
        md5.update(&data);
        assert_eq!(hashes.md5_hex, to_hex(&md5.finalize()));
    }
}
