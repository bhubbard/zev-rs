//! Fast XXH3 SIMD Hashing utilities.
//!
//! Synthesized from `kadraj` (`xxhash-rust::xxh3`).
//!
//! Provides ~30 GB/s hashing throughput on Apple Silicon / x86_64, replacing
//! slower SipHash (1 GB/s) for state deduplication, context hashing, and cache keys.

use std::hash::{BuildHasher, Hasher};
use xxhash_rust::xxh3::{xxh3_128 as raw_xxh3_128, xxh3_64 as raw_xxh3_64, Xxh3};

/// Compute 64-bit XXH3 hash for a byte slice.
#[inline(always)]
pub fn xxh3_64(data: &[u8]) -> u64 {
    raw_xxh3_64(data)
}

/// Compute 128-bit XXH3 hash for a byte slice.
#[inline(always)]
pub fn xxh3_128(data: &[u8]) -> u128 {
    raw_xxh3_128(data)
}

/// Compute 64-bit fast hash for a state string.
#[inline(always)]
pub fn hash_state_fast(state: &str) -> u64 {
    xxh3_64(state.as_bytes())
}

/// Compute combined fast hash across multiple tokens.
pub fn hash_tokens_fast(tokens: &[&str]) -> u64 {
    let mut hasher = Xxh3::new();
    for token in tokens {
        hasher.update(token.as_bytes());
        hasher.update(b"\0");
    }
    hasher.digest()
}

/// BuildHasher implementation powered by XXH3 for HashMap / HashSet.
#[derive(Clone, Copy, Debug, Default)]
pub struct Xxh3BuildHasher;

impl BuildHasher for Xxh3BuildHasher {
    type Hasher = Xxh3Hasher;
    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        Xxh3Hasher(Xxh3::new())
    }
}

pub struct Xxh3Hasher(Xxh3);

impl Hasher for Xxh3Hasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0.digest()
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_xxh3_hashing_consistency() {
        let text = "Production database is reporting connection refused errors";
        let h1 = hash_state_fast(text);
        let h2 = hash_state_fast(text);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);

        let h_128 = xxh3_128(text.as_bytes());
        assert_ne!(h_128, 0);
    }

    #[test]
    fn test_xxh3_hashset() {
        let mut set: HashSet<&str, Xxh3BuildHasher> = HashSet::with_hasher(Xxh3BuildHasher);
        set.insert("alpha");
        set.insert("beta");
        assert!(set.contains("alpha"));
        assert!(!set.contains("gamma"));
    }
}
