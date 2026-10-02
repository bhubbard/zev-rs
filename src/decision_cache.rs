use crate::fast_hash::xxh3_64;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// A thread-safe, high-concurrency decision cache with TTL expiration,
/// capacity bounding, and atomic performance telemetry.
///
/// Ported and synthesized from `frontlane-serp/src/core/cache.rs` and powered
/// by XXH3 64-bit SIMD hashing for sub-microsecond cache lookups on repeated decisions.
#[derive(Debug)]
pub struct DecisionCache<K = u64, V = String> {
    max_capacity: usize,
    ttl: Option<Duration>,
    entries: RwLock<HashMap<K, CacheEntry<V>>>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

#[derive(Debug)]
struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    access_count: AtomicU64,
}

impl<V: Clone> Clone for CacheEntry<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            created_at: self.created_at,
            access_count: AtomicU64::new(self.access_count.load(Ordering::Relaxed)),
        }
    }
}

impl<K, V> DecisionCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Creates a new DecisionCache with specified max capacity and optional TTL.
    pub fn new(max_capacity: usize, ttl: Option<Duration>) -> Self {
        Self {
            max_capacity: max_capacity.max(1),
            ttl,
            entries: RwLock::new(HashMap::with_capacity(max_capacity.min(1024))),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Looks up a key in the cache. Returns Some(value) if found and not expired.
    pub fn get(&self, key: &K) -> Option<V> {
        let read_guard = self.entries.read().ok()?;
        if let Some(entry) = read_guard.get(key) {
            if let Some(ttl) = self.ttl {
                if entry.created_at.elapsed() > ttl {
                    drop(read_guard);
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
            entry.access_count.fetch_add(1, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.value.clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Inserts a key-value pair into the cache, performing bounded eviction if full.
    pub fn insert(&self, key: K, value: V) {
        let mut write_guard = match self.entries.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        // If at capacity and inserting a new key, evict expired or oldest entries
        if write_guard.len() >= self.max_capacity && !write_guard.contains_key(&key) {
            // First pass: purge expired entries
            if let Some(ttl) = self.ttl {
                let now = Instant::now();
                write_guard.retain(|_, entry| now.duration_since(entry.created_at) <= ttl);
            }

            // If still full, evict the entry with the oldest created_at
            if write_guard.len() >= self.max_capacity {
                if let Some((oldest_key, _)) =
                    write_guard.iter().min_by_key(|(_, entry)| entry.created_at)
                {
                    let key_to_remove = oldest_key.clone();
                    write_guard.remove(&key_to_remove);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        write_guard.insert(
            key,
            CacheEntry {
                value,
                created_at: Instant::now(),
                access_count: AtomicU64::new(0),
            },
        );
    }

    /// Returns the number of live entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Returns true if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all entries in the cache.
    pub fn clear(&self) {
        if let Ok(mut g) = self.entries.write() {
            g.clear();
        }
    }

    /// Total lookup hit count.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Total lookup miss count.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Total evictions performed.
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Calculates current hit rate as a fraction between 0.0 and 1.0.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits();
        let total = h + self.misses();
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }
}

/// Helper to generate a 64-bit XXH3 cache key from any serializable request.
pub fn hash_decision_request<T: serde::Serialize>(req: &T) -> Result<u64, serde_json::Error> {
    let bytes = serde_json::to_vec(req)?;
    Ok(xxh3_64(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_decision_cache_hit_and_miss() {
        let cache = DecisionCache::<u64, String>::new(10, Some(Duration::from_secs(60)));
        assert_eq!(cache.get(&12345), None);
        assert_eq!(cache.misses(), 1);

        cache.insert(12345, "best_action".to_string());
        assert_eq!(cache.get(&12345), Some("best_action".to_string()));
        assert_eq!(cache.hits(), 1);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_decision_cache_ttl_expiration() {
        let cache = DecisionCache::<u64, String>::new(10, Some(Duration::from_millis(20)));
        cache.insert(999, "transient_result".to_string());
        assert_eq!(cache.get(&999), Some("transient_result".to_string()));

        sleep(Duration::from_millis(30));
        assert_eq!(cache.get(&999), None);
    }

    #[test]
    fn test_decision_cache_eviction() {
        let cache = DecisionCache::<u64, String>::new(2, None);
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        assert_eq!(cache.len(), 2);

        cache.insert(3, "three".to_string());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.evictions(), 1);
    }

    #[test]
    fn test_hash_decision_request() {
        let req1 = serde_json::json!({"state": "page loaded", "choices": ["click", "scroll"]});
        let req2 = serde_json::json!({"state": "page loaded", "choices": ["click", "scroll"]});
        let req3 = serde_json::json!({"state": "page error", "choices": ["click", "scroll"]});

        let h1 = hash_decision_request(&req1).unwrap();
        let h2 = hash_decision_request(&req2).unwrap();
        let h3 = hash_decision_request(&req3).unwrap();

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
