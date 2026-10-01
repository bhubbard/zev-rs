//! Semantic sieve for zev-rs leveraging dense vector representations and MLX embeddings.
//!
//! Bridges the gap between sub-6µs SIMD lexical heuristics and 150ms full LLM generation,
//! allowing high-confidence semantic routing in sub-millisecond execution time.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SieveResult {
    pub candidate_id: String,
    pub top_score: f64,
    pub margin: f64,
    pub scores: BTreeMap<String, f64>,
    pub decisive: bool,
}

#[derive(Debug, Clone)]
pub struct CandidateVector {
    pub id: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SemanticSieve {
    pub candidates: Vec<CandidateVector>,
    pub margin_threshold: f64,
    pub min_confidence: f64,
}

impl SemanticSieve {
    pub fn new(margin_threshold: f64, min_confidence: f64) -> Self {
        Self {
            candidates: Vec::new(),
            margin_threshold,
            min_confidence,
        }
    }

    /// Add a candidate with a pre-normalized dense vector
    pub fn add_candidate(&mut self, id: impl Into<String>, mut vector: Vec<f32>) {
        Self::normalize_l2(&mut vector);
        self.candidates.push(CandidateVector {
            id: id.into(),
            vector,
        });
    }

    /// L2 normalize a slice of floats in-place
    pub fn normalize_l2(v: &mut [f32]) {
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-9 {
            let inv = 1.0 / norm;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    /// Compiles this `SemanticSieve` into a Metal GPU-accelerated `MlxSemanticSieve`.
    #[cfg(feature = "mlx")]
    pub fn to_mlx(&self) -> crate::error::Result<crate::mlx::MlxSemanticSieve> {
        let pairs: Vec<(String, Vec<f32>)> = self
            .candidates
            .iter()
            .map(|c| (c.id.clone(), c.vector.clone()))
            .collect();
        crate::mlx::MlxSemanticSieve::from_candidates(
            &pairs,
            self.margin_threshold,
            self.min_confidence,
        )
    }

    /// Dot product between two normalized vectors (cosine similarity)
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut dot = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
        }
        dot
    }

    /// Clusters vectors on CPU using pairwise cosine similarity.
    pub fn cluster_vectors_cpu(
        vectors: &[Vec<f32>],
        similarity_threshold: f32,
    ) -> Vec<(usize, Vec<usize>)> {
        let b = vectors.len();
        let mut visited = vec![false; b];
        let mut clusters = Vec::new();

        for i in 0..b {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            let mut members = vec![i];

            for j in (i + 1)..b {
                if visited[j] {
                    continue;
                }
                let sim = Self::cosine_similarity(&vectors[i], &vectors[j]);
                if sim >= similarity_threshold {
                    visited[j] = true;
                    members.push(j);
                }
            }

            clusters.push((i, members));
        }

        clusters.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));
        clusters
    }

    /// Evaluates a query embedding vector against all candidates
    pub fn evaluate_vector(&self, query: &[f32]) -> Option<SieveResult> {
        if self.candidates.is_empty() {
            return None;
        }

        let mut norm_query = query.to_vec();
        Self::normalize_l2(&mut norm_query);

        let mut scores: Vec<(String, f64)> = self
            .candidates
            .iter()
            .map(|c| {
                let score = Self::cosine_similarity(&norm_query, &c.vector) as f64;
                (c.id.clone(), score)
            })
            .collect();

        // Sort descending by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top1 = scores[0].clone();
        let top2_score = if scores.len() > 1 { scores[1].1 } else { 0.0 };
        let margin = top1.1 - top2_score;

        let decisive = margin >= self.margin_threshold && top1.1 >= self.min_confidence;

        let score_map: BTreeMap<String, f64> = scores.into_iter().collect();

        Some(SieveResult {
            candidate_id: top1.0,
            top_score: top1.1,
            margin,
            scores: score_map,
            decisive,
        })
    }

    /// Generate lightweight deterministic pseudo-embedding from text n-grams
    /// for fast testing and zero-weight fallback environments
    pub fn hash_embed(text: &str, dims: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; dims];
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        for (w_idx, word) in words.iter().enumerate() {
            let mut h = 2166136261u32;
            for b in word.bytes() {
                h ^= b as u32;
                h = h.wrapping_mul(16777619);
            }
            let idx = (h as usize) % dims;
            let weight = 1.0 / (1.0 + (w_idx as f32 * 0.05));
            vec[idx] += weight;

            // Character tri-grams
            if word.len() >= 3 {
                for window in word.as_bytes().windows(3) {
                    let mut tri_h = 2166136261u32;
                    for &b in window {
                        tri_h ^= b as u32;
                        tri_h = tri_h.wrapping_mul(16777619);
                    }
                    let tri_idx = (tri_h as usize) % dims;
                    vec[tri_idx] += 0.5 * weight;
                }
            }
        }

        Self::normalize_l2(&mut vec);
        vec
    }
}

/// Request payload for remote embedding endpoints (/v1/embeddings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteEmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
}

/// Individual item in remote embedding response data array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteEmbeddingItem {
    #[serde(default)]
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// Standard response payload for remote embeddings (compatible with apfel-rs, Ollama, OpenAI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteEmbeddingResponse {
    #[serde(default)]
    pub data: Vec<RemoteEmbeddingItem>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// Remote HTTP embedding client querying `/v1/embeddings` (compatible with `apfel serve`, Ollama, vLLM).
#[derive(Debug, Clone)]
pub struct RemoteEmbeddingProvider {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

impl RemoteEmbeddingProvider {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api_key: None,
            timeout_secs: 10,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Creates provider from environment variables:
    /// - `ZEV_EMBEDDING_URL` or `APFEL_EMBEDDING_URL` or `APFEL_URL` (appends `/v1/embeddings`)
    /// - `ZEV_EMBEDDING_MODEL` (defaults to "default")
    /// - `ZEV_EMBEDDING_API_KEY` (optional)
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("ZEV_EMBEDDING_URL")
            .or_else(|_| std::env::var("APFEL_EMBEDDING_URL"))
            .or_else(|_| {
                std::env::var("APFEL_URL")
                    .map(|base| format!("{}/v1/embeddings", base.trim_end_matches('/')))
            })
            .ok()?;

        let model = std::env::var("ZEV_EMBEDDING_MODEL").unwrap_or_else(|_| "default".to_string());
        let api_key = std::env::var("ZEV_EMBEDDING_API_KEY").ok();

        Some(Self {
            endpoint,
            model,
            api_key,
            timeout_secs: 10,
        })
    }

    /// Parse response body string into normalized float vectors.
    pub fn parse_response_body(body: &str) -> crate::error::Result<Vec<Vec<f32>>> {
        let parsed: RemoteEmbeddingResponse = serde_json::from_str(body).map_err(|e| {
            crate::error::ZevError::Internal(format!(
                "Failed parsing embedding JSON: {e} (body: {body})"
            ))
        })?;

        let mut vectors = if !parsed.data.is_empty() {
            let mut sorted = parsed.data;
            sorted.sort_by_key(|item| item.index);
            sorted.into_iter().map(|item| item.embedding).collect()
        } else if let Some(single) = parsed.embedding {
            vec![single]
        } else {
            return Err(crate::error::ZevError::Internal(
                "Remote embedding response contained neither 'data' nor 'embedding'".to_string(),
            ));
        };

        for v in vectors.iter_mut() {
            SemanticSieve::normalize_l2(v);
        }
        Ok(vectors)
    }

    /// Embed a single text string into a normalized dense vector.
    pub fn embed_text(&self, text: &str) -> crate::error::Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text])?;
        results.pop().ok_or_else(|| {
            crate::error::ZevError::Internal("Remote embedding returned empty vector".to_string())
        })
    }

    /// Embed a batch of texts into normalized dense vectors.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn embed_batch(&self, texts: &[&str]) -> crate::error::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let payload = RemoteEmbeddingRequest {
            model: self.model.clone(),
            input: texts.iter().map(|s| s.to_string()).collect(),
        };

        let mut req = ureq::post(&self.endpoint);
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", &format!("Bearer {}", key));
        }

        let mut resp = req.send_json(&payload).map_err(|e| {
            crate::error::ZevError::Internal(format!("Embedding HTTP request failed: {e}"))
        })?;

        use std::io::Read;
        let mut body_str = String::new();
        resp.body_mut()
            .as_reader()
            .read_to_string(&mut body_str)
            .map_err(|e| {
                crate::error::ZevError::Internal(format!("Failed reading response body: {e}"))
            })?;

        Self::parse_response_body(&body_str)
    }

    /// Embed a batch of texts into normalized dense vectors.
    #[cfg(target_arch = "wasm32")]
    pub fn embed_batch(&self, _texts: &[&str]) -> crate::error::Result<Vec<Vec<f32>>> {
        Err(crate::error::ZevError::Internal(
            "Remote embeddings via ureq are not supported on wasm32".into(),
        ))
    }

    /// Populate a `SemanticSieve` with candidates by embedding their text descriptions.
    pub fn populate_sieve(
        &self,
        sieve: &mut SemanticSieve,
        candidates: &[(&str, &str)],
    ) -> crate::error::Result<()> {
        let texts: Vec<&str> = candidates.iter().map(|(_, desc)| *desc).collect();
        let vectors = self.embed_batch(&texts)?;

        for ((id, _), vec) in candidates.iter().zip(vectors) {
            sieve.add_candidate(*id, vec);
        }

        Ok(())
    }

    /// Queries the remote provider to embed `query_text` and evaluates against the sieve.
    pub fn route_query(
        &self,
        sieve: &SemanticSieve,
        query_text: &str,
    ) -> crate::error::Result<Option<SieveResult>> {
        let vec = self.embed_text(query_text)?;
        Ok(sieve.evaluate_vector(&vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remote_embedding_parse_v1_format() {
        let json_payload = r#"{
            "object": "list",
            "data": [
                { "object": "embedding", "index": 1, "embedding": [0.0, 3.0, 4.0] },
                { "object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0] }
            ],
            "model": "text-embedding-3-small"
        }"#;

        let parsed = RemoteEmbeddingProvider::parse_response_body(json_payload)
            .expect("Parsing OpenAI/apfel-rs format must succeed");

        assert_eq!(parsed.len(), 2);
        // Index 0 was [1.0, 0.0, 0.0], normalized = [1.0, 0.0, 0.0]
        assert!((parsed[0][0] - 1.0).abs() < 1e-5);
        // Index 1 was [0.0, 3.0, 4.0], norm = 5.0, normalized = [0.0, 0.6, 0.8]
        assert!((parsed[1][1] - 0.6).abs() < 1e-5);
        assert!((parsed[1][2] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_remote_embedding_parse_ollama_format() {
        let json_payload = r#"{
            "embedding": [0.0, 5.0, 12.0]
        }"#;

        let parsed = RemoteEmbeddingProvider::parse_response_body(json_payload)
            .expect("Parsing Ollama direct format must succeed");

        assert_eq!(parsed.len(), 1);
        // [0.0, 5.0, 12.0], norm = 13.0, normalized = [0.0, 5/13, 12/13]
        let expected_y = 5.0 / 13.0;
        let expected_z = 12.0 / 13.0;
        assert!((parsed[0][1] - expected_y).abs() < 1e-5);
        assert!((parsed[0][2] - expected_z).abs() < 1e-5);
    }

    #[test]
    fn test_semantic_sieve_decisive_routing() {
        let mut sieve = SemanticSieve::new(0.20, 0.40);

        // Pre-embed candidates into 64-dim vector space
        let billing_vec = SemanticSieve::hash_embed(
            "invoice billing payment charge refund payment subscription",
            64,
        );
        let tech_vec =
            SemanticSieve::hash_embed("database server outage cluster connection error crash", 64);
        let sales_vec = SemanticSieve::hash_embed(
            "enterprise contract discount pricing custom quote sales",
            64,
        );

        sieve.add_candidate("billing", billing_vec);
        sieve.add_candidate("tech", tech_vec);
        sieve.add_candidate("sales", sales_vec);

        // Test Query 1: Billing inquiry
        let query_billing =
            SemanticSieve::hash_embed("I have an unexpected charge on my subscription invoice", 64);
        let res1 = sieve
            .evaluate_vector(&query_billing)
            .expect("Evaluation should succeed");

        assert_eq!(res1.candidate_id, "billing");
        assert!(res1.decisive);
        assert!(res1.margin >= 0.20);

        // Test Query 2: Technical outage
        let query_tech = SemanticSieve::hash_embed(
            "Production database has connection error and server crashed",
            64,
        );
        let res2 = sieve
            .evaluate_vector(&query_tech)
            .expect("Evaluation should succeed");

        assert_eq!(res2.candidate_id, "tech");
        assert!(res2.decisive);
    }

    #[test]
    fn test_remote_embedding_builder_and_env() {
        let p = RemoteEmbeddingProvider::new("http://localhost:11434/v1/embeddings", "nomic-embed")
            .with_api_key("secret-key")
            .with_timeout_secs(30);

        assert_eq!(p.endpoint, "http://localhost:11434/v1/embeddings");
        assert_eq!(p.model, "nomic-embed");
        assert_eq!(p.api_key.as_deref(), Some("secret-key"));
        assert_eq!(p.timeout_secs, 30);

        // from_env with ZEV_EMBEDDING_URL
        std::env::set_var("ZEV_EMBEDDING_URL", "http://custom-host:8080/v1/embeddings");
        std::env::set_var("ZEV_EMBEDDING_MODEL", "custom-model");
        std::env::set_var("ZEV_EMBEDDING_API_KEY", "env-key");
        let p_env = RemoteEmbeddingProvider::from_env().expect("Must read from env");
        assert_eq!(p_env.endpoint, "http://custom-host:8080/v1/embeddings");
        assert_eq!(p_env.model, "custom-model");
        assert_eq!(p_env.api_key.as_deref(), Some("env-key"));
        std::env::remove_var("ZEV_EMBEDDING_URL");
        std::env::remove_var("ZEV_EMBEDDING_MODEL");
        std::env::remove_var("ZEV_EMBEDDING_API_KEY");

        // from_env with APFEL_URL (trailing slash)
        std::env::set_var("APFEL_URL", "http://localhost:9000/");
        let p_apfel = RemoteEmbeddingProvider::from_env().expect("Must read from APFEL_URL");
        assert_eq!(p_apfel.endpoint, "http://localhost:9000/v1/embeddings");
        std::env::remove_var("APFEL_URL");
    }

    #[test]
    fn test_remote_embedding_parse_errors() {
        assert!(RemoteEmbeddingProvider::parse_response_body("").is_err());
        assert!(RemoteEmbeddingProvider::parse_response_body("{bad-json").is_err());
        assert!(RemoteEmbeddingProvider::parse_response_body("{}").is_err());
    }

    #[test]
    fn test_normalize_l2_zero_and_tiny() {
        let mut zero = vec![0.0f32; 4];
        SemanticSieve::normalize_l2(&mut zero);
        assert_eq!(zero, vec![0.0f32; 4]);

        let mut tiny = vec![1e-11f32; 4];
        SemanticSieve::normalize_l2(&mut tiny);
        assert_eq!(tiny, vec![1e-11f32; 4]);

        let mut normal = vec![3.0f32, 4.0f32];
        SemanticSieve::normalize_l2(&mut normal);
        assert!((normal[0] - 0.6).abs() < 1e-6);
        assert!((normal[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cluster_vectors_cpu_edge_cases() {
        assert!(SemanticSieve::cluster_vectors_cpu(&[], 0.8).is_empty());

        let single = vec![vec![1.0f32, 0.0f32]];
        let clusters = SemanticSieve::cluster_vectors_cpu(&single, 0.8);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].1, vec![0]);

        // Orthogonal vectors
        let orthogonal = vec![vec![1.0f32, 0.0f32], vec![0.0f32, 1.0f32]];
        let ortho_clusters = SemanticSieve::cluster_vectors_cpu(&orthogonal, 0.5);
        assert_eq!(ortho_clusters.len(), 2);
    }

    #[test]
    fn test_semantic_sieve_edge_cases() {
        let empty_sieve = SemanticSieve::new(0.2, 0.5);
        assert!(empty_sieve.evaluate_vector(&[1.0, 0.0]).is_none());

        // Single candidate
        let mut single_sieve = SemanticSieve::new(0.2, 0.5);
        single_sieve.add_candidate("only_one", vec![1.0, 0.0]);
        let res = single_sieve.evaluate_vector(&[1.0, 0.0]).unwrap();
        assert_eq!(res.candidate_id, "only_one");
        assert_eq!(res.margin, res.top_score);
        assert!(res.decisive);

        // Indecisive candidate (low confidence)
        let mut strict_sieve = SemanticSieve::new(0.2, 0.99);
        strict_sieve.add_candidate("c1", vec![1.0, 0.0]);
        strict_sieve.add_candidate("c2", vec![0.0, 1.0]);
        let low_conf_res = strict_sieve.evaluate_vector(&[0.707, 0.707]).unwrap();
        assert!(!low_conf_res.decisive);
    }
}
