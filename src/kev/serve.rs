//! /v1/systemone on the Candle Kev model. Request handlers encode and queue; one model thread takes everything queued
//! when it frees up and runs it as one batch: one padded state pass for the new states, then padded row passes for
//! every question of the batch, each row continuing its own state (kev.serve + kev.cuda_graphs, without the graphs).

use super::model::{LayerState, Model, Past};
use super::{to_answers, to_record, Encoder, RowSpec};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::{Result, Tensor};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{mpsc, Arc};
use std::time::Instant;

const MAX_BATCH: usize = 64; // requests the model thread takes at once
                             // A pass holds at most TOKENS padded key positions (bounds the gathered state caches and activations) and
                             // CELLS attention scores per head (rows x queries x keys: the f32 score buffers, ~4 copies of 16 heads x 4 B each).
                             // ponytail: plain attention, no flash kernel; states past ~2k tokens run one per pass and pay T^2 memory.
const TOKENS: usize = 8192;
const CELLS: usize = 2 << 20;
const MAX_ROWS: usize = 64;
const MAX_QUEUE_DEPTH: usize = 1024;

type JobReply =
    tokio::sync::oneshot::Sender<std::result::Result<(Vec<Vec<f64>>, f64, bool), String>>;

struct Job {
    state: Vec<u32>,
    rows: Vec<RowSpec>,
    reply: JobReply,
}

/// A state's cache, batch size 1: attention k/v [1, kv, S, hd] and DeltaNet conv/recurrent states.
struct StateCache {
    len: usize,
    layers: Vec<LayerState>,
}

/// LRU of state prefixes keyed by their token ids (kev.serve.PrefixCache).
struct Prefixes {
    size: usize,
    order: VecDeque<Vec<u32>>,
    map: HashMap<Vec<u32>, Arc<StateCache>>,
    hits: usize,
    misses: usize,
}

impl Prefixes {
    fn get(&mut self, k: &[u32]) -> Option<Arc<StateCache>> {
        let v = self.map.get(k).cloned()?;
        if self.order.back().is_none_or(|last| last.as_slice() != k) {
            if let Some(pos) = self.order.iter().position(|x| x.as_slice() == k) {
                let item = self.order.remove(pos).unwrap();
                self.order.push_back(item);
            }
        }
        Some(v)
    }
    fn put(&mut self, k: Vec<u32>, v: Arc<StateCache>) {
        if self.size == 0 {
            return;
        }
        if let Some(pos) = self.order.iter().position(|x| *x == k) {
            self.order.remove(pos);
        }
        self.order.push_back(k.clone());
        self.map.insert(k, v);
        while self.order.len() > self.size {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

struct Worker {
    m: Model,
    cache: Prefixes,
}

fn ids_tensor(rows: &[&[u32]], t: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0u32; rows.len() * t];
    for (i, r) in rows.iter().enumerate() {
        v[i * t..i * t + r.len()].copy_from_slice(r);
    }
    Tensor::from_vec(v, (rows.len(), t), dev)
}

fn pos_tensor(starts: &[usize], t: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let v: Vec<u32> = starts
        .iter()
        .flat_map(|&s| (0..t).map(move |j| (s + j) as u32))
        .collect();
    Tensor::from_vec(v, (starts.len(), t), dev)
}

/// Greedy groups of consecutive items (already sorted by length), each item (cached state length, new tokens):
/// a pass of n items padded to (a, b) holds n*(a+b) key positions and n*b*(a+b) scores per head.
fn groups(lens: &[(usize, usize)], cap: usize) -> Vec<std::ops::Range<usize>> {
    let (mut out, mut start, mut a, mut b) = (Vec::new(), 0, 0, 0);
    for (i, &(x, y)) in lens.iter().enumerate() {
        let (na, nb) = (a.max(x), b.max(y));
        let n = i - start + 1;
        if i > start && (n * (na + nb) > TOKENS || n * nb * (na + nb) > CELLS || i - start == cap) {
            out.push(start..i);
            start = i;
            (a, b) = (x, y);
        } else {
            (a, b) = (na, nb);
        }
    }
    if start < lens.len() {
        out.push(start..lens.len());
    }
    out
}

impl Worker {
    /// State passes for distinct new states. -> one cache per state.
    fn states(&self, states: &[&[u32]]) -> Result<Vec<Arc<StateCache>>> {
        let dev = &self.m.dev;
        let mut order: Vec<usize> = (0..states.len()).collect();
        order.sort_by_key(|&i| states[i].len());
        let mut out: Vec<Option<Arc<StateCache>>> = vec![None; states.len()];
        let lens: Vec<(usize, usize)> = order.iter().map(|&i| (0, states[i].len())).collect();
        for g in groups(&lens, 16) {
            let idx = &order[g];
            let t = idx.iter().map(|&i| states[i].len()).max().unwrap();
            let rows: Vec<&[u32]> = idx.iter().map(|&i| states[i]).collect();
            let ls: Vec<u32> = rows.iter().map(|r| r.len() as u32).collect();
            let (_, layers) = self.m.forward(
                &ids_tensor(&rows, t, dev)?,
                &pos_tensor(&vec![0; rows.len()], t, dev)?,
                &ls,
                None,
                true,
            )?;
            for (n, &i) in idx.iter().enumerate() {
                let len = states[i].len();
                let ls = layers
                    .iter()
                    .map(|l| {
                        Ok(match l {
                            LayerState::Attn { k, v } => LayerState::Attn {
                                k: k.narrow(0, n, 1)?.narrow(2, 0, len)?.force_contiguous()?,
                                v: v.narrow(0, n, 1)?.narrow(2, 0, len)?.force_contiguous()?,
                            },
                            LayerState::Gdn { conv, rec } => LayerState::Gdn {
                                conv: conv.narrow(0, n, 1)?.force_contiguous()?,
                                rec: rec.narrow(0, n, 1)?.force_contiguous()?,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                out[i] = Some(Arc::new(StateCache { len, layers: ls }));
            }
        }
        Ok(out.into_iter().map(Option::unwrap).collect())
    }

    /// Row passes: each row continues its state. -> probabilities per row, in input order.
    fn rows(&self, rows: &[(&StateCache, &RowSpec)]) -> Result<Vec<Vec<f64>>> {
        let dev = &self.m.dev;
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| (rows[i].0.len + rows[i].1.ids.len(), rows[i].1.ids.len()));
        let lens: Vec<(usize, usize)> = order
            .iter()
            .map(|&i| (rows[i].0.len, rows[i].1.ids.len()))
            .collect();
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); rows.len()];
        for g in groups(&lens, MAX_ROWS) {
            let idx = &order[g];
            let n = idx.len();
            let smax = idx.iter().map(|&i| rows[i].0.len).max().unwrap();
            let t = idx.iter().map(|&i| rows[i].1.ids.len()).max().unwrap();
            let nlayers = rows[idx[0]].0.layers.len();
            let mut layers = Vec::with_capacity(nlayers);
            for l in 0..nlayers {
                layers.push(match &rows[idx[0]].0.layers[l] {
                    LayerState::Attn { .. } => {
                        let (mut ks, mut vs) = (Vec::with_capacity(n), Vec::with_capacity(n));
                        for &i in idx {
                            let LayerState::Attn { k, v } = &rows[i].0.layers[l] else {
                                unreachable!()
                            };
                            let s = k.dim(2)?;
                            ks.push(k.pad_with_zeros(2, 0, smax - s)?);
                            vs.push(v.pad_with_zeros(2, 0, smax - s)?);
                        }
                        LayerState::Attn {
                            k: Tensor::cat(&ks, 0)?,
                            v: Tensor::cat(&vs, 0)?,
                        }
                    }
                    LayerState::Gdn { .. } => {
                        let (mut cs, mut rs) = (Vec::with_capacity(n), Vec::with_capacity(n));
                        for &i in idx {
                            let LayerState::Gdn { conv, rec } = &rows[i].0.layers[l] else {
                                unreachable!()
                            };
                            cs.push(conv.clone());
                            rs.push(rec.clone());
                        }
                        LayerState::Gdn {
                            conv: Tensor::cat(&cs, 0)?,
                            rec: Tensor::cat(&rs, 0)?,
                        }
                    }
                });
            }
            let past = Past {
                layers,
                plen: idx.iter().map(|&i| rows[i].0.len as u32).collect(),
                smax,
            };
            let ids: Vec<&[u32]> = idx.iter().map(|&i| rows[i].1.ids.as_slice()).collect();
            let starts: Vec<usize> = idx.iter().map(|&i| rows[i].0.len).collect();
            let ls: Vec<u32> = ids.iter().map(|r| r.len() as u32).collect();
            let (h, _) = self.m.forward(
                &ids_tensor(&ids, t, dev)?,
                &pos_tensor(&starts, t, dev)?,
                &ls,
                Some(&past),
                false,
            )?;
            let mut picks = Vec::new();
            let mut ks = Vec::new();
            for (r, &i) in idx.iter().enumerate() {
                let spec = rows[i].1;
                picks.push((r * t + spec.decide) as u32);
                picks.extend(spec.opts.iter().map(|&o| (r * t + o) as u32));
                ks.push(spec.opts.len());
            }
            let x = h
                .reshape((n * t, ()))?
                .index_select(&Tensor::new(picks, dev)?, 0)?;
            for (p, &i) in self.m.head(&x, &ks)?.into_iter().zip(idx) {
                out[i] = p;
            }
        }
        Ok(out)
    }

    fn run(&mut self, jobs: &[Job]) -> Result<Vec<(Vec<Vec<f64>>, bool)>> {
        let mut caches: Vec<Option<Arc<StateCache>>> =
            jobs.iter().map(|j| self.cache.get(&j.state)).collect();
        let hit: Vec<bool> = caches.iter().map(Option::is_some).collect();
        let mut new: Vec<&[u32]> = Vec::new();
        for (j, c) in jobs.iter().zip(&caches) {
            if c.is_none() && !new.contains(&j.state.as_slice()) {
                new.push(&j.state);
            }
        }
        if !new.is_empty() {
            let made = self.states(&new)?;
            for (s, c) in new.iter().zip(made) {
                self.cache.put(s.to_vec(), c.clone());
                for (j, slot) in jobs.iter().zip(caches.iter_mut()) {
                    if slot.is_none() && j.state.as_slice() == *s {
                        *slot = Some(c.clone());
                    }
                }
            }
        }
        self.cache.hits += hit.iter().filter(|h| **h).count();
        self.cache.misses += hit.iter().filter(|h| !**h).count();
        let rows: Vec<(&StateCache, &RowSpec)> = jobs
            .iter()
            .zip(&caches)
            .flat_map(|(j, c)| j.rows.iter().map(move |r| (c.as_deref().unwrap(), r)))
            .collect();
        let mut probs = self.rows(&rows)?.into_iter();
        let mut out = Vec::with_capacity(jobs.len());
        for (j, h) in jobs.iter().zip(hit) {
            let mut j_probs = Vec::with_capacity(j.rows.len());
            for _ in 0..j.rows.len() {
                let p = probs.next().ok_or_else(|| {
                    candle_core::Error::Msg("mismatched row count from model head".into())
                })?;
                j_probs.push(p);
            }
            out.push((j_probs, h));
        }
        Ok(out)
    }
}

#[derive(Clone)]
pub struct KevState {
    tx: mpsc::SyncSender<Job>,
    enc: Arc<Encoder>,
    api_key: Option<String>,
    card: Arc<Value>,
}

pub fn spawn(m: Model, enc: Encoder, prefix_cache: usize, card: Value) -> KevState {
    let (tx, rx) = mpsc::sync_channel::<Job>(MAX_QUEUE_DEPTH);
    std::thread::Builder::new()
        .name("kev-model".into())
        .spawn(move || {
            let mut w = Worker {
                m,
                cache: Prefixes {
                    size: prefix_cache,
                    order: VecDeque::new(),
                    map: HashMap::new(),
                    hits: 0,
                    misses: 0,
                },
            };
            while let Ok(first) = rx.recv() {
                let mut batch = vec![first];
                while batch.len() < MAX_BATCH {
                    match rx.try_recv() {
                        Ok(j) => batch.push(j),
                        Err(_) => break,
                    }
                }
                let t0 = Instant::now();
                let res = w.run(&batch);
                let _ = w.m.dev.synchronize();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                match res {
                    Ok(rs) => {
                        for (j, (p, h)) in batch.into_iter().zip(rs) {
                            let _ = j.reply.send(Ok((p, ms, h)));
                        }
                    }
                    Err(e) => {
                        #[cfg(feature = "cuda")]
                        let mem = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                            .map(|(f, t)| format!("{} of {} MiB free", f >> 20, t >> 20))
                            .unwrap_or_default();
                        #[cfg(not(feature = "cuda"))]
                        let mem = "";
                        eprintln!(
                            "kev: batch of {} ({} rows, longest state {}) failed: {e} {mem}",
                            batch.len(),
                            batch.iter().map(|j| j.rows.len()).sum::<usize>(),
                            batch.iter().map(|j| j.state.len()).max().unwrap_or(0)
                        );
                        for j in batch {
                            let _ = j.reply.send(Err(e.to_string()));
                        }
                    }
                }
            }
        })
        .expect("spawn model thread");
    KevState {
        tx,
        enc: Arc::new(enc),
        api_key: std::env::var("KEV_API_KEY").ok().filter(|k| !k.is_empty()),
        card: Arc::new(card),
    }
}

type Reply = std::result::Result<Json<Value>, (StatusCode, Json<Value>)>;

fn err(code: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"detail": msg.into()})))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn systemone(State(s): State<KevState>, headers: HeaderMap, body: String) -> Reply {
    if let Some(key) = &s.api_key {
        let got = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let expected = format!("Bearer {key}");
        if !constant_time_eq(got.as_bytes(), expected.as_bytes()) {
            return Err(err(
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>",
            ));
        }
    }
    let req: Value = serde_json::from_str(&body)
        .map_err(|e| err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
    let (state, qs) = to_record(&req).map_err(|e| err(StatusCode::UNPROCESSABLE_ENTITY, e))?;
    let (ids, rows) = s
        .enc
        .encode(&state, &qs)
        .map_err(|e| err(StatusCode::UNPROCESSABLE_ENTITY, e))?;
    let tokens = ids.len() + rows.iter().map(|r| r.ids.len()).sum::<usize>();
    let (tx, rx) = tokio::sync::oneshot::channel();
    s.tx.try_send(Job {
        state: ids,
        rows,
        reply: tx,
    })
    .map_err(|e| match e {
        mpsc::TrySendError::Full(_) => err(
            StatusCode::TOO_MANY_REQUESTS,
            "inference queue saturated; please retry shortly",
        ),
        mpsc::TrySendError::Disconnected(_) => {
            err(StatusCode::SERVICE_UNAVAILABLE, "model thread stopped")
        }
    })?;
    let (probs, ms, _hit) = rx
        .await
        .map_err(|_| {
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model thread dropped the request",
            )
        })?
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let answers = to_answers(&probs, &qs);
    let out_tokens = s
        .enc
        .tok
        .encode(serde_json::to_string(&answers).unwrap_or_default(), false)
        .map(|e| e.len())
        .unwrap_or(0);
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("kev-latest");
    Ok(Json(
        json!({"model": model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": out_tokens}, "latency_ms": (ms * 10.0).round() / 10.0}),
    ))
}

async fn models(State(s): State<KevState>) -> Json<Value> {
    let names = ["kev-latest", "jev-latest"];
    Json(
        json!({"models": names.iter().map(|n| { let mut c = (*s.card).clone(); c["name"] = json!(n); c }).collect::<Vec<_>>()}),
    )
}

pub fn router(s: KevState) -> Router {
    Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status": "ok", "engine": "zev-candle"})) }),
        )
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"Bearer secret123", b"Bearer secret123"));
        assert!(!constant_time_eq(b"Bearer secret123", b"Bearer secret124"));
        assert!(!constant_time_eq(b"Bearer secret123", b"Bearer secret12"));
        assert!(!constant_time_eq(b"Bearer secret12", b"Bearer secret123"));
        assert!(!constant_time_eq(b"", b"Bearer secret123"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_prefixes_lru() {
        let mut cache = Prefixes {
            size: 2,
            order: VecDeque::new(),
            map: HashMap::new(),
            hits: 0,
            misses: 0,
        };

        let dummy_cache1 = Arc::new(StateCache {
            len: 1,
            layers: Vec::new(),
        });
        let dummy_cache2 = Arc::new(StateCache {
            len: 2,
            layers: Vec::new(),
        });
        let dummy_cache3 = Arc::new(StateCache {
            len: 3,
            layers: Vec::new(),
        });

        cache.put(vec![1, 2], dummy_cache1.clone());
        cache.put(vec![3, 4], dummy_cache2.clone());

        assert_eq!(cache.get(&[1, 2]).unwrap().len, 1);

        // Putting 3rd item should evict [3, 4] because [1, 2] was recently accessed
        cache.put(vec![5, 6], dummy_cache3.clone());

        assert!(cache.get(&[3, 4]).is_none());
        assert_eq!(cache.get(&[1, 2]).unwrap().len, 1);
        assert_eq!(cache.get(&[5, 6]).unwrap().len, 3);
    }
}
