# zev-rs Code Audit

**Date:** 2026-09-24
**Scope:** everything in the working tree, including your uncommitted changes to `engine.rs`, `order_invariant.rs`, `calibration.rs`, `clm.rs` and `lib.rs`. Covers `src/` (about 3,700 lines), the tests, CI, packaging (Cargo, npm, Docker) and README claims.
**Areas:** performance, coding standards, code organization, reliability.

---

## 1. Executive summary

The crate builds. The whole suite passes (2,060 tests), and clippy reports only a handful of default-level warnings. The math core is small and sound: numerically stable softmax, ECE, and golden-section temperature fitting.

The problems are one level up, in how decisions are made and reported:

1. **The lexical scorer misfires in ways that silently flip decisions.** Negation detection matches substrings rather than words. "unresolved" counts as a negation of what follows it, and so do "casino", "prefixed" and "no longer". So "ticket unresolved: database outage still ongoing" routes to **billing at 97%**. A description that starts with "Normal…" is scored as a *negative* polarity option, and one that starts with "Yesterday…" as an *affirmative* one.
2. **Three protocol paths behave like three different engines.** Native `/v1/decisions`, SystemOne `/v1/systemone` and Tev1 each have their own scoring, temperature rules and meaning of "confidence". The same input gives confidence **0.54, 0.81 and 0.89**.
3. **Guardrails leak.** Numeric questions with `allow_abstain: false` return `status: "ok"` and an in-range estimate (72.3) even when 88% of the probability mass says the value is *above* the range. The advertised `MAX_QUESTIONS = 64` is never enforced (1,000 questions are accepted). Tev1 panics in debug builds when a request has more than 190 options.
4. **Some reported numbers are made up.** The CLM fallback reports a fixed confidence of 0.85, and the neural fallback reports fixed probabilities (0.95/0.05), entropy (0.1) and concentration (0.9). These are presented in the same fields as real calibrated output.
5. **Docs and runtime claims don't match the code.** The README promises "zero heap allocations" and "SIMD", but a 3-option request makes **79 heap allocations** and there is no SIMD code. The Candle backend it describes doesn't exist in the crate.
6. **Build and release hygiene has gaps.** `--no-default-features` doesn't compile. `Cargo.lock` is gitignored, but the Dockerfile `COPY`s it, and it pins a Rust version (1.80) that the locked dependencies no longer support (they need 1.85). CI runs coverage only: no clippy, no fmt check, no macOS or `neural` feature build.

None of these are hard to fix. Section 3 is ordered so the first few changes remove most of the risk.

### Scorecard

| Area | Grade | One-line reason |
|---|---|---|
| Performance | B- | Fast for small inputs (about 8 µs per request); O(words × premise length) scanning grows linearly into hundreds of ms on large states; the stack-buffer "optimizations" are undone by heap allocations right after them |
| Coding standards | C+ | Clippy is mostly clean, but the code isn't rustfmt'd (105 diffs), uses stringly-typed status and family fields, uses magic constants, and hard-codes benchmark phrases into the scorer |
| Code organization | C | `evaluate_system_one` is a 280-line function that duplicates the decoder; there are two engine types plus a trait for one job; CLM and prototype code live in the core library with no feature gate |
| Reliability | C- | Substring negation, unenforced limits, a panic path, fabricated confidence, a date-dependent premise, and a public API that can hang |

### How findings were verified

- **Commands run:** `cargo build`, `cargo clippy --all-targets` (default and pedantic), `cargo fmt --check`, `cargo check --no-default-features` and `cargo test` against the committed HEAD. The same checks were run on a rebuild of your uncommitted changes. The one gap is `tests/benchmark_regressions.rs`: it is untracked, and I couldn't transfer it to compile, so it was **reviewed by reading only**.
- **Probe tests:** 15 small tests that exercise specific suspicions (P01–P15, quoted throughout). Results were identical on HEAD and on your working tree, except for the SystemOne probes, which exercise new code.
- **Cargo features:** the `neural` feature (apfel-rs, macOS only) was reviewed by reading only; it was not compiled.

---

## 2. Findings

Severity: **P0** = wrong results or crash reachable from the public API/HTTP · **P1** = significant reliability or maintainability risk · **P2** = worth fixing soon · **P3** = polish.

### 2.1 Reliability

#### R1 (P0): Negation detection matches substrings, not words
`order_invariant.rs:21-29, 33-55`

`is_negated` checks `prefix.ends_with("no")` and similar, and `window.contains("resolved")` and similar, on raw lowercase bytes. It returns `true` if *any* occurrence of the word is "negated", and the 45-byte window crosses sentence boundaries.

| Premise | Word | `is_negated` | Why |
|---|---|---|---|
| "ticket **unresolved**: database outage" | database | true | "unresolved" contains "resolved" |
| "refund for the **casino** deposit" | deposit | true | "casino" ends with "no" |
| "the **prefixed** invoice number" | invoice | true | "prefixed" contains "fixed" |

End to end (P03): the premise "ticket unresolved: database outage still ongoing" with options `database_outage` and `billing` returns **billing, p = 0.972**.

**Fix:** tokenize the premise once into words with byte offsets, and test negators as whole tokens within a clause window (stop at `.`, `;`, `but`). Keep negator lists as `const &[&str]`. Add the table above as regression tests.

#### R2 (P0): Polarity detection keys on string prefixes
`order_invariant.rs:143-144`

`desc_trimmed.starts_with("no")` marks any description beginning "Normal…", "None…", "Notify…" or "November…" as the *negative* boolean pole. `starts_with("yes")` catches "Yesterday…". The rule applies to **every** candidate, not just booleans.

P04, with the neutral premise "please escalate this ticket":

- "Normal priority" scores **−1.0**, while "Standard priority" scores 0.5.
- "Yesterday's orders" scores **2.9**, while "Recent orders" scores 0.5.

The `has_neg` checks (`contains("no ")`, `contains("can ")`) have the same substring problem: "piano ", "scan ". The `has_neg` block is also duplicated verbatim (lines 147-152 and 164-169).

**Fix:** only apply polarity logic to candidates the caller marks as boolean poles (`id == "true"`/`"false"`, or a flag on `Candidate`), and match first *tokens*, not prefixes.

#### R3 (P1): Benchmark phrases are hard-coded into the scorer
`order_invariant.rs:143-144`

`starts_with("every required")` and `starts_with("a condition is missing")` are the exact criteria strings from `tests/benchmark_regressions.rs:224-225`. `"not under any circumstances"` and `"stop renewing"` (lines 28, 46) look like the same pattern. This tunes the engine to the benchmark rather than to the task, and it will silently regress on rephrased criteria. It also makes the benchmark numbers in the README unreliable.

**Fix:** remove the phrase hacks. If you need them, make them data (a configurable lexicon), and keep a held-out benchmark split you never tune against.

#### R4 (P0): Numeric out-of-range is reported as `ok` with an in-range value
`decoding.rs:51-60, 201-217, 265-274`

`__below_range__` and `__above_range__` are *always* added for numeric questions, but the out-of-range status check only runs `if policy.allow_abstain`. With `allow_abstain: false`, P07 shows:

```
probs = {__above_range__: 0.876, __below_range__: 0.093, "0": 0.009, "1": 0.022}
status = "ok", decision = 72.32
```

The expected value is computed only over in-range anchors, so the caller gets a confident mid-range number when the model says "above 100".

**Fix:** decide out-of-range independently of `allow_abstain` (it isn't abstention), or omit the BELOW/ABOVE candidates when the policy forbids them.

#### R5 (P0): Tev1 labels overflow and collide
`tev1.rs:161`

`(b'A' + idx as u8) as char`:

- With 27–190 options, labels run past `Z` into `[`, `\`, `]`, `^` (P06).
- From 191 options, `u8` addition overflows. That **panics in debug builds** (P06 reproduced this) and wraps to control characters in release. There is no option-count limit on `/v1/tev1`.
- Explicit and implicit letters can collide. `["A: one", "A: two", "three"]` returns only `{A, C}`, so one option silently disappears from `probabilities`.

**Fix:** reject more than 26 options (or use `AA`, `AB`, …), detect duplicate labels, and return 400.

#### R6 (P1): Tev1 prompt parser misreads ordinary text
`tev1.rs:48-63, 107-131`

P05:

- `"Options: A: Yes, it is fine. B: No"` parses to **3** options: `["A: Yes, it is", "fine.", "B: No"]`. Any word ending in `.` or `:` is treated as an option label.
- A state line beginning with "Answer…" (for example "Answer the customer politely") **stops parsing**, and the request fails with "Missing 'Question'".
- A line beginning with "Statement:" is taken as a `State` header. The resulting text is `"Order arrived broken.ment: customer attached photos."`

**Fix:** match headers with an anchored pattern (`^(state|question|options|answer)\s*[:\t]|\s{2,}`), and option labels with `^[A-Z][:.)]\s` on single letters only.

#### R7 (P1): SystemOne `noul` answers the wording of the question, not the state
`engine.rs:195-202`

For `noul` questions the instructions are appended to the premise, and the criteria are then matched against that premise. P14, with the unrelated state "The weather is nice today.":

- "Is the refund approved?" gives `p_true = 0.87`.
- "Is the refund NOT approved?" gives `p_true = 0.14`.

The answer is driven by the question's own words and negation.

**Fix:** never score candidates against the question text. Use instructions only to pick the family, and to fall back when the state is empty. Choice/Score already do this; make `noul` consistent.

#### R8 (P1): Confidence, fallbacks and "calibration" mean different things per path

- **The same input gives three confidences (P15):** native `confidence` = 0.543 (normalized-entropy concentration), SystemOne = 0.808 (top-1 minus top-2 margin), Tev1 = 0.891 (top probability).
- **The `gate` CLI compares its threshold to *concentration*.** On "Payment confirmed. Payment confirmed by bank." the decision is `true` with top_prob 0.857, yet the gate reports `passed: false` (confidence 0.408 < 0.7) (P10). Users will read the threshold as a probability.
- **The CLM fallback** (`engine.rs:313-332`) replaces the choice but sets confidence to a fixed **0.85**. It leaves `probabilities` untouched, so `choice` can disagree with the argmax of the probabilities it's returned alongside.
- **The neural fallback** (`engine.rs:333-356`, `neural.rs:108-130`) returns fixed probabilities (0.95 and 0.05/(n−1)), `entropy_nats: 0.1`, `concentration: 0.9` and `temperature: 1.0`, in the same fields as real outputs. For Score/Numeric questions it returns a *string* id as `decision` where the native path returns a number, so `evaluate_speculative_hybrid` changes the response schema.
- **Family temperatures and margin dampening** (the new `calibration.rs:58-95`) apply only on the SystemOne path. The native path, which is the one that's documented and benchmarked, doesn't get them.

**Fix:** define one `Confidence` (a top probability, say) and expose the others under explicit names (`margin`, `concentration`). Mark fallback answers with `source: "clm" | "neural"` and leave calibrated fields `null` when they weren't computed.

#### R9 (P1): Preprocessing destroys or pollutes input
`preprocessor.rs:6-62`

- **`clean_text` truncates at the first line starting with `---` or `___`** (P01):
  - A state that *begins* with `---` (YAML front matter, a markdown rule) becomes **`""`**.
  - `"Login fails.\n---\nUpdate: database outage confirmed"` becomes `"Login fails."`, dropping the key evidence.
- **Temporal facts are appended to the premise** that is then scored. `"[Temporal Facts: reference_date=…, yesterday=…, 7_days_ago=…]"` adds the tokens "temporal", "facts", "reference", "yesterday", "days" and "ago". Any option mentioning them gets boosted: P02 shows the logits shifting by about 2 in both directions.
  - Because the premise includes `Utc::now()`, results are **not reproducible across days**, which is a problem for tests, audits and caching.
  - The dates are UTC, not the caller's timezone.
  - The trigger is loose: "hours" and "today" anywhere turn it on.
- **SystemOne and Tev1 always inject temporal facts.** `enable_temporal_facts` can't be turned off there.

**Fix:** only strip signature blocks that match known footer patterns in the *trailing* part of the text. Pass temporal facts as structured context that is never tokenized into the premise, and take `now` as a parameter (inject a clock).

#### R10 (P1): Advertised limits aren't enforced; there's no protection against large requests
`server.rs`, `engine.rs:119`

- `MAX_QUESTIONS = 64` is published by `/v1/limits` but never checked. P08 accepted 1,000 questions.
- `/v1/tev1`, SystemOne choice criteria and state size are all unbounded (axum's default 2 MB body limit is the only cap).
- Scoring cost is about O(questions × candidate words × premise length). P11 measured 25 options at 0.6 ms for 1 KB, 27 ms for 100 KB and **250 ms for 1 MB**. A single 2 MB request with 64 questions ties up a worker for tens of seconds.
- Handlers do this CPU work directly on the async runtime (no `spawn_blocking`). With the `neural` feature they make **blocking** model calls inside the async handler. There's no timeout or concurrency limit.
- The HF Space binds `0.0.0.0` publicly with none of these limits.

**Fix:**

- Validate limits at the top of `evaluate`, `evaluate_system_one` and `evaluate_tev1`: questions, options, state bytes.
- Add `tower_http::limit::RequestBodyLimitLayer`, `TimeoutLayer` and `ConcurrencyLimitLayer`.
- Wrap evaluation in `tokio::task::spawn_blocking`.

#### R11 (P1): A public function can hang forever
`order_invariant.rs:64-89`

`PremiseContext::contains_bounded("")` (and `is_negated("")`) loops forever: `find("")` returns the same position, and `start` advances by `pattern.len() == 0`. P09 confirmed it never returned. Internal callers happen to avoid empty patterns, but both functions are `pub` and re-exported.

**Fix:** `if pattern.is_empty() { return false; }`, and advance by at least one character.

#### R12 (P2): Other error-handling gaps

- **Validation happens after work.** `ZevEngine::evaluate` runs `validate` inside the per-question loop (`engine.rs:145`), so a bad 40th question fails the request after 39 were computed. Validate everything first.
- **Choice questions are silently shortlisted.** When options exceed the slot limit they're trimmed before `validate` (`engine.rs:120-145`), so a request that exceeds `max_slots` is never rejected and the caller isn't told options were dropped. At minimum, report `shortlisted: true`.
- **Invalid temperatures are swallowed.** `ZevEngine::new` falls back to a duplicated literal (`engine.rs:89`), so `zev decide -t -1` silently runs at 2.179. Return `Result`.
- **Every error becomes HTTP 400**, including internal ones (`Io`, non-finite softmax), and the body is plain text rather than JSON (`server.rs:91, 102, 113`). Map `ZevError` to 400/422/500 with a JSON body.
- **`zev decide` hides parse errors.** It tries `ZevRequest` and then `SystemOneRequest`, and on failure prints a generic message (`bin/zev.rs:114-123`). Show the native parse error, or add `--format`.
- **`zev tev1 --options` splits on commas**, which breaks options that contain commas (`bin/zev.rs:148`).
- **The VectorArena has panic paths:**
  - `VectorArena::new(0, d)` followed by `insert` panics (`clm.rs:117`).
  - `dot_product` and `score_keys` index `query_norm[i]` without checking its length.
- **`parse_candidate_choice` maps numbered answers wrongly** (`neural.rs:187-196`). It maps "2." against the *full* candidate list, but the prompt numbered the *shortlisted* list, so it picks the wrong option whenever there are more than 6 candidates. Its keyword fallback (`206-210`) picks the first id that appears as a substring; a short id like `a` matches almost anything, and the result depends on option order.

### 2.2 Performance

The small-request path is fast: **about 8 µs** mean on x86 for a 3-option request (P12), so latency isn't the problem. The issues are scaling, wasted work, and optimizations that don't achieve what they claim.

#### F1 (P1): Scoring rescans the whole premise for every word
`order_invariant.rs:178-213`

For each description word the scorer does:

- a lowercase allocation,
- `contains_bounded` (linear scan),
- `recency_weight` (`rfind`, another scan, and not word-bounded),
- `is_negated` (another scan, plus `trim_end` and `ends_with` per hit),
- and, when unmatched, a stem allocation and another scan.

That is roughly 3–4 full passes over the premise per word per candidate, which explains the linear growth in P11 (250 ms at 1 MB). The doc comment "pre-tokenized premise" (`engine.rs:112`, `order_invariant.rs:224`) is inaccurate: `PremiseContext` stores only a lowercase string.

**Fix:** tokenize once in `PremiseContext::new` into `HashMap<&str, SmallVec<[u32; 4]>>` (token → positions) plus a precomputed negation mask per position. Lookups become O(1) and recency and negation become array reads. Expect roughly 10–100× on large states, and no per-word allocation.

#### F2 (P2): Stack buffers don't avoid allocation
`decoding.rs:122`, `engine.rs:184-185`

`[0.0; 128]` / `[0.0; 64]` buffers avoid one small `Vec`. Right after that, the code:

- clones every candidate id twice into two `BTreeMap<String, f64>`s,
- allocates `sorted_probs` (`decoding.rs:184`) only to find the top two values (`dampen_temperature_by_margin` already shows the one-pass way),
- builds `valid_cands` and `cond_probs` Vecs, and formats strings.

P13 counted **79 heap allocations** for a single 3-option request. The buffers add code (duplicated `n <= 64` / `else` branches in `engine.rs:263-303` and `377-425`) without any measurable gain.

**Fix:** drop the stack buffers and the duplicated branches, and write into a `Vec` reused per request. If allocations matter, measure with a counting allocator in a bench (the one in P13 works) and target the `BTreeMap<String, _>` outputs; `Arc<str>` or index-keyed output would help.

#### F3 (P1): The CLM fallback allocates 4 MB per question
`engine.rs:314`, `clm.rs:319-327`

With `ZEV_FALLBACK=clm`, every low-confidence choice question builds `HybridVerifier::default()`, which zero-fills a **2048 × 512 × f32 = 4 MB** arena, fills it, and then never reads from it (scoring uses `head.score(&state_emb, &action_emb)` directly). The code also embeds the *un-preprocessed* state, so its view of the input differs from the lexical scorer's.

**Fix:** delete the arena use here, or hold one verifier on `ZevEngine` behind a `Mutex`/`RwLock`.

#### F4 (P2): The environment is read on the hot path
`engine.rs:310`

`std::env::var("ZEV_FALLBACK")` runs per choice question per request. It is a libc call under a global lock, and a hidden global configuration input for a library.

**Fix:** read it once into `ZevEngine` config (the CLI can set it from the env or a flag).

#### F5 (P2): Smaller performance items

- **LRU is O(n) on every hit.** `VectorArena::get` scans the `VecDeque` (`clm.rs:93-96`). Use a generation counter or an intrusive list.
- **`evaluate_hybrid` rebuilds the premise context per candidate.** It calls `PremiseContext::new(state)`, lowercasing the whole state, on every candidate cache miss (`clm.rs:385`). Build it once.
- **The engine evaluates backends one at a time.** `ApfelNeuralBackend::new()` is created per question (`engine.rs:347`), and neural calls in `evaluate_speculative_hybrid` run sequentially.
- **Eager `to_string()` calls.** `lvl.as_str().unwrap_or(&lvl.to_string())` (`engine.rs:388, 413`) allocates even when the value is a string; use `unwrap_or_else` or `map_or_else`.
- **`clean_text` lowercases each line twice** (`preprocessor.rs:53-54`).
- **`tokio = { features = ["full"] }` is a non-optional dependency** even though the library never uses tokio. That inflates build time and binary size for library users.

### 2.3 Code organization

#### O1 (P1): The SystemOne path is a second implementation of the engine
`engine.rs:172-448`

`evaluate_system_one` re-implements candidate generation, softmax, argmax, confidence and statistics (score expected value) inline, for each wire type, twice (stack-buffer and heap branches). That duplication is why R8's inconsistencies exist.

**Fix:** convert `WireQuestion` into `Question` (a `From`/`TryFrom` in a `wire.rs` module), call the same `evaluate` pipeline, and convert `ZevAnswer` back to the wire JSON. Do the same for Tev1. One decoder, one set of temperature rules, one definition of confidence, and each adapter is about 50 lines.

#### O2 (P2): There are two engine types and a trait for one job
`engine.rs:15-75`

`DecisionEngine` wraps `ZevEngine`, implements `Deref` to it (an anti-pattern for non-smart-pointers), and re-declares `evaluate`/`evaluate_system_one`. The `Evaluable` trait adds a third calling convention (`eng.eval(&req)`). Library users now see four ways to do the same thing.

**Fix:** keep one `Engine` with a builder (`Engine::builder().temperature(t).fallback(Fallback::Clm).build()?`). If you want a generic entry point, keep `Evaluable` and drop `DecisionEngine`.

#### O3 (P2): Module boundaries and feature gating

- **`clm.rs`** (VectorArena, ContrastiveHead, embed_text) is compiled into every build and re-exported at the crate root, but it's only used by an env-var fallback. `ContrastiveHead` isn't a learned projection (it applies GELU and normalizes); `input_dim` and `hidden_dim` are unused, and the "Matches CLM-8B projection architecture" doc is misleading. Put it behind a `clm` feature and label it experimental.
- **`neural.rs`** puts `#[cfg(feature = "neural")]` on every item, although `lib.rs` already gates the module. Remove the redundant attributes.
- **The binary needs `server` but doesn't declare it.** `src/bin/zev.rs` uses `create_router`/`axum` unconditionally, so `cargo check --no-default-features` fails with `E0432 unresolved import zev::create_router` (verified). Add `required-features = ["server"]` to `[[bin]]`, or cfg-gate the `Serve` subcommand.
- **`lib.rs` re-exports everything** (`pub use types::*`) and makes every module `pub` as well, so each item has two public paths. Pick one public surface: modules private, curated re-exports.

#### O4 (P2): Configuration is scattered as magic values

- The calibrated temperature `2.179078721266035` appears 3 times: `types.rs:9`, `engine.rs:89` and `server.rs:78`. `/v1/limits` reports the constant, not the engine's configured temperature.
- Scorer weights (0.5, 3.5, 2.5, 2.2, 1.8, 1.4, 0.6, 2.4, 1.5, 0.6 recency, the 45-byte window), family multipliers, the fallback threshold 0.35, the dampening threshold 0.40 and the neural shortlist size 6 are inline literals.
- Family selection is by English keywords in the instructions (`engine.rs:251-260`): "term" matches "determine", "action" matches "transaction".

**Fix:** a `ScorerConfig` / `EngineConfig` struct with documented defaults, serializable so a calibration run can emit it.

#### O5 (P2): Stringly-typed API
`types.rs:203-204`

`ZevAnswer.status: String` and `question_type: String`, plus the `family: &str` match, all mean typos compile. Use enums with `#[serde(rename_all = "snake_case")]`; the wire format stays the same.

### 2.4 Coding standards and tooling

| # | Sev | Finding | Fix |
|---|---|---|---|
| S1 | P1 | **CI only runs coverage.** No `clippy -D warnings`, no `fmt --check`, no `--no-default-features` build (which is broken today, see O3), no macOS job for `neural`, no MSRV check, no cache. | Add a `ci.yml` with a matrix: `{ubuntu, macos} × {default, no-default, all-features on macOS}` plus fmt, clippy and `Swatinem/rust-cache`. |
| S2 | P1 | **`Cargo.lock` is gitignored** even though this crate ships a binary, an npm package and a Docker image. `hf-space/Dockerfile` does `COPY Cargo.lock`, so a clean checkout can't build the Space. The Dockerfile pins `rust:1.80`, but the locked `clap 4.6.7`, `indexmap 2.14` and `hashbrown 0.17` require **1.85**. | Commit `Cargo.lock`, set `rust-version = "1.85"` in `Cargo.toml`, and bump the Docker image. |
| S3 | P2 | **Not rustfmt'd:** `cargo fmt --check` shows 105 diff hunks. There's no `rustfmt.toml` or `clippy.toml`. | Run `cargo fmt` once in its own commit and enforce it in CI. |
| S4 | P2 | **The 51,264-line generated test file** `tests/scale_2000_tests.rs` contains 2,000 near-identical cases, which produce 200+ clippy warnings. It's cheap to compile (about 3 s) but noisy. It inflates the "2,000 tests" and coverage figures without adding distinct behaviours, and `scripts/generate_2000_tests.py` must be re-run to change anything. | Replace it with table-driven or `proptest` tests: a permutation-invariance property, monotonicity, and "no panic on arbitrary input". Those will find R5/R11-style bugs that fixed cases can't. |
| S5 | P2 | **The npm package works only on Apple Silicon.** It ships a single darwin-arm64 binary (`bin/zev-bin`, 2.6 MB) to every platform, and `zev.js` falls back to `zev` on `PATH` elsewhere. `npm test` only runs `--help`. | Use per-platform optional dependencies (the esbuild/biome pattern), or a postinstall download from GitHub Releases. |
| S6 | P2 | **The working tree has 711 lines of uncommitted changes plus an untracked test file**, including the new calibration protocols and fallbacks. | Commit them in reviewable slices; several findings above are in this new code. |
| S7 | P3 | **Temperature errors carry a misleading message and are validated twice.** `scaled_softmax_slice` reports a length mismatch as "Logits array cannot be empty" (`calibration.rs:29-31`), and `scaled_softmax` validates everything twice. | Fix the message and validate once. |
| S8 | P3 | **`fit_temperature` breaks on empty input.** It returns NaN-driven garbage for `pairs = []` (0/0) and silently skips softmax errors. | Return `Result` and require a non-empty input. |
| S9 | P3 | **`unwrap()` where the invariant is local.** `engine.rs:137, 462, 488` and `order_invariant.rs:73, 80` rely on unwrap. | Use `expect("…")` with the invariant, or restructure (for example the `shortlisted_storage` dance can be a `Cow<Question>`). |
| S10 | P3 | **Public items are mostly undocumented.** No crate-level docs, and pedantic clippy flags missing `# Errors`/`# Panics` sections on public functions. | Add `#![warn(missing_docs)]` once the API is trimmed (O2/O3). |

### 2.5 Documentation accuracy

- **"Zero heap allocations" / "0 MB RAM"** (README lines 13, 89, 162): measured 79 allocations per small request (P13).
- **"SIMD"** is used throughout, but there are no SIMD intrinsics, `std::simd` or explicit vectorization; it's scalar string search.
- **The "Candle" backend and its benchmark column** (README 129-160, 276-302) describe code that isn't in this crate.
- **"100% order-invariant" / "0.0% permutation flip rate"** (`/v1/limits`, README) holds for the native path, where ties are broken by id. It doesn't hold for Tev1, whose labels are assigned by position, or for the neural keyword fallback.
- **The `/v1/models` release dates are hard-coded to today.**

Claims like these tend to get checked by the first person who benchmarks the crate. It's safer to state measured numbers with hardware and input size, and to move roadmap items (Candle) into a "Planned" section.

---

## 3. Recommended order of work

**Week 1: correctness (P0)**

1. Word-token negation and polarity (R1, R2), plus a regression table. Remove the benchmark phrase hacks (R3).
2. Numeric out-of-range status independent of `allow_abstain` (R4).
3. Tev1 label bounds and duplicate check; parser anchoring (R5, R6).
4. Empty-pattern guard (R11). Enforce `MAX_QUESTIONS`, option counts and state size in all three entry points (R10).

**Week 2: one engine**

5. Route SystemOne and Tev1 through the native pipeline via `wire.rs` adapters (O1). This fixes R7 and most of R8 structurally.
6. A single confidence definition; mark fallback answers and stop fabricating calibrated fields (R8).
7. Preprocessing: tail-only signature stripping, structured temporal context, injected clock (R9).
8. `EngineConfig`, including fallback mode, replacing env vars and magic numbers (O4, F4); one engine type (O2).

**Week 3: performance and hygiene**

9. Tokenize the premise once (F1). Add a `criterion` bench (small, 100 KB and 1 MB states) to CI so regressions show up.
10. Remove the stack-buffer duplication (F2); fix the CLM allocation (F3).
11. CI matrix, committed lockfile, MSRV, fmt pass, feature-gate `clm`, `required-features` for the bin (S1–S3, O3).
12. `spawn_blocking` plus tower limits/timeouts in the server; JSON errors with correct status codes (R10, R12).
13. Replace the 2,000 generated tests with property tests (S4). Correct the README claims (2.5).

---

## Appendix A: Probe results (raw)

```
P01 leading ---: ""
P01 mid ---: "Login fails."
P02 temporal ON  logits={"followup": 5.03, "general": 12.90}
P02 temporal OFF logits={"followup": 3.19, "general": 15.14}
P03 'unresolved' negates 'database'? true
P03 'casino' negates 'deposit'? true
P03 'prefixed' negates 'invoice'? true
P03 engine decision: billing, probs={billing: 0.972, database_outage: 0.028}
P04 'Normal priority' = -1.0 | 'Standard priority' = 0.5
P04 "Yesterday's orders" = 2.9 | 'Recent orders' = 0.5
P05 options: ["A: Yes, it is", "fine.", "B: No"]
P05 'Answer' in state: Err(InvalidRequest("Missing 'Question' in Tev1 prompt"))
P05 'Statement' line: "Order arrived broken.ment: customer attached photos."
P06 30 options -> labels include "[", "\\", "]", "^"
P06 200 options -> panic at src/tev1.rs:161 (debug build)
P06 duplicate letters -> probs {A, C} (3 options in, 2 out)
P07 status=ok decision=72.32 probs={__above_range__: 0.876, __below_range__: 0.093, "0": 0.009, "1": 0.022}
P08 1000 questions accepted (MAX_QUESTIONS=64)
P09 contains_bounded("") returned within 2s? false (hang)
P10 gate passed=false decision=true concentration=0.408 top_prob=0.857
P11 25 options: 1KB 0.61ms | 10KB 2.56ms | 100KB 27ms | 1MB 250ms (release, x86)
P12 small 3-option request: 7.8 µs mean (release, x86)
P13 heap allocations for one 3-option request: 79
P14 unrelated state: "Is the refund approved?" p_true=0.87 | "...NOT approved?" p_true=0.14
P15 same input: native conf=0.543 | systemone conf=0.808 | tev1 conf=0.891
```

## Appendix B: Tool output summary

| Check | Result |
|---|---|
| `cargo build --all-targets` | ✅ |
| `cargo test` | ✅ 2,060 passed (lib 20, integration 18, cli 7, scale 2,000, doc 0); `benchmark_regressions.rs` not run, see methodology |
| `cargo clippy --all-targets` | 7 warnings in `src/` and non-generated tests (explicit counter loop, needless borrows, collapsible if, identical if-blocks); 200 in the generated scale file |
| `cargo clippy -W clippy::pedantic` | about 2,600 warnings, mostly in generated tests (casts, similar names, `uninlined_format_args`) |
| `cargo fmt --check` | ❌ 105 diff hunks |
| `cargo check --no-default-features` | ❌ `E0432 unresolved import zev::create_router` in the binary |
| MSRV (locked deps) | 1.85 (Dockerfile uses 1.80) |
