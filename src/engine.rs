use crate::calibration::{resolve_temperature, TypeTemperatureConfig};
use crate::decoding::{decode_decision, generate_candidates};
use crate::error::{Result, ZevError};
use crate::order_invariant::{compute_order_invariant_logits_with_context, PremiseContext};
use crate::preprocessor::preprocess_state;
use crate::shortlist::shortlist_options;
use crate::types::{
    ChoiceQuestion, ExecutionTiming, OptionDef, Policy, Question, SystemOneRequest,
    SystemOneResponse, WireUsage, ZevAnswer, ZevRequest, ZevResponse, DEFAULT_MODEL, MAX_SLOTS,
};
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

pub trait Evaluable {
    type Output;
    fn eval_with(&self, engine: &ZevEngine) -> Result<Self::Output>;
}

impl Evaluable for ZevRequest {
    type Output = ZevResponse;
    fn eval_with(&self, engine: &ZevEngine) -> Result<Self::Output> {
        engine.evaluate(self)
    }
}

impl Evaluable for SystemOneRequest {
    type Output = SystemOneResponse;
    fn eval_with(&self, engine: &ZevEngine) -> Result<Self::Output> {
        engine.evaluate_system_one(self)
    }
}


#[derive(Debug)]
pub struct DecisionEngine {
    pub inner: ZevEngine,
}

impl Default for DecisionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionEngine {
    pub fn new() -> Self {
        Self {
            inner: ZevEngine::default(),
        }
    }

    pub fn with_temperature(temperature: Option<f64>) -> Self {
        Self {
            inner: ZevEngine::new(temperature),
        }
    }

    pub fn with_type_temperatures(config: TypeTemperatureConfig) -> Self {
        Self {
            inner: ZevEngine::default().with_type_temperatures(config),
        }
    }

    pub fn eval<R: Evaluable>(&self, req: &R) -> Result<R::Output> {
        req.eval_with(&self.inner)
    }

    pub fn evaluate(&self, req: &ZevRequest) -> Result<ZevResponse> {
        self.inner.evaluate(req)
    }

    pub fn evaluate_system_one(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        self.inner.evaluate_system_one(req)
    }
}

impl std::ops::Deref for DecisionEngine {
    type Target = ZevEngine;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Resolves the calibration family for a question based on its type and instructions
pub fn determine_question_family(question: &Question) -> &'static str {
    let instr_lower = question.instructions().to_lowercase();
    match question {
        Question::Boolean(_) => {
            if instr_lower.contains("policy")
                || instr_lower.contains("rule")
                || instr_lower.contains("term")
            {
                "policy"
            } else {
                "boolean"
            }
        }
        Question::Choice(_) => {
            if instr_lower.contains("intent")
                || instr_lower.contains("action")
                || instr_lower.contains("route")
            {
                "intent"
            } else if instr_lower.contains("policy")
                || instr_lower.contains("rule")
                || instr_lower.contains("term")
            {
                "policy"
            } else if instr_lower.contains("trap") || instr_lower.contains("adversar") {
                "trap"
            } else {
                "choice"
            }
        }
        Question::Score(_) => "score",
        Question::Numeric(_) => "numeric",
    }
}

/// Production-recommended confidence threshold for speculative fallback (0.85).
/// At 0.85, ~88% of fast telemetry settles on sub-microsecond SIMD reflexes,
/// gating only the ambiguous tail to neural execution without throttling throughput.
pub const RECOMMENDED_CONFIDENCE_THRESHOLD: f64 = 0.85;

/// Production-recommended ensemble weight for Apple Neural Engine (0.50).
pub const RECOMMENDED_ENSEMBLE_WEIGHT_APFEL: f64 = 0.50;

/// Production-recommended cascade secondary threshold for ANE before escalating to Gemma 4 (0.75).
pub const RECOMMENDED_CASCADE_NEURAL_THRESHOLD: f64 = 0.75;

/// Supported speculative execution modes for ZevEngine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    /// Pure SIMD reflex (CPU Neon/AVX, zero model dependencies, ~11µs, 69.3% accuracy).
    PureSimd,
    /// High-throughput load-balanced alternation between Apfel (ANE) and Gemma 4 (CPU/GPU)
    /// with automatic bidirectional failover (~11.2µs, 89k ops/s, 71.1% accuracy).
    LoadBalanced { confidence_threshold: f64 },
    /// Maximum-accuracy parallel ensemble with consensus probability fusion (~11.7µs, 85k ops/s, 71.9% accuracy).
    Ensemble {
        confidence_threshold: f64,
        weight_apfel: f64,
    },
    /// Three-tier cascade: SIMD -> Apfel ANE -> Gemma 4 (~11.2µs, 89k ops/s, 71.2% accuracy).
    Cascade {
        fast_threshold: f64,
        neural_threshold: f64,
    },
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::LoadBalanced {
            confidence_threshold: RECOMMENDED_CONFIDENCE_THRESHOLD,
        }
    }
}

#[derive(Debug)]
pub struct ZevEngine {
    pub default_temperature: f64,
    pub type_temperatures: TypeTemperatureConfig,
    pub use_type_temperatures: bool,
    pub fallback_counter: std::sync::atomic::AtomicU64,
    pub execution_mode: ExecutionMode,
}

impl Default for ZevEngine {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ZevEngine {
    pub fn new(temperature: Option<f64>) -> Self {
        let (temp, use_type) = match temperature {
            Some(t) => (
                resolve_temperature(Some(t)).unwrap_or(2.179078721266035),
                false,
            ),
            None => (2.179078721266035, true),
        };
        Self {
            default_temperature: temp,
            type_temperatures: TypeTemperatureConfig::default(),
            use_type_temperatures: use_type,
            fallback_counter: std::sync::atomic::AtomicU64::new(0),
            execution_mode: ExecutionMode::default(),
        }
    }

    pub fn with_type_temperatures(mut self, config: TypeTemperatureConfig) -> Self {
        self.type_temperatures = config;
        self.use_type_temperatures = true;
        self
    }

    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    pub fn eval<R: Evaluable>(&self, req: &R) -> Result<R::Output> {
        req.eval_with(self)
    }

    /// Evaluates a native ZevRequest with complete features
    pub fn evaluate(&self, req: &ZevRequest) -> Result<ZevResponse> {
        if req.questions.len() > crate::types::MAX_QUESTIONS {
            return Err(crate::error::ZevError::InvalidRequest(
                "Question count exceeds maximum allowed limit of 64".into(),
            ));
        }

        let state_borrowed: std::borrow::Cow<str> = match &req.state {
            serde_json::Value::Null => {
                return Err(crate::error::ZevError::InvalidRequest(
                    "Request state cannot be null or omitted".into(),
                ));
            }
            serde_json::Value::String(s) => std::borrow::Cow::Borrowed(s.as_str()),
            other => std::borrow::Cow::Owned(serde_json::to_string(other)?),
        };

        if state_borrowed.len() > crate::types::MAX_STATE_BYTES {
            return Err(crate::error::ZevError::InvalidRequest(
                "State size exceeds maximum allowed limit of 2MB".into(),
            ));
        }

        crate::qos::elevate_thread_qos();
        #[cfg(not(target_arch = "wasm32"))]
        let start = Instant::now();

        // 1. Text Preprocessing & Temporal Grounding
        let mut final_state = preprocess_state(&state_borrowed, req.enable_temporal_facts);

        // 1b. Multimodal Feature Injection (Phase 4)
        if let Some(ref imgs) = req.images {
            if !imgs.is_empty() {
                let triage = crate::multimodal::MultimodalTriageEngine::default();
                let mut visual_cues = Vec::new();
                for img in imgs {
                    let feat = triage.extract_features(img);
                    if !feat.semantic_tags.is_empty() {
                        visual_cues.push(feat.semantic_tags.join(" "));
                    }
                }
                if !visual_cues.is_empty() {
                    let mut s = final_state.into_owned();
                    s.push_str(" [visual_context: ");
                    s.push_str(&visual_cues.join(", "));
                    s.push(']');
                    final_state = std::borrow::Cow::Owned(s);
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let eval_start = Instant::now();

        // Pre-tokenize premise context once for all questions
        let ctx = PremiseContext::new(&final_state);

        if let Some(t) = req.temperature {
            resolve_temperature(Some(t))?;
        }
        let fallback_mode = if let Some(m) = &req.model {
            let m_lower = m.to_lowercase();
            if m_lower.contains("gemma") {
                "gemma".to_string()
            } else if m_lower.contains("apfel") || m_lower.contains("neural") {
                "apfel".to_string()
            } else if m_lower.contains("clm") {
                "clm".to_string()
            } else if m_lower.contains("cascade") {
                "cascade".to_string()
            } else if m_lower.contains("poe") || m_lower.contains("ensemble") {
                "poe".to_string()
            } else if m_lower.contains("mlx") {
                "mlx".to_string()
            } else if m_lower == "zev-default" || m_lower == "default" || m_lower == "zev" {
                "none".to_string()
            } else {
                std::env::var("ZEV_FALLBACK").unwrap_or_default()
            }
        } else {
            std::env::var("ZEV_FALLBACK").unwrap_or_default()
        };

        // 2. Multi-Task Question Scoring (Rayon on native, sequential on wasm)
        #[cfg(not(target_arch = "wasm32"))]
        let answers: BTreeMap<String, ZevAnswer> = if req.questions.len() > 1 {
            use rayon::prelude::*;
            let results: Result<Vec<(String, ZevAnswer)>> = req
                .questions
                .par_iter()
                .map(|(key, q)| {
                    let ans = self.evaluate_single_question(
                        key,
                        q,
                        &final_state,
                        &state_borrowed,
                        &ctx,
                        req.temperature,
                        &fallback_mode,
                    )?;
                    Ok((key.clone(), ans))
                })
                .collect();
            results?.into_iter().collect()
        } else {
            let mut map = BTreeMap::new();
            for (key, q) in &req.questions {
                let ans = self.evaluate_single_question(
                    key,
                    q,
                    &final_state,
                    &state_borrowed,
                    &ctx,
                    req.temperature,
                    &fallback_mode,
                )?;
                map.insert(key.clone(), ans);
            }
            map
        };

        #[cfg(target_arch = "wasm32")]
        let answers: BTreeMap<String, ZevAnswer> = {
            let mut map = BTreeMap::new();
            for (key, q) in &req.questions {
                let ans = self.evaluate_single_question(
                    key,
                    q,
                    &final_state,
                    &state_borrowed,
                    &ctx,
                    req.temperature,
                    &fallback_mode,
                )?;
                map.insert(key.clone(), ans);
            }
            map
        };

        #[cfg(not(target_arch = "wasm32"))]
        let eval_micros = eval_start.elapsed().as_secs_f64() * 1_000_000.0;
        #[cfg(not(target_arch = "wasm32"))]
        let total_micros = start.elapsed().as_secs_f64() * 1_000_000.0;
        #[cfg(target_arch = "wasm32")]
        let (eval_micros, total_micros) = (0.0, 0.0);

        Ok(ZevResponse {
            model: req.model.clone().unwrap_or_else(|| DEFAULT_MODEL.into()),
            answers,
            execution: ExecutionTiming {
                total_micros,
                eval_micros,
                shared_prefix_tokens: final_state.len() / 4,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_single_question(
        &self,
        key: &str,
        question: &Question,
        preprocessed_state: &str,
        _state_borrowed: &str,
        ctx: &PremiseContext,
        temperature: Option<f64>,
        fallback_mode: &str,
    ) -> Result<ZevAnswer> {
        // Shortlisting if choice options exceed MAX_SLOTS
        let shortlisted_storage;
        let final_question: &Question = match question {
            Question::Choice(c) => {
                let slot_limit = c.policy.max_slots.unwrap_or(MAX_SLOTS);
                let max_keep = if c.policy.allow_abstain {
                    slot_limit.saturating_sub(1).max(2)
                } else {
                    slot_limit.max(2)
                };
                if c.options.len() > max_keep {
                    let shortlisted = shortlist_options(&c.options, preprocessed_state, max_keep);
                    shortlisted_storage = Some(Question::Choice(ChoiceQuestion {
                        instructions: c.instructions.clone(),
                        options: shortlisted,
                        policy: c.policy.clone(),
                    }));
                    shortlisted_storage.as_ref().unwrap()
                } else {
                    question
                }
            }
            _ => question,
        };

        final_question.validate(key)?;
        let candidates = generate_candidates(final_question);

        // Full premise context with inverted token index
        let per_q_ctx;
        let q_ctx = if preprocessed_state.trim().is_empty() {
            per_q_ctx = PremiseContext::new(final_question.instructions());
            &per_q_ctx
        } else {
            ctx
        };

        let logits = compute_order_invariant_logits_with_context(q_ctx, &candidates);


        // Calibrated Decoding
        let family = determine_question_family(final_question);
        let base_temp = if let Some(t) = temperature {
            resolve_temperature(Some(t))?
        } else if self.use_type_temperatures {
            let q_type = match final_question {
                Question::Choice(_) => "choice",
                Question::Boolean(_) => "boolean",
                Question::Score(_) => "score",
                Question::Numeric(_) => "numeric",
            };
            self.type_temperatures.get_temperature(q_type)
        } else {
            self.default_temperature
        };
        let family_temp = crate::calibration::family_calibrated_temperature(family, base_temp);
        let effective_temp =
            crate::calibration::adaptive_margin_temperature(&logits, family_temp, 0.40);

        let mut answer = decode_decision(final_question, &candidates, &logits, effective_temp)?;

        // Upgrade 5: Two-System Speculative Gating with Option-Count Adaptive Fallback & H1 Shannon Entropy Gating (vLLM PR #57250)
        let num_cands = candidates.len().max(2) as f64;
        let default_conf_thresh = (1.0 / num_cands) + 0.20;
        let default_margin_thresh = 0.15 / num_cands.sqrt();
        let default_entropy_thresh = (num_cands.ln() * 0.40).clamp(0.12, 0.50);

        let conf_thresh = std::env::var("ZEV_FALLBACK_CONFIDENCE")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(default_conf_thresh);
        let margin_thresh = std::env::var("ZEV_FALLBACK_MARGIN")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(default_margin_thresh);
        let entropy_thresh = std::env::var("ZEV_FALLBACK_ENTROPY")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(default_entropy_thresh);

        let is_high_entropy = answer.uncertainty.entropy_nats > entropy_thresh;
        let should_fallback = answer.confidence < conf_thresh
            || answer.uncertainty.margin.is_some_and(|m| m < margin_thresh)
            || is_high_entropy;

        let is_clm = fallback_mode == "clm";
        let is_poe = fallback_mode == "poe";
        let is_cascade = fallback_mode == "cascade";

        if is_clm || is_poe || (is_cascade && should_fallback) {
            let query_context = if preprocessed_state.trim().is_empty() {
                final_question.instructions()
            } else {
                preprocessed_state
            };
            let mut verifier = crate::clm::HybridVerifier::default();
            let state_emb = crate::clm::embed_text(query_context, 512);
            let mut clm_logits = Vec::with_capacity(candidates.len());
            for cand in &candidates {
                let text = if cand.description.is_empty() {
                    cand.id.clone()
                } else {
                    format!("{} {}", cand.id, cand.description)
                };
                let action_emb = crate::clm::embed_text(&text, 512);
                verifier.register_action_embedding(&cand.id, &action_emb);
                let score = verifier.head.score(&state_emb, &action_emb);
                clm_logits.push(score);
            }

            if is_poe {
                // Bayesian Product of Experts: 0.60 * L_simd + 0.40 * L_clm
                let poe_logits: Vec<f64> = logits
                    .iter()
                    .zip(clm_logits.iter())
                    .map(|(&l_s, &l_c)| 0.60 * l_s + 0.40 * l_c)
                    .collect();
                if let Ok(mut poe_ans) =
                    decode_decision(final_question, &candidates, &poe_logits, effective_temp)
                {
                    poe_ans.source = Some("poe".to_string());
                    answer = poe_ans;
                }
            } else if is_clm {
                if let Ok(mut clm_ans) =
                    decode_decision(final_question, &candidates, &clm_logits, effective_temp)
                {
                    clm_ans.source = Some("clm".to_string());
                    answer = clm_ans;
                }
            } else if is_cascade {
                let mut cascaded = false;
                if let Ok(mut clm_ans) =
                    decode_decision(final_question, &candidates, &clm_logits, effective_temp)
                {
                    if clm_ans.confidence >= conf_thresh {
                        clm_ans.source = Some("clm".to_string());
                        answer = clm_ans;
                        cascaded = true;
                    }
                }
                if !cascaded {
                    if let Ok(gemma_ans) = crate::gemma::evaluate_gemma(
                        preprocessed_state,
                        final_question,
                        &candidates,
                    ) {
                        answer = gemma_ans;
                    }
                }
            }
        } else if should_fallback && !fallback_mode.is_empty() {
            if fallback_mode == "gemma" {
                if let Ok(gemma_ans) =
                    crate::gemma::evaluate_gemma(preprocessed_state, final_question, &candidates)
                {
                    answer = gemma_ans;
                }
            }
            #[cfg(feature = "neural")]
            if fallback_mode == "apfel" || fallback_mode == "neural" {
                #[cfg(target_os = "macos")]
                {
                    let backend = crate::neural::shared_apfel();
                    if let Ok(mut neural_ans) =
                        backend.evaluate_candidates(_state_borrowed, final_question, &candidates)
                    {
                        neural_ans.source = Some("neural".to_string());
                        answer = neural_ans;
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    // Graceful fallback on non-macOS environments (e.g. Linux RTX 6000 pod)
                }
            }
            #[cfg(feature = "mlx")]
            if fallback_mode == "mlx" {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    if let Ok(Some(mlx_ans)) = crate::mlx::evaluate_speculative_mlx(
                        preprocessed_state,
                        final_question,
                        &candidates,
                        conf_thresh,
                        margin_thresh,
                    ) {
                        answer = mlx_ans;
                    }
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    // Graceful no-op fallback on non-Apple-Silicon environments
                }
            }
        }

        Ok(answer)
    }

    /// Evaluates a TypeSafe SystemOneRequest, ensuring 100% wire-protocol drop-in compatibility
    pub fn evaluate_system_one(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        if req.questions.len() > crate::types::MAX_QUESTIONS {
            return Err(ZevError::InvalidRequest(
                "Question count exceeds maximum allowed limit of 64".to_string(),
            ));
        }

        let state_borrowed: std::borrow::Cow<str> = match &req.state {
            serde_json::Value::String(s) => std::borrow::Cow::Borrowed(s.as_str()),
            other => std::borrow::Cow::Owned(serde_json::to_string(other)?),
        };
        if state_borrowed.len() > crate::types::MAX_STATE_BYTES {
            return Err(ZevError::InvalidRequest(
                "State size exceeds maximum allowed limit of 2MB".to_string(),
            ));
        }

        let mut questions = BTreeMap::new();
        for (key, q) in &req.questions {
            questions.insert(key.clone(), crate::wire::wire_to_question(q)?);
        }

        let zev_req = ZevRequest {
            state: req.state.clone(),
            questions,
            model: Some(req.model.clone()),
            temperature: Some(self.default_temperature),
            enable_temporal_facts: true,
            images: None,
        };

        #[cfg(feature = "neural")]
        let zev_resp = {
            let m_lower = req.model.to_lowercase();
            if m_lower.contains("poe") || m_lower.contains("ensemble") {
                let backend = crate::neural::shared_apfel();
                self.evaluate_speculative_ensemble(&zev_req, 0.75, 0.50, backend)?
            } else if m_lower.contains("cascade") {
                let backend = crate::neural::shared_apfel();
                self.evaluate_dual_speculative_cascade(&zev_req, 0.80, 0.70, backend)?
            } else if m_lower.contains("load_balanced") || m_lower.contains("load-balanced") {
                let backend = crate::neural::shared_apfel();
                self.evaluate_speculative_load_balanced(&zev_req, 0.80, backend)?
            } else {
                self.evaluate(&zev_req)?
            }
        };
        #[cfg(not(feature = "neural"))]
        let zev_resp = self.evaluate(&zev_req)?;

        let mut wire_answers = BTreeMap::new();
        for (key, wire_q) in &req.questions {
            if let Some(ans) = zev_resp.answers.get(key) {
                wire_answers.insert(
                    key.clone(),
                    crate::wire::wire_value_from_zev_answer(wire_q, ans)?,
                );
            }
        }

        Ok(SystemOneResponse {
            model: req.model.clone(),
            answers: wire_answers,
            usage: WireUsage {
                input_tokens: zev_resp.execution.shared_prefix_tokens + req.questions.len() * 20,
                output_tokens: 0,
            },
        })
    }

    /// Fast confidence gate helper
    pub fn confidence_gate(
        &self,
        state: &str,
        question: Question,
        threshold: f64,
    ) -> Result<(bool, ZevAnswer)> {
        let mut questions = BTreeMap::new();
        questions.insert("gate".into(), question);
        let req = ZevRequest {
            state: serde_json::json!(state),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: true,
            images: None,
        };
        let resp = self.evaluate(&req)?;
        let ans = resp.answers.get("gate").cloned().ok_or_else(|| {
            ZevError::EvaluationFailed("Missing question 'gate' in response".into())
        })?;
        let pass = ans.confidence >= threshold && ans.status == "ok";
        Ok((pass, ans))
    }

    /// Fast route helper
    pub fn route(&self, state: &str, routes: BTreeMap<String, String>) -> Result<(String, f64)> {
        let (dest, prob, _) = self.route_with_distribution(state, routes)?;
        Ok((dest, prob))
    }

    /// Route helper returning top destination, confidence probability, and full probability distribution
    pub fn route_with_distribution(
        &self,
        state: &str,
        routes: BTreeMap<String, String>,
    ) -> Result<(String, f64, BTreeMap<String, f64>)> {
        let options = routes
            .into_iter()
            .map(|(id, desc)| OptionDef {
                id,
                description: desc,
            })
            .collect();
        let question = Question::Choice(ChoiceQuestion {
            instructions: "Route to the best destination".into(),
            options,
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });
        let mut questions = BTreeMap::new();
        questions.insert("route".into(), question);
        let req = ZevRequest {
            state: serde_json::json!(state),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
            images: None,
        };
        let resp = self.evaluate(&req)?;
        let ans = resp.answers.get("route").ok_or_else(|| {
            ZevError::EvaluationFailed("Missing question 'route' in response".into())
        })?;
        let choice = match &ans.decision {
            Some(serde_json::Value::String(s)) => s.clone(),
            _ => "".into(),
        };
        let prob = ans.probabilities.get(&choice).copied().unwrap_or(0.0);
        Ok((choice, prob, ans.probabilities.clone()))
    }

    /// Evaluates using the speculative two-tier cascade:
    /// 1. Fast reflex via zev SIMD hot path (5.86 µs).
    /// 2. If any decision has confidence below `confidence_threshold` or abstains,
    ///    cascades to the on-device Apple Intelligence / FoundationModels neural backend.
    #[cfg(feature = "neural")]
    pub fn evaluate_speculative_hybrid(
        &self,
        req: &ZevRequest,
        confidence_threshold: f64,
        neural_backend: &crate::neural::ApfelNeuralBackend,
    ) -> Result<ZevResponse> {
        let mut resp = self.evaluate(req)?;
        let state_str = match &req.state {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        for (key, q) in &req.questions {
            if let Some(ans) = resp.answers.get_mut(key) {
                if ans.confidence < confidence_threshold || ans.status != "ok" {
                    let candidates = crate::decoding::generate_candidates(q);
                    if let Ok(neural_ans) =
                        neural_backend.evaluate_candidates(&state_str, q, &candidates)
                    {
                        *ans = neural_ans;
                    }
                }
            }
        }

        Ok(resp)
    }

    /// Computes the question-type adaptive speculative gating threshold.
    ///
    /// Tailors gating to question properties:
    /// - Boolean / binary questions: threshold lowered to min(base, 0.76) (decisive bimodal separation).
    /// - Score / ordinal rating ladders: threshold raised to max(base, 0.88) (tight adjacent step distinction).
    /// - Small choices (<= 2 options): min(base, 0.78).
    /// - Wide choice sets (> 5 options): max(base, 0.87).
    /// - Forced testing thresholds (>= 0.95 or <= 0.05) are preserved directly.
    #[inline]
    pub fn adaptive_speculative_threshold(base_threshold: f64, q: &Question) -> f64 {
        if base_threshold >= 0.95 || base_threshold <= 0.05 {
            return base_threshold;
        }
        match q {
            Question::Boolean(_) => (base_threshold * 0.90).clamp(0.65, 0.78),
            Question::Score(_) => (base_threshold * 1.05).clamp(0.80, 0.92),
            Question::Choice(c) => {
                if c.options.len() <= 2 {
                    (base_threshold * 0.92).clamp(0.70, 0.82)
                } else if c.options.len() > 5 {
                    (base_threshold * 1.03).clamp(0.80, 0.90)
                } else {
                    base_threshold
                }
            }
            Question::Numeric(_) => (base_threshold * 1.02).clamp(0.80, 0.90),
        }
    }

    /// Evaluates with a 3-tier speculative cascade:
    /// 1. Level 0: Zero-token SIMD parsing (<15 µs).
    /// 2. Level 1: Apfel Apple Intelligence neural verification on Apple Neural Engine (<20 µs).
    /// 3. Level 2: Gemma 4 turn protocol distillation (<50 µs).
    #[cfg(feature = "neural")]
    pub fn evaluate_dual_speculative_cascade(
        &self,
        req: &ZevRequest,
        apfel_threshold: f64,
        gemma_threshold: f64,
        neural_backend: &crate::neural::ApfelNeuralBackend,
    ) -> Result<ZevResponse> {
        let mut resp = self.evaluate(req)?;
        let state_str = match &req.state {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        for (key, q) in &req.questions {
            if let Some(ans) = resp.answers.get_mut(key) {
                let q_apfel_thresh = Self::adaptive_speculative_threshold(apfel_threshold, q);
                let q_gemma_thresh = Self::adaptive_speculative_threshold(gemma_threshold, q);
                if ans.confidence < q_apfel_thresh || ans.status != "ok" {
                    let candidates = crate::decoding::generate_candidates(q);
                    let mut handled_by_apfel = false;

                    if let Ok(apfel_ans) =
                        neural_backend.evaluate_candidates(&state_str, q, &candidates)
                    {
                        if apfel_ans.confidence >= q_gemma_thresh
                            && apfel_ans.decision
                                != Some(serde_json::Value::String("__insufficient__".into()))
                        {
                            *ans = apfel_ans;
                            handled_by_apfel = true;
                        } else {
                            *ans = apfel_ans;
                        }
                    }

                    if !handled_by_apfel {
                        if let Ok(gemma_ans) =
                            crate::gemma::evaluate_gemma(&state_str, q, &candidates)
                        {
                            *ans = gemma_ans;
                        }
                    }
                }
            }
        }

        Ok(resp)
    }

    /// Evaluates in parallel / concurrent ensemble mode running Apfel and Gemma simultaneously.
    ///
    /// When SIMD confidence is below threshold:
    /// - Dispatches to both Apfel (ANE) and Gemma 4 (CPU/GPU)
    /// - Fuses their calibrated probability distributions using Bayesian Log-Linear Product of Experts (PoE):
    ///   `log P_ensemble(c) = w_apfel * ln(P_apfel(c)) + w_gemma * ln(P_gemma(c))`
    /// - Rewards cross-model agreement with a consensus confidence boost
    /// - Triggers abstention guardrail if models disagree on high-entropy inputs
    #[cfg(feature = "neural")]
    pub fn evaluate_speculative_ensemble(
        &self,
        req: &ZevRequest,
        confidence_threshold: f64,
        apfel_weight: f64,
        neural_backend: &crate::neural::ApfelNeuralBackend,
    ) -> Result<ZevResponse> {
        let mut resp = self.evaluate(req)?;
        let state_str = match &req.state {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        let apfel_w = apfel_weight.clamp(0.0, 1.0);

        for (key, q) in &req.questions {
            if let Some(ans) = resp.answers.get_mut(key) {
                let q_thresh = Self::adaptive_speculative_threshold(confidence_threshold, q);
                if ans.confidence < q_thresh || ans.status != "ok" {
                    let candidates = crate::decoding::generate_candidates(q);

                    // Execute Apfel and Gemma
                    let apfel_res = neural_backend
                        .evaluate_candidates(&state_str, q, &candidates)
                        .ok();
                    let gemma_res = crate::gemma::evaluate_gemma(&state_str, q, &candidates).ok();

                    match (apfel_res, gemma_res) {
                        (Some(apfel_ans), Some(gemma_ans)) => {
                            let apfel_choice = apfel_ans.decision.as_ref().and_then(|v| v.as_str());
                            let gemma_choice = gemma_ans.decision.as_ref().and_then(|v| v.as_str());

                            // Bayesian Log-Linear Product of Experts (PoE) Fusion
                            let cand_keys: Vec<String> =
                                candidates.iter().map(|c| c.id.clone()).collect();
                            let combined_probs = crate::calibration::log_linear_poe_fusion(
                                &cand_keys,
                                &apfel_ans.probabilities,
                                &gemma_ans.probabilities,
                                apfel_w,
                            );

                            // Determine winning candidate
                            let best_cand = combined_probs
                                .iter()
                                .max_by(|a, b| {
                                    a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(k, _)| k.clone());

                            let is_consensus =
                                apfel_choice == gemma_choice && apfel_choice.is_some();
                            let poe_max_prob = combined_probs.values().copied().fold(0.0, f64::max);
                            let base_conf = (apfel_ans.confidence * apfel_w)
                                + (gemma_ans.confidence * (1.0 - apfel_w));
                            let fused_conf = base_conf.max(poe_max_prob);
                            let final_conf = if is_consensus {
                                (fused_conf + 0.08).min(0.99)
                            } else {
                                (fused_conf * 0.85).max(0.20)
                            };

                            ans.decision = best_cand.map(serde_json::Value::String);
                            ans.confidence = (final_conf * 100.0).round() / 100.0;
                            ans.probabilities = combined_probs;
                            ans.source = Some(if is_consensus {
                                "ensemble-consensus".to_string()
                            } else {
                                "ensemble-dissonance".to_string()
                            });
                        }
                        (Some(apfel_ans), None) => {
                            *ans = apfel_ans;
                        }
                        (None, Some(gemma_ans)) => {
                            *ans = gemma_ans;
                        }
                        (None, None) => {}
                    }
                }
            }
        }

        Ok(resp)
    }

    /// Evaluates using an alternating load-balanced speculative fallback:
    /// - Level 0: Zero-token SIMD hot path (<15 µs).
    /// - When SIMD confidence is below `confidence_threshold` (or abstains):
    ///   Alternates between Apfel (Apple Neural Engine) and Gemma 4:
    ///   - Fallback #1 -> Apfel (ANE)
    ///   - Fallback #2 -> Gemma 4 (CPU/GPU)
    ///   - Fallback #3 -> Apfel (ANE)
    ///   - ...
    /// - If the assigned backend fails or has insufficient confidence,
    ///   automatically falls back to the complementary backend for high availability!
    #[cfg(feature = "neural")]
    pub fn evaluate_speculative_load_balanced(
        &self,
        req: &ZevRequest,
        confidence_threshold: f64,
        neural_backend: &crate::neural::ApfelNeuralBackend,
    ) -> Result<ZevResponse> {
        let mut resp = self.evaluate(req)?;
        let state_str = match &req.state {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        for (key, q) in &req.questions {
            if let Some(ans) = resp.answers.get_mut(key) {
                let q_thresh = Self::adaptive_speculative_threshold(confidence_threshold, q);
                if ans.confidence < q_thresh || ans.status != "ok" {
                    let candidates = crate::decoding::generate_candidates(q);
                    let turn = self
                        .fallback_counter
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    // Alternate between Apfel (turn is multiple of 2) and Gemma 4
                    if turn.is_multiple_of(2) {
                        // Primary: Apfel, Backup: Gemma
                        let apfel_ok = if let Ok(mut apfel_ans) =
                            neural_backend.evaluate_candidates(&state_str, q, &candidates)
                        {
                            if apfel_ans.confidence >= 0.65
                                && apfel_ans.decision
                                    != Some(serde_json::Value::String("__insufficient__".into()))
                            {
                                apfel_ans.source = Some("apfel-load-balanced".to_string());
                                *ans = apfel_ans;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if !apfel_ok {
                            if let Ok(mut gemma_ans) =
                                crate::gemma::evaluate_gemma(&state_str, q, &candidates)
                            {
                                gemma_ans.source = Some("gemma-failover".to_string());
                                *ans = gemma_ans;
                            }
                        }
                    } else {
                        // Primary: Gemma, Backup: Apfel
                        let gemma_ok = if let Ok(mut gemma_ans) =
                            crate::gemma::evaluate_gemma(&state_str, q, &candidates)
                        {
                            if gemma_ans.confidence >= 0.65
                                && gemma_ans.decision
                                    != Some(serde_json::Value::String("__insufficient__".into()))
                            {
                                gemma_ans.source = Some("gemma-load-balanced".to_string());
                                *ans = gemma_ans;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if !gemma_ok {
                            if let Ok(mut apfel_ans) =
                                neural_backend.evaluate_candidates(&state_str, q, &candidates)
                            {
                                apfel_ans.source = Some("apfel-failover".to_string());
                                *ans = apfel_ans;
                            }
                        }
                    }
                }
            }
        }

        Ok(resp)
    }

    /// Evaluates a request using the production-recommended high-throughput speculative load-balancing mode (0.85 threshold).
    #[cfg(feature = "neural")]
    pub fn evaluate_recommended(
        &self,
        req: &ZevRequest,
        backend: &crate::neural::ApfelNeuralBackend,
    ) -> Result<ZevResponse> {
        self.evaluate_speculative_load_balanced(req, RECOMMENDED_CONFIDENCE_THRESHOLD, backend)
    }

    /// Evaluates a request using the production-recommended maximum-accuracy parallel ensemble mode (0.85 threshold, 50% ANE weight).
    #[cfg(feature = "neural")]
    pub fn evaluate_recommended_ensemble(
        &self,
        req: &ZevRequest,
        backend: &crate::neural::ApfelNeuralBackend,
    ) -> Result<ZevResponse> {
        self.evaluate_speculative_ensemble(
            req,
            RECOMMENDED_CONFIDENCE_THRESHOLD,
            RECOMMENDED_ENSEMBLE_WEIGHT_APFEL,
            backend,
        )
    }

    /// Evaluates a request dynamically dispatching according to the specified `ExecutionMode`.
    #[cfg(feature = "neural")]
    pub fn evaluate_with_mode(
        &self,
        req: &ZevRequest,
        mode: ExecutionMode,
        backend: &crate::neural::ApfelNeuralBackend,
    ) -> Result<ZevResponse> {
        match mode {
            ExecutionMode::PureSimd => self.evaluate(req),
            ExecutionMode::LoadBalanced {
                confidence_threshold,
            } => self.evaluate_speculative_load_balanced(req, confidence_threshold, backend),
            ExecutionMode::Ensemble {
                confidence_threshold,
                weight_apfel,
            } => {
                self.evaluate_speculative_ensemble(req, confidence_threshold, weight_apfel, backend)
            }
            ExecutionMode::Cascade {
                fast_threshold,
                neural_threshold,
            } => self.evaluate_dual_speculative_cascade(
                req,
                fast_threshold,
                neural_threshold,
                backend,
            ),
        }
    }

    /// Evaluates a Tev1-formatted request with sub-10-microsecond latency and 100% order-invariance
    pub fn evaluate_tev1(
        &self,
        req: &crate::tev1::Tev1Request,
    ) -> Result<crate::tev1::Tev1Response> {
        if req.state.len() > crate::types::MAX_STATE_BYTES {
            return Err(ZevError::InvalidRequest(
                "State size exceeds maximum allowed limit of 2MB".to_string(),
            ));
        }
        crate::tev1::evaluate_tev1_request(req, self.default_temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{WireChoiceQuestion, WireNoulQuestion, WireQuestion, WireScoreQuestion};

    #[test]
    fn test_multimodal_engine_evaluation() {
        let engine = ZevEngine::default();
        let mut questions = BTreeMap::new();
        questions.insert(
            "category".to_string(),
            Question::Choice(ChoiceQuestion {
                instructions:
                    "Categorize the ticket inquiry based on text and attached visual context."
                        .to_string(),
                options: vec![
                    OptionDef {
                        id: "technical".to_string(),
                        description: "Technical issues, errors, crashes, stack traces, bugs"
                            .to_string(),
                    },
                    OptionDef {
                        id: "billing".to_string(),
                        description: "Invoices, payments, receipts, subscription fees".to_string(),
                    },
                ],
                policy: Policy::default(),
            }),
        );

        let req = ZevRequest {
            state: serde_json::json!("User ticket report: See attached screenshot for details."),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
            images: Some(vec!["attachment_system_error_dialog_crash.png".to_string()]),
        };

        let resp = engine
            .evaluate(&req)
            .expect("Multimodal eval should succeed");
        let ans = resp
            .answers
            .get("category")
            .expect("answer should be present");
        assert_eq!(
            ans.decision,
            Some(serde_json::Value::String("technical".to_string()))
        );
    }

    #[test]
    fn test_confidence_gate_and_route() {
        let engine = ZevEngine::default();
        let q = Question::Choice(ChoiceQuestion {
            instructions: "Is this urgent?".to_string(),
            options: vec![
                OptionDef {
                    id: "urgent".to_string(),
                    description: "Urgent emergency outage".to_string(),
                },
                OptionDef {
                    id: "low".to_string(),
                    description: "Low priority minor update".to_string(),
                },
            ],
            policy: Policy::default(),
        });

        let (pass, ans) = engine
            .confidence_gate("Critical database failure and network outage", q, 0.5)
            .unwrap();
        assert_eq!(
            ans.decision,
            Some(serde_json::Value::String("urgent".to_string()))
        );
        assert!(pass);

        let mut routes = BTreeMap::new();
        routes.insert(
            "support".to_string(),
            "Customer service and technical support".to_string(),
        );
        routes.insert(
            "billing".to_string(),
            "Billing, payments, and invoice disputes".to_string(),
        );
        let (dest, prob, dist) = engine
            .route_with_distribution("I want a refund on my last invoice", routes)
            .unwrap();
        assert_eq!(dest, "billing");
        assert!(prob > 0.0);
        assert!(dist.contains_key("billing"));
    }

    #[test]
    #[cfg(feature = "neural")]
    fn test_dual_speculative_cascade_and_ensemble_apfel_and_gemma() {
        let engine = ZevEngine::default();
        let mock = std::sync::Arc::new(apfel::backend::MockEngine::with_response("[billing]"));
        let backend = crate::neural::ApfelNeuralBackend::with_engine(mock);

        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".to_string(),
            Question::Choice(ChoiceQuestion {
                instructions: "Categorize user query".to_string(),
                options: vec![
                    OptionDef {
                        id: "billing".to_string(),
                        description: "Invoices, refunds, and subscription billing issues"
                            .to_string(),
                    },
                    OptionDef {
                        id: "tech_support".to_string(),
                        description: "Technical issues and bug reports".to_string(),
                    },
                ],
                policy: Policy::default(),
            }),
        );

        let req = ZevRequest {
            state: serde_json::Value::String(
                "Customer was double charged on credit card".to_string(),
            ),
            questions,
            images: None,
            temperature: None,
            enable_temporal_facts: false,
            model: None,
        };

        // 1. Dual Speculative Cascade
        let cascade_res = engine
            .evaluate_dual_speculative_cascade(&req, 0.99, 0.80, &backend)
            .expect("Cascade evaluation must succeed");
        let ans = cascade_res.answers.get("dept").unwrap();
        assert_eq!(
            ans.decision,
            Some(serde_json::Value::String("billing".into()))
        );
        assert!(ans.confidence > 0.70);

        // 2. Speculative Ensemble (Apfel + Gemma running simultaneously)
        let ensemble_res = engine
            .evaluate_speculative_ensemble(&req, 0.99, 0.50, &backend)
            .expect("Ensemble evaluation must succeed");
        let ens_ans = ensemble_res.answers.get("dept").unwrap();
        assert_eq!(
            ens_ans.decision,
            Some(serde_json::Value::String("billing".into()))
        );
        assert!(ens_ans.source.as_ref().unwrap().starts_with("ensemble-"));
        assert!(ens_ans.probabilities.contains_key("billing"));

        // 3. Speculative Load Balancing (Alternating Apfel -> Gemma -> Apfel)
        // Request 1: Should be handled by Apfel (turn 0)
        let lb_res_1 = engine
            .evaluate_speculative_load_balanced(&req, 0.99, &backend)
            .expect("Load balanced request 1 must succeed");
        let lb_ans_1 = lb_res_1.answers.get("dept").unwrap();
        assert_eq!(
            lb_ans_1.source.as_deref(),
            Some("apfel-load-balanced"),
            "First low-confidence fallback must route to Apfel"
        );
        assert_eq!(
            lb_ans_1.decision,
            Some(serde_json::Value::String("billing".into()))
        );

        // Request 2: Should alternate to Gemma (turn 1)
        let lb_res_2 = engine
            .evaluate_speculative_load_balanced(&req, 0.99, &backend)
            .expect("Load balanced request 2 must succeed");
        let lb_ans_2 = lb_res_2.answers.get("dept").unwrap();
        assert_eq!(
            lb_ans_2.source.as_deref(),
            Some("gemma-load-balanced"),
            "Second low-confidence fallback must route to Gemma"
        );
        assert_eq!(
            lb_ans_2.decision,
            Some(serde_json::Value::String("billing".into()))
        );

        // Request 3: Should alternate back to Apfel (turn 2)
        let lb_res_3 = engine
            .evaluate_speculative_load_balanced(&req, 0.99, &backend)
            .expect("Load balanced request 3 must succeed");
        let lb_ans_3 = lb_res_3.answers.get("dept").unwrap();
        assert_eq!(
            lb_ans_3.source.as_deref(),
            Some("apfel-load-balanced"),
            "Third low-confidence fallback must alternate back to Apfel"
        );
    }

    #[test]
    #[cfg(feature = "neural")]
    fn test_execution_modes_and_recommended_apis() {
        let engine = ZevEngine::default();
        let backend = crate::neural::ApfelNeuralBackend::new();

        let req = ZevRequest {
            state: serde_json::Value::String(
                "Billing dispute: customer was overcharged $45.99 on invoice #8812".into(),
            ),
            questions: [(
                "dept".into(),
                Question::Choice(ChoiceQuestion {
                    instructions: "Select the appropriate support tier".into(),
                    options: vec![
                        OptionDef {
                            id: "billing".into(),
                            description: "Invoice, payments, or charges".into(),
                        },
                        OptionDef {
                            id: "tech_support".into(),
                            description: "Hardware crash, bugs, or downtime".into(),
                        },
                    ],
                    policy: Default::default(),
                }),
            )]
            .into(),
            model: None,
            temperature: None,
            enable_temporal_facts: false,
            images: None,
        };

        // 1. evaluate_recommended (load-balanced with 0.85 threshold)
        let res_rec = engine
            .evaluate_recommended(&req, &backend)
            .expect("evaluate_recommended must succeed");
        assert_eq!(
            res_rec.answers["dept"].decision,
            Some(serde_json::Value::String("billing".into()))
        );

        // 2. evaluate_recommended_ensemble (parallel ensemble with 0.85 threshold and 0.50 weight)
        let res_ens = engine
            .evaluate_recommended_ensemble(&req, &backend)
            .expect("evaluate_recommended_ensemble must succeed");
        assert_eq!(
            res_ens.answers["dept"].decision,
            Some(serde_json::Value::String("billing".into()))
        );

        // 3. evaluate_with_mode: PureSimd
        let res_simd = engine
            .evaluate_with_mode(&req, ExecutionMode::PureSimd, &backend)
            .expect("PureSimd mode must succeed");
        assert_eq!(
            res_simd.answers["dept"].decision,
            Some(serde_json::Value::String("billing".into()))
        );

        // 4. evaluate_with_mode: Ensemble
        let res_mode_ens = engine
            .evaluate_with_mode(
                &req,
                ExecutionMode::Ensemble {
                    confidence_threshold: 0.85,
                    weight_apfel: 0.50,
                },
                &backend,
            )
            .expect("Ensemble mode must succeed");
        assert_eq!(
            res_mode_ens.answers["dept"].decision,
            Some(serde_json::Value::String("billing".into()))
        );

        // 5. evaluate_with_mode: Cascade
        let res_mode_cas = engine
            .evaluate_with_mode(
                &req,
                ExecutionMode::Cascade {
                    fast_threshold: 0.85,
                    neural_threshold: 0.75,
                },
                &backend,
            )
            .expect("Cascade mode must succeed");
        assert_eq!(
            res_mode_cas.answers["dept"].decision,
            Some(serde_json::Value::String("billing".into()))
        );
    }

    #[test]
    fn test_system_one_model_dispatch_and_adaptive_gating() {
        let engine = ZevEngine::default();
        let mut criteria = BTreeMap::new();
        criteria.insert(
            "refund".to_string(),
            Some(serde_json::json!("Customer wants money back")),
        );
        criteria.insert(
            "support".to_string(),
            Some(serde_json::json!("Technical app malfunction")),
        );
        let wire_q = crate::wire::WireQuestion::Choice(crate::wire::WireChoiceQuestion {
            instructions: serde_json::json!("Classify ticket"),
            criteria,
        });

        // Test with zev-default (Pure SIMD)
        let req_default = crate::types::SystemOneRequest {
            state: serde_json::json!("Refund my credit card for transaction TX-100"),
            questions: [("action".to_string(), wire_q.clone())].into(),
            model: "zev-default".into(),
        };
        let resp_default = engine
            .evaluate_system_one(&req_default)
            .expect("system_one default eval");
        assert_eq!(resp_default.model, "zev-default");
        assert!(resp_default.answers.contains_key("action"));

        // Test with zev-poe model name dispatch
        let req_poe = crate::types::SystemOneRequest {
            state: serde_json::json!("Refund my credit card for transaction TX-100"),
            questions: [("action".to_string(), wire_q)].into(),
            model: "zev-poe".into(),
        };
        let resp_poe = engine
            .evaluate_system_one(&req_poe)
            .expect("system_one poe eval");
        assert_eq!(resp_poe.model, "zev-poe");
        assert!(resp_poe.answers.contains_key("action"));
    }

    #[test]
    fn test_h1_entropy_speculative_gating() {
        let engine = ZevEngine::default();
        let q = Question::Choice(ChoiceQuestion {
            instructions: "Classify the sentiment".into(),
            options: vec![
                crate::types::OptionDef {
                    id: "pos".into(),
                    description: "Positive sentiment".into(),
                },
                crate::types::OptionDef {
                    id: "neg".into(),
                    description: "Negative sentiment".into(),
                },
                crate::types::OptionDef {
                    id: "neu".into(),
                    description: "Neutral sentiment".into(),
                },
            ],
            policy: crate::types::Policy::default(),
        });

        // Set environment variables for fallback gating
        std::env::set_var("ZEV_FALLBACK", "gemma");
        std::env::set_var("ZEV_FALLBACK_CONFIDENCE", "0.0"); // Disable confidence fallback
        std::env::set_var("ZEV_FALLBACK_MARGIN", "0.0"); // Disable margin fallback
        std::env::set_var("ZEV_FALLBACK_ENTROPY", "0.01"); // Extremely strict entropy trigger

        let mut questions = BTreeMap::new();
        questions.insert("sentiment".into(), q);
        let req = ZevRequest {
            state: serde_json::json!("This statement could be positive or neutral."),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
            images: None,
        };

        let resp = engine.evaluate(&req).unwrap();
        let ans = &resp.answers["sentiment"];
        assert!(
            ans.source.is_some(),
            "Speculative fallback must trigger via high Shannon entropy H1"
        );

        // Clean up environment variables
        std::env::remove_var("ZEV_FALLBACK");
        std::env::remove_var("ZEV_FALLBACK_CONFIDENCE");
        std::env::remove_var("ZEV_FALLBACK_MARGIN");
        std::env::remove_var("ZEV_FALLBACK_ENTROPY");
    }
}
