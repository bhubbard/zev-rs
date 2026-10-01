pub mod abstain;
pub mod calibration;
pub mod cascade;
pub mod clef;
pub mod clm;
pub mod compaction;
pub mod concept_knowledge;
pub mod config;
pub mod decoding;
pub mod engine;
pub mod error;
pub mod gemma;
#[cfg(not(target_arch = "wasm32"))]
pub mod grep;
pub mod intent_sieve;
pub mod kv_rewind;
pub mod logic;
pub mod matrix;
#[cfg(not(target_arch = "wasm32"))]
pub mod mcp;
pub mod multimodal;
pub mod order_invariant;
pub mod premise_window;
pub mod preprocessor;
pub mod qos;
pub mod readout;
pub mod semantic_sieve;
pub mod shortlist;
pub mod tabular;
pub mod tev1;
pub mod types;
pub mod wire;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(feature = "server")]
pub mod server;

#[cfg(not(target_arch = "wasm32"))]
pub use mcp::run_stdio_server;

pub use abstain::{is_abstain_candidate, is_abstain_text, ABSTAIN_EXACT, ABSTAIN_PREFIXES};
pub use calibration::{
    compute_binary_brier_score, compute_brier_score, compute_dataset_brier_score, compute_ece,
    fit_temperature, fit_temperature_brier, fit_temperatures_by_type, resolve_temperature,
    scaled_softmax, TypeTemperatureConfig,
};
pub use cascade::{
    CascadeReport, CascadeStage, PredicateCascade, SequentialCascadeRunner, StageKind,
};
pub use clef::{
    CloudflareClefConfig, CloudflareClefProvider, CLEF_FLASH_MODEL, DEFAULT_CLEF_MODEL,
};
pub use clm::{ContrastiveHead, HeadConfig, HybridVerifier, VectorArena};
pub use compaction::{ToolCallRecord, ToolCompactionAction, ToolCompactionDecision, ToolCompactor};
pub use config::{
    config_path, load_or_init, run_setup_wizard, SystemEnvironment, ZevConfig, ZevMethod,
    DEFAULT_APFEL_WEIGHT, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MARGIN_THRESHOLD,
};
pub use decoding::{decode_decision, generate_candidates, smooth_ordinal_probabilities, summarize_moments};
pub use engine::{
    DecisionEngine, Evaluable, ExecutionMode, ZevEngine, RECOMMENDED_CASCADE_NEURAL_THRESHOLD,
    RECOMMENDED_CONFIDENCE_THRESHOLD, RECOMMENDED_ENSEMBLE_WEIGHT_APFEL,
};
pub use error::{Result, ZevError};
#[cfg(not(target_arch = "wasm32"))]
pub use grep::*;
pub use kv_rewind::{BlockTable, PagedContextArena, DEFAULT_PAGE_SIZE};
pub use logic::{
    AluMode, AluOp, AluReport, CircuitGate, CircuitReport, ClockMode, ClockPulse, CpuCycleReport,
    CpuRunReport, DFlipFlop, GateType, GatedDLatch, MemoryHalfLifeReport, MicroInstruction,
    ProbabilisticCircuitReport, ProbabilisticClock, ProbabilisticGateResult, Register,
    SemanticAluEngine, SemanticStateAlu, SrLatch, TopologicalCircuit, WireId, ZevLogicEngine,
    ZevMicroCpu,
};
pub use matrix::{AnchorPartnerEvaluator, PartnerMatrix};
pub use multimodal::{MultimodalTriageEngine, VisualFeature};
pub use order_invariant::{
    compute_order_invariant_logits, compute_order_invariant_logits_with_context, PremiseContext,
};
pub use preprocessor::{clean_text, inject_temporal_facts, preprocess_state};
pub use qos::elevate_thread_qos;
pub use readout::{
    BinaryReadout, ClassTokenPool, PrunedHead, DEFAULT_FALSE_SPELLINGS, DEFAULT_TRUE_SPELLINGS,
};
pub use semantic_sieve::{CandidateVector, RemoteEmbeddingProvider, SemanticSieve, SieveResult};
pub use shortlist::shortlist_options;
pub use tabular::{
    BatchExecutionReport, RunningStats, TabularBatch, TabularEngine, TabularFilterPredicate,
    TabularRow,
};
pub use tev1::{Tev1Request, Tev1Response};
pub use types::*;

#[cfg(feature = "server")]
pub use server::create_router;

#[cfg(feature = "candle")]
pub mod kev;

#[cfg(feature = "neural")]
pub mod neural;
#[cfg(feature = "neural")]
pub use neural::ApfelNeuralBackend;

#[cfg(feature = "mlx")]
pub mod mlx;
#[cfg(feature = "mlx")]
pub use mlx::{
    DlqClusterSummary, MlxDlqClusterer, MlxProjectionHead, MlxSemanticSieve, MlxTriageClassifier,
    SlmModelFamily,
};
