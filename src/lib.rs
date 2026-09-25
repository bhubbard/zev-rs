pub mod calibration;
pub mod clm;
pub mod compaction;
pub mod decoding;
pub mod engine;
pub mod error;
pub mod order_invariant;
pub mod preprocessor;
pub mod shortlist;
pub mod tev1;
pub mod types;
pub mod wire;

#[cfg(feature = "server")]
pub mod server;

pub use calibration::{compute_ece, fit_temperature, resolve_temperature, scaled_softmax};
pub use clm::{ContrastiveHead, HeadConfig, HybridVerifier, VectorArena};
pub use compaction::{ToolCallRecord, ToolCompactionAction, ToolCompactionDecision, ToolCompactor};
pub use decoding::{decode_decision, generate_candidates, summarize_moments};
pub use engine::{DecisionEngine, Evaluable, ZevEngine};
pub use error::{Result, ZevError};
pub use order_invariant::{
    compute_order_invariant_logits, compute_order_invariant_logits_with_context, PremiseContext,
};
pub use preprocessor::{clean_text, inject_temporal_facts, preprocess_state};
pub use shortlist::shortlist_options;
pub use tev1::{Tev1Request, Tev1Response};
pub use types::*;

#[cfg(feature = "server")]
pub use server::create_router;

#[cfg(feature = "neural")]
pub mod neural;
#[cfg(feature = "neural")]
pub use neural::ApfelNeuralBackend;
