pub mod error;
pub mod types;
pub mod preprocessor;
pub mod shortlist;
pub mod order_invariant;
pub mod calibration;
pub mod decoding;
pub mod engine;

#[cfg(feature = "server")]
pub mod server;

pub use error::{Result, ZevError};
pub use types::*;
pub use preprocessor::{clean_text, inject_temporal_facts, preprocess_state};
pub use shortlist::shortlist_options;
pub use order_invariant::{compute_order_invariant_logits, score_candidate_isolated};
pub use calibration::{compute_ece, fit_temperature, resolve_temperature, scaled_softmax};
pub use decoding::{decode_decision, generate_candidates, summarize_moments};
pub use engine::ZevEngine;

#[cfg(feature = "server")]
pub use server::create_router;
