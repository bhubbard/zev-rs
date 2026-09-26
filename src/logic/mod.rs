//! Logic Circuit, Native Gate, and Semantic ALU Engine
//!
//! Provides zero-token logic gate primitives, topological DAG compilation,
//! concurrent multi-stage evaluation, and Macro-Gate Semantic ALUs.

pub mod alu;
pub mod circuit;
pub mod gate;

pub use alu::{AluMode, AluOp, AluReport, SemanticAluEngine, SemanticStateAlu};
pub use circuit::{
    CircuitGate, CircuitReport, ProbabilisticCircuitReport, TopologicalCircuit, WireId,
};
pub use gate::{GateType, ProbabilisticGateResult, ZevLogicEngine};
