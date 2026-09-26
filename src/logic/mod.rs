//! Logic Circuit, Native Gate, Sequential Memory, and Semantic CPU Engine
//!
//! Provides zero-token logic gate primitives, topological DAG compilation,
//! NAND flip-flops, probabilistic clocks, and minimal Turing-complete sequential CPUs.

pub mod alu;
pub mod circuit;
pub mod clock;
pub mod cpu;
pub mod gate;
pub mod sequential;

pub use alu::{AluMode, AluOp, AluReport, SemanticAluEngine, SemanticStateAlu};
pub use circuit::{
    CircuitGate, CircuitReport, ProbabilisticCircuitReport, TopologicalCircuit, WireId,
};
pub use clock::{ClockMode, ClockPulse, ProbabilisticClock};
pub use cpu::{CpuCycleReport, CpuRunReport, MicroInstruction, ZevMicroCpu};
pub use gate::{GateType, ProbabilisticGateResult, ZevLogicEngine};
pub use sequential::{
    DFlipFlop, GatedDLatch, MemoryHalfLifeReport, Register, SrLatch,
};
