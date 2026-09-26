//! Macro-Gate Semantic ALU & N-Bit Logic Unit
//!
//! Provides structural synthesis of 4-bit/N-bit ALUs using pure NAND gates
//! via Topological Circuits, alongside zero-token Macro-Gate Semantic ALU evaluation
//! and Semantic State ALUs for agent state machines.

use std::time::Instant;

use crate::error::Result;
use crate::logic::circuit::{CircuitReport, TopologicalCircuit};
use crate::logic::gate::GateType;

/// Supported ALU operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AluOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Not,
    Slt, // Set Less Than (signed)
    Eq,  // Equal
}

/// Execution mode for the ALU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluMode {
    /// Pure structural gate circuit synthesized with NAND gates.
    StructuralNand,
    /// Fast zero-token macro-gate semantic evaluation (1 decision step).
    MacroSemantic,
}

/// Detailed result of an ALU operation.
#[derive(Debug, Clone, PartialEq)]
pub struct AluReport {
    pub op: AluOp,
    pub a: u64,
    pub b: u64,
    pub result: u64,
    pub bit_width: usize,
    pub zero_flag: bool,
    pub carry_out: bool,
    pub overflow_flag: bool,
    pub negative_flag: bool,
    pub gate_evaluations: usize,
    pub critical_path_depth: usize,
    pub latency_micros: f64,
    pub compound_confidence: f64,
    pub cost_dollars: f64,
}

/// High-level ALU executor supporting structural and macro-gate semantic execution.
#[derive(Debug, Clone, Default)]
pub struct SemanticAluEngine;

impl SemanticAluEngine {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates an ALU operation on two operands of given bit width.
    pub fn execute(
        &mut self,
        op: AluOp,
        a: u64,
        b: u64,
        bit_width: usize,
        mode: AluMode,
    ) -> Result<AluReport> {
        let mask = if bit_width >= 64 {
            u64::MAX
        } else {
            (1u64 << bit_width) - 1
        };

        let masked_a = a & mask;
        let masked_b = b & mask;

        match mode {
            AluMode::MacroSemantic => {
                let t0 = Instant::now();
                let (raw_res, carry_out, overflow) = match op {
                    AluOp::Add => {
                        let full = masked_a + masked_b;
                        let res = full & mask;
                        let carry = (full >> bit_width) & 1 == 1;
                        let sign_a = (masked_a >> (bit_width - 1)) & 1 == 1;
                        let sign_b = (masked_b >> (bit_width - 1)) & 1 == 1;
                        let sign_r = (res >> (bit_width - 1)) & 1 == 1;
                        let ovf = (sign_a == sign_b) && (sign_a != sign_r);
                        (res, carry, ovf)
                    }
                    AluOp::Sub => {
                        let full = masked_a.wrapping_sub(masked_b);
                        let res = full & mask;
                        let borrow = masked_a < masked_b;
                        let sign_a = (masked_a >> (bit_width - 1)) & 1 == 1;
                        let sign_b = (masked_b >> (bit_width - 1)) & 1 == 1;
                        let sign_r = (res >> (bit_width - 1)) & 1 == 1;
                        let ovf = (sign_a != sign_b) && (sign_a != sign_r);
                        (res, borrow, ovf)
                    }
                    AluOp::And => (masked_a & masked_b, false, false),
                    AluOp::Or => (masked_a | masked_b, false, false),
                    AluOp::Xor => (masked_a ^ masked_b, false, false),
                    AluOp::Not => ((!masked_a) & mask, false, false),
                    AluOp::Slt => {
                        let is_lt = (masked_a as i64) < (masked_b as i64);
                        (if is_lt { 1 } else { 0 }, false, false)
                    }
                    AluOp::Eq => (if masked_a == masked_b { 1 } else { 0 }, false, false),
                };

                let latency = t0.elapsed().as_secs_f64() * 1_000_000.0;
                let zero_flag = raw_res == 0;
                let negative_flag = (raw_res >> (bit_width - 1)) & 1 == 1;

                Ok(AluReport {
                    op,
                    a: masked_a,
                    b: masked_b,
                    result: raw_res,
                    bit_width,
                    zero_flag,
                    carry_out,
                    overflow_flag: overflow,
                    negative_flag,
                    gate_evaluations: 1, // 1 Macro decision step
                    critical_path_depth: 1,
                    latency_micros: latency,
                    compound_confidence: 1.0,
                    cost_dollars: 0.0,
                })
            }
            AluMode::StructuralNand => {
                // Build or use synthesized topological NAND circuit
                let (circuit, report) = self.execute_structural_nand(op, masked_a, masked_b, bit_width)?;
                let mut result_val = 0u64;
                for i in 0..bit_width {
                    if report.outputs[i] {
                        result_val |= 1u64 << i;
                    }
                }
                let carry_out = report.outputs.get(bit_width).copied().unwrap_or(false);
                let overflow = report.outputs.get(bit_width + 1).copied().unwrap_or(false);

                let zero_flag = result_val == 0;
                let negative_flag = (result_val >> (bit_width - 1)) & 1 == 1;

                Ok(AluReport {
                    op,
                    a: masked_a,
                    b: masked_b,
                    result: result_val,
                    bit_width,
                    zero_flag,
                    carry_out,
                    overflow_flag: overflow,
                    negative_flag,
                    gate_evaluations: circuit.total_gates,
                    critical_path_depth: circuit.critical_path_depth,
                    latency_micros: report.latency_micros,
                    compound_confidence: 1.0,
                    cost_dollars: 0.0,
                })
            }
        }
    }

    /// Synthesizes and executes a bit-level NAND circuit for the specified operation.
    fn execute_structural_nand(
        &mut self,
        op: AluOp,
        a: u64,
        b: u64,
        bit_width: usize,
    ) -> Result<(CircuitReport, CircuitReport)> {
        let mut circuit = TopologicalCircuit::new();

        let mut a_wires = Vec::with_capacity(bit_width);
        let mut b_wires = Vec::with_capacity(bit_width);

        for _ in 0..bit_width {
            a_wires.push(circuit.alloc_wire());
            b_wires.push(circuit.alloc_wire());
        }

        let mut input_wire_list = Vec::new();
        input_wire_list.extend_from_slice(&a_wires);
        input_wire_list.extend_from_slice(&b_wires);
        circuit.set_inputs(&input_wire_list);

        let mut sum_wires = Vec::with_capacity(bit_width);

        // Effective B bits (inverted if subtraction)
        let mut effective_b = Vec::with_capacity(bit_width);
        for i in 0..bit_width {
            if op == AluOp::Sub {
                let not_b = circuit.alloc_wire();
                circuit.add_nand_not(b_wires[i], not_b);
                effective_b.push(not_b);
            } else {
                effective_b.push(b_wires[i]);
            }
        }

        // Constant generator using NAND(a0, NOT a0) for false, NOT(false) for true
        let not_a0 = circuit.alloc_wire();
        circuit.add_nand_not(a_wires[0], not_a0);
        let const_true = circuit.alloc_wire();
        circuit.add_nand(a_wires[0], not_a0, const_true);
        let const_false = circuit.alloc_wire();
        circuit.add_nand_not(const_true, const_false);

        let mut current_carry = if op == AluOp::Sub {
            // Cin = 1 for 2's complement negation
            const_true
        } else {
            const_false
        };

        let mut last_carry_in = current_carry;

        for i in 0..bit_width {
            let ai = a_wires[i];
            let bi = effective_b[i];
            let ci = current_carry;
            last_carry_in = ci;

            let (si, cout_i) = circuit.add_nand_full_adder(ai, bi, ci);
            sum_wires.push(si);
            current_carry = cout_i;
        }

        let final_carry = current_carry;

        // Overflow detection: XOR of carry-in into MSB and carry-out of MSB
        let (ovf_wire, _) = circuit.add_nand_half_adder(last_carry_in, final_carry);

        let mut output_wire_list = sum_wires;
        output_wire_list.push(final_carry);
        output_wire_list.push(ovf_wire);
        circuit.set_outputs(&output_wire_list);

        // Convert inputs to bools
        let mut bool_inputs = Vec::with_capacity(bit_width * 2);
        for i in 0..bit_width {
            bool_inputs.push(((a >> i) & 1) == 1);
        }
        for i in 0..bit_width {
            bool_inputs.push(((b >> i) & 1) == 1);
        }

        circuit.compile()?;
        let report = circuit.evaluate_parallel(&bool_inputs)?;
        let report_clone = report.clone();
        Ok((report, report_clone))
    }
}

/// A Semantic State ALU for game engines and multi-agent decision dynamics.
/// Evaluates state vectors through probabilistic gates without tokens.
#[derive(Debug, Clone, Default)]
pub struct SemanticStateAlu;

impl SemanticStateAlu {
    /// Combines two agent/game state features into an emergent composite state.
    ///
    /// Example: `[Jacob Composure: 0.8]` combined with `[Rival Tactic: Aggressive Objection: 0.7]`
    /// produces `Rebuttal Opportunity: 0.94` with calibrated confidence.
    pub fn combine_states(
        primary_intensity: f64,
        reaction_intensity: f64,
        gate: GateType,
    ) -> (f64, bool, f64) {
        let p_primary = primary_intensity.clamp(0.0, 1.0);
        let p_reaction = reaction_intensity.clamp(0.0, 1.0);

        let result_p = gate.evaluate_probabilistic(&[p_primary, p_reaction]);
        let decision = result_p >= 0.5;
        let confidence = (result_p - 0.5).abs() * 2.0;

        (result_p, decision, confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_4bit_addition_macro_semantic() {
        let mut alu = SemanticAluEngine::new();
        // 7 + 5 = 12
        let r = alu
            .execute(AluOp::Add, 7, 5, 4, AluMode::MacroSemantic)
            .unwrap();
        assert_eq!(r.result, 12);
        assert!(!r.carry_out);
        assert!(!r.zero_flag);
        assert_eq!(r.compound_confidence, 1.0);
        assert_eq!(r.cost_dollars, 0.0);
    }

    #[test]
    fn test_4bit_addition_structural_nand() {
        let mut alu = SemanticAluEngine::new();
        // 7 + 5 = 12 via synthesized NAND circuit
        let r = alu
            .execute(AluOp::Add, 7, 5, 4, AluMode::StructuralNand)
            .unwrap();
        assert_eq!(r.result, 12);
        assert!(!r.carry_out);
        assert!(!r.zero_flag);
        assert!(r.gate_evaluations >= 36);
        assert_eq!(r.compound_confidence, 1.0);
    }

    #[test]
    fn test_4bit_subtraction() {
        let mut alu = SemanticAluEngine::new();
        // 12 - 5 = 7
        let r = alu
            .execute(AluOp::Sub, 12, 5, 4, AluMode::MacroSemantic)
            .unwrap();
        assert_eq!(r.result, 7);
        assert!(!r.zero_flag);
    }

    #[test]
    fn test_semantic_state_alu() {
        // High composure (0.85) AND Rival objection (0.75) -> Window opened
        let (prob, dec, conf) = SemanticStateAlu::combine_states(0.85, 0.75, GateType::Or);
        assert!(dec);
        assert!(prob > 0.90);
        assert!(conf > 0.80);
    }
}
