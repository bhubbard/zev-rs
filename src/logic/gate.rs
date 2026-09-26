//! Native Logic Gates and Fuzzy / Probabilistic Gate Evaluation
//!
//! Provides deterministic zero-token logic gate primitives (NAND, AND, OR, XOR, etc.)
//! and calibrated probabilistic logic gates (fuzzy truth evaluation with confidence margins).

use crate::error::{Result, ZevError};
use crate::readout::BinaryReadout;

/// Fundamental logic gate operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateType {
    Nand,
    And,
    Or,
    Xor,
    Not,
    Nor,
    Xnor,
    Mux,
}

impl GateType {
    /// Returns the minimal number of inputs required for this gate.
    pub fn min_inputs(&self) -> usize {
        match self {
            GateType::Not => 1,
            GateType::Mux => 3, // (sel, a, b)
            _ => 2,
        }
    }

    /// Evaluates the gate deterministically on boolean inputs.
    #[inline(always)]
    pub fn evaluate_bool(&self, inputs: &[bool]) -> bool {
        match self {
            GateType::Nand => {
                let a = inputs.first().copied().unwrap_or(false);
                let b = inputs.get(1).copied().unwrap_or(false);
                !(a && b)
            }
            GateType::And => {
                let a = inputs.first().copied().unwrap_or(false);
                let b = inputs.get(1).copied().unwrap_or(false);
                a && b
            }
            GateType::Or => {
                let a = inputs.first().copied().unwrap_or(false);
                let b = inputs.get(1).copied().unwrap_or(false);
                a || b
            }
            GateType::Xor => {
                let a = inputs.first().copied().unwrap_or(false);
                let b = inputs.get(1).copied().unwrap_or(false);
                a ^ b
            }
            GateType::Not => {
                let a = inputs.first().copied().unwrap_or(false);
                !a
            }
            GateType::Nor => {
                let a = inputs.first().copied().unwrap_or(false);
                let b = inputs.get(1).copied().unwrap_or(false);
                !(a || b)
            }
            GateType::Xnor => {
                let a = inputs.first().copied().unwrap_or(false);
                let b = inputs.get(1).copied().unwrap_or(false);
                !(a ^ b)
            }
            GateType::Mux => {
                let sel = inputs.first().copied().unwrap_or(false);
                let a = inputs.get(1).copied().unwrap_or(false);
                let b = inputs.get(2).copied().unwrap_or(false);
                if sel { b } else { a }
            }
        }
    }

    /// Evaluates probabilistic / fuzzy inputs in range [0.0, 1.0].
    /// Returns the probability of the output being true.
    pub fn evaluate_probabilistic(&self, inputs: &[f64]) -> f64 {
        match self {
            GateType::Nand => {
                let p_a = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_b = inputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                1.0 - (p_a * p_b)
            }
            GateType::And => {
                let p_a = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_b = inputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                p_a * p_b
            }
            GateType::Or => {
                let p_a = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_b = inputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                p_a + p_b - (p_a * p_b)
            }
            GateType::Xor => {
                let p_a = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_b = inputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                p_a * (1.0 - p_b) + p_b * (1.0 - p_a)
            }
            GateType::Not => {
                let p_a = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                1.0 - p_a
            }
            GateType::Nor => {
                let p_a = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_b = inputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                1.0 - (p_a + p_b - (p_a * p_b))
            }
            GateType::Xnor => {
                let p_a = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_b = inputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                1.0 - (p_a * (1.0 - p_b) + p_b * (1.0 - p_a))
            }
            GateType::Mux => {
                let p_sel = inputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_a = inputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let p_b = inputs.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                (1.0 - p_sel) * p_a + p_sel * p_b
            }
        }
    }
}

/// Evaluates discrete and calibrated logic decisions with zero token overhead.
#[derive(Debug, Clone, Default)]
pub struct ZevLogicEngine {
    pub gate_evaluations: usize,
}

impl ZevLogicEngine {
    pub fn new() -> Self {
        Self {
            gate_evaluations: 0,
        }
    }

    /// Evaluates a NAND gate deterministically: returns `(output, confidence)`.
    #[inline(always)]
    pub fn nand(&mut self, a: bool, b: bool) -> (bool, f64) {
        self.gate_evaluations += 1;
        (!(a && b), 1.0)
    }

    /// Evaluates a NOT gate built purely from NAND: `NOT(a) = NAND(a, a)`.
    #[inline(always)]
    pub fn not(&mut self, a: bool) -> (bool, f64) {
        self.nand(a, a)
    }

    /// Evaluates an AND gate built from NANDs: `AND(a, b) = NOT(NAND(a, b))`.
    #[inline(always)]
    pub fn and(&mut self, a: bool, b: bool) -> (bool, f64) {
        let (n, c1) = self.nand(a, b);
        let (out, c2) = self.not(n);
        (out, c1 * c2)
    }

    /// Evaluates an OR gate built from NANDs: `OR(a, b) = NAND(NOT(a), NOT(b))`.
    #[inline(always)]
    pub fn or(&mut self, a: bool, b: bool) -> (bool, f64) {
        let (na, c1) = self.not(a);
        let (nb, c2) = self.not(b);
        let (out, c3) = self.nand(na, nb);
        (out, c1 * c2 * c3)
    }

    /// Evaluates an XOR gate built from 4 NANDs.
    #[inline(always)]
    pub fn xor(&mut self, a: bool, b: bool) -> (bool, f64) {
        // NAND 1: n1 = NAND(a, b)
        let (n1, c1) = self.nand(a, b);
        // NAND 2: n2 = NAND(a, n1)
        let (n2, c2) = self.nand(a, n1);
        // NAND 3: n3 = NAND(b, n1)
        let (n3, c3) = self.nand(b, n1);
        // NAND 4: out = NAND(n2, n3)
        let (out, c4) = self.nand(n2, n3);
        (out, c1 * c2 * c3 * c4)
    }

    /// Evaluates a 2-to-1 Multiplexer built from NANDs: `MUX(sel, a, b) = (NOT sel AND a) OR (sel AND b)`.
    pub fn mux(&mut self, sel: bool, a: bool, b: bool) -> (bool, f64) {
        let (n_sel, c1) = self.not(sel);
        let (term1, c2) = self.nand(n_sel, a);
        let (term2, c3) = self.nand(sel, b);
        let (out, c4) = self.nand(term1, term2);
        (out, c1 * c2 * c3 * c4)
    }

    /// Evaluates a gate on probabilistic inputs with calibrated certainty and margin of abstention.
    pub fn evaluate_probabilistic(
        &mut self,
        gate: GateType,
        inputs: &[f64],
        abstain_margin: f64,
    ) -> Result<ProbabilisticGateResult> {
        self.gate_evaluations += 1;
        if inputs.len() < gate.min_inputs() {
            return Err(ZevError::Internal(format!(
                "Gate {:?} requires at least {} inputs, got {}",
                gate,
                gate.min_inputs(),
                inputs.len()
            )));
        }

        let p_out = gate.evaluate_probabilistic(inputs);
        // Distance from 0.5 decision threshold determines raw certainty [0.0, 1.0]
        let distance = (p_out - 0.5).abs();
        let confidence = distance * 2.0;

        let should_abstain = distance < abstain_margin;
        let decision = p_out >= 0.5;

        Ok(ProbabilisticGateResult {
            decision,
            probability: p_out,
            confidence,
            abstained: should_abstain,
        })
    }

    /// Bridges a raw language model / readout layer to a boolean gate decision.
    pub fn evaluate_from_readout(
        &mut self,
        readout: &BinaryReadout,
        logits: &[f32],
    ) -> (bool, f64) {
        self.gate_evaluations += 1;
        let score = readout.score(logits);
        let decision = score >= 0.5;
        let confidence = ((score - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        (decision, confidence)
    }
}

/// The result of an evaluated probabilistic logic gate.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilisticGateResult {
    pub decision: bool,
    pub probability: f64,
    pub confidence: f64,
    pub abstained: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_nand_truth_table() {
        let mut engine = ZevLogicEngine::new();
        assert_eq!(engine.nand(false, false), (true, 1.0));
        assert_eq!(engine.nand(false, true), (true, 1.0));
        assert_eq!(engine.nand(true, false), (true, 1.0));
        assert_eq!(engine.nand(true, true), (false, 1.0));
        assert_eq!(engine.gate_evaluations, 4);
    }

    #[test]
    fn test_nand_derived_gates() {
        let mut engine = ZevLogicEngine::new();
        // NOT
        assert_eq!(engine.not(false).0, true);
        assert_eq!(engine.not(true).0, false);

        // AND
        assert_eq!(engine.and(false, false).0, false);
        assert_eq!(engine.and(false, true).0, false);
        assert_eq!(engine.and(true, false).0, false);
        assert_eq!(engine.and(true, true).0, true);

        // OR
        assert_eq!(engine.or(false, false).0, false);
        assert_eq!(engine.or(true, false).0, true);
        assert_eq!(engine.or(false, true).0, true);
        assert_eq!(engine.or(true, true).0, true);

        // XOR
        assert_eq!(engine.xor(false, false).0, false);
        assert_eq!(engine.xor(true, false).0, true);
        assert_eq!(engine.xor(false, true).0, true);
        assert_eq!(engine.xor(true, true).0, false);
    }

    #[test]
    fn test_probabilistic_gate() {
        let mut engine = ZevLogicEngine::new();
        // High confidence true inputs -> NAND should be close to false
        let res = engine
            .evaluate_probabilistic(GateType::Nand, &[0.95, 0.90], 0.05)
            .unwrap();
        assert!(!res.decision);
        assert!(!res.abstained);
        assert!(res.probability < 0.2);

        // Ambiguous inputs near boundary
        let ambiguous = engine
            .evaluate_probabilistic(GateType::Nand, &[0.7071, 0.7071], 0.05)
            .unwrap();
        // 0.7071 * 0.7071 = 0.50 -> 1 - 0.50 = 0.50 -> distance ~ 0 -> should abstain
        assert!(ambiguous.abstained);
    }
}
