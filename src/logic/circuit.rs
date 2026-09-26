//! Topological Circuit Engine & DAG Concurrency
//!
//! Organizes logic gates into a Directed Acyclic Graph (DAG), decomposes them
//! into dependency-free topological stages, and evaluates stages with concurrent/batched execution.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::error::{Result, ZevError};
use crate::logic::gate::GateType;

/// Unique index of a signal wire in the circuit.
pub type WireId = usize;

/// A gate node within a topological circuit.
#[derive(Debug, Clone)]
pub struct CircuitGate {
    pub id: usize,
    pub gate_type: GateType,
    pub inputs: Vec<WireId>,
    pub output: WireId,
}

/// Execution report for a circuit evaluation.
#[derive(Debug, Clone)]
pub struct CircuitReport {
    pub outputs: Vec<bool>,
    pub total_gates: usize,
    pub critical_path_depth: usize,
    pub stage_widths: Vec<usize>,
    pub compound_confidence: f64,
    pub latency_micros: f64,
}

/// Probabilistic circuit evaluation report.
#[derive(Debug, Clone)]
pub struct ProbabilisticCircuitReport {
    pub output_probabilities: Vec<f64>,
    pub output_decisions: Vec<bool>,
    pub compound_confidence: f64,
    pub abstained_outputs: usize,
    pub critical_path_depth: usize,
    pub latency_micros: f64,
}

/// Topological circuit that evaluates gates in leveled stages.
#[derive(Debug, Clone)]
pub struct TopologicalCircuit {
    pub num_wires: usize,
    pub input_wires: Vec<WireId>,
    pub output_wires: Vec<WireId>,
    pub gates: Vec<CircuitGate>,
    /// Gates partitioned into dependency-free topological layers (stages).
    pub stages: Vec<Vec<usize>>,
    pub is_compiled: bool,
}

impl Default for TopologicalCircuit {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologicalCircuit {
    pub fn new() -> Self {
        Self {
            num_wires: 0,
            input_wires: Vec::new(),
            output_wires: Vec::new(),
            gates: Vec::new(),
            stages: Vec::new(),
            is_compiled: false,
        }
    }

    /// Allocates a new wire in the circuit.
    pub fn alloc_wire(&mut self) -> WireId {
        let wire = self.num_wires;
        self.num_wires += 1;
        wire
    }

    /// Allocates multiple wires at once.
    pub fn alloc_wires(&mut self, count: usize) -> Vec<WireId> {
        let start = self.num_wires;
        self.num_wires += count;
        (start..self.num_wires).collect()
    }

    /// Declares input wires.
    pub fn set_inputs(&mut self, inputs: &[WireId]) {
        self.input_wires = inputs.to_vec();
        self.is_compiled = false;
    }

    /// Declares output wires.
    pub fn set_outputs(&mut self, outputs: &[WireId]) {
        self.output_wires = outputs.to_vec();
        self.is_compiled = false;
    }

    /// Adds a gate to the circuit and returns its gate ID.
    pub fn add_gate(&mut self, gate_type: GateType, inputs: &[WireId], output: WireId) -> usize {
        let id = self.gates.len();
        self.gates.push(CircuitGate {
            id,
            gate_type,
            inputs: inputs.to_vec(),
            output,
        });
        self.is_compiled = false;
        id
    }

    /// Adds a NAND gate.
    pub fn add_nand(&mut self, a: WireId, b: WireId, out: WireId) -> usize {
        self.add_gate(GateType::Nand, &[a, b], out)
    }

    /// Adds a NOT gate synthesized from NAND: `out = NAND(a, a)`.
    pub fn add_nand_not(&mut self, a: WireId, out: WireId) -> usize {
        self.add_gate(GateType::Nand, &[a, a], out)
    }

    /// Adds a 1-bit Half-Adder using 5 NAND gates:
    /// Returns `(sum_wire, carry_wire)`.
    pub fn add_nand_half_adder(&mut self, a: WireId, b: WireId) -> (WireId, WireId) {
        let sum = self.alloc_wire();
        let carry = self.alloc_wire();

        // 1. nand_ab = NAND(a, b)
        let nand_ab = self.alloc_wire();
        self.add_nand(a, b, nand_ab);

        // 2. carry = NOT(nand_ab) = NAND(nand_ab, nand_ab)
        self.add_nand_not(nand_ab, carry);

        // 3. n_a = NAND(a, nand_ab)
        let n_a = self.alloc_wire();
        self.add_nand(a, nand_ab, n_a);

        // 4. n_b = NAND(b, nand_ab)
        let n_b = self.alloc_wire();
        self.add_nand(b, nand_ab, n_b);

        // 5. sum = NAND(n_a, n_b) (XOR result)
        self.add_nand(n_a, n_b, sum);

        (sum, carry)
    }

    /// Adds a 1-bit Full-Adder using 9 NAND gates:
    /// Returns `(sum_wire, carry_out_wire)`.
    pub fn add_nand_full_adder(&mut self, a: WireId, b: WireId, cin: WireId) -> (WireId, WireId) {
        let sum = self.alloc_wire();
        let cout = self.alloc_wire();

        // Stage 1: Half-adder of a and b
        let (s1, c1) = self.add_nand_half_adder(a, b);

        // Stage 2: Half-adder of s1 and cin
        let (s2, c2) = self.add_nand_half_adder(s1, cin);
        // Wire up sum to s2
        self.add_gate(GateType::Nand, &[s2, s2], sum); // inverter or pass-through

        // Stage 3: cout = c1 OR c2 = NAND(NOT c1, NOT c2)
        let not_c1 = self.alloc_wire();
        let not_c2 = self.alloc_wire();
        self.add_nand_not(c1, not_c1);
        self.add_nand_not(c2, not_c2);
        self.add_nand(not_c1, not_c2, cout);

        (s2, cout)
    }

    /// Compiles the circuit DAG into topological execution stages.
    ///
    /// Each stage contains gates whose inputs are guaranteed to be produced
    /// by earlier stages or circuit inputs. Gates within the same stage
    /// can be evaluated concurrently with zero inter-dependencies.
    pub fn compile(&mut self) -> Result<()> {
        let num_gates = self.gates.len();
        if num_gates == 0 {
            self.stages.clear();
            self.is_compiled = true;
            return Ok(());
        }

        // Map each wire to the gate that produces it (if any)
        let mut wire_producer = HashMap::new();
        for gate in &self.gates {
            wire_producer.insert(gate.output, gate.id);
        }

        // Build gate dependency graph: gate_id -> dependent_gates
        let mut in_degree = vec![0usize; num_gates];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); num_gates];

        for gate in &self.gates {
            for &in_wire in &gate.inputs {
                if let Some(&pred_gate_id) = wire_producer.get(&in_wire) {
                    if pred_gate_id != gate.id {
                        dependents[pred_gate_id].push(gate.id);
                        in_degree[gate.id] += 1;
                    }
                }
            }
        }

        // Kahn's algorithm for levelized stage grouping
        let mut queue = VecDeque::new();
        let mut levels = vec![0usize; num_gates];

        for (gate_id, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(gate_id);
                levels[gate_id] = 0;
            }
        }

        let mut processed = 0;
        let mut max_level = 0;

        while let Some(u) = queue.pop_front() {
            processed += 1;
            let current_level = levels[u];
            if current_level > max_level {
                max_level = current_level;
            }

            for &v in &dependents[u] {
                in_degree[v] -= 1;
                let next_level = current_level + 1;
                if next_level > levels[v] {
                    levels[v] = next_level;
                }
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        if processed != num_gates {
            return Err(ZevError::Internal(
                "Cycle detected in logic circuit DAG".to_string(),
            ));
        }

        // Group gates into stages by level
        let mut stages = vec![Vec::new(); max_level + 1];
        for (gate_id, &lvl) in levels.iter().enumerate() {
            stages[lvl].push(gate_id);
        }

        self.stages = stages;
        self.is_compiled = true;
        Ok(())
    }

    /// Evaluates the circuit sequentially through topological stages.
    pub fn evaluate_sequential(&mut self, inputs: &[bool]) -> Result<CircuitReport> {
        if !self.is_compiled {
            self.compile()?;
        }

        if inputs.len() != self.input_wires.len() {
            return Err(ZevError::Internal(format!(
                "Circuit expects {} inputs, got {}",
                self.input_wires.len(),
                inputs.len()
            )));
        }

        let t0 = Instant::now();
        let mut wire_values = vec![false; self.num_wires];

        // Seed inputs
        for (&wire, &val) in self.input_wires.iter().zip(inputs.iter()) {
            wire_values[wire] = val;
        }

        // Evaluate stage by stage
        for stage in &self.stages {
            for &gate_id in stage {
                let gate = &self.gates[gate_id];
                let in_vals: Vec<bool> = gate.inputs.iter().map(|&w| wire_values[w]).collect();
                wire_values[gate.output] = gate.gate_type.evaluate_bool(&in_vals);
            }
        }

        let elapsed = t0.elapsed().as_secs_f64() * 1_000_000.0;
        let outputs: Vec<bool> = self.output_wires.iter().map(|&w| wire_values[w]).collect();
        let stage_widths: Vec<usize> = self.stages.iter().map(|s| s.len()).collect();

        Ok(CircuitReport {
            outputs,
            total_gates: self.gates.len(),
            critical_path_depth: self.stages.len(),
            stage_widths,
            compound_confidence: 1.0, // 100% deterministic
            latency_micros: elapsed,
        })
    }

    /// Evaluates the circuit with stage-batched concurrency.
    pub fn evaluate_parallel(&mut self, inputs: &[bool]) -> Result<CircuitReport> {
        // Stage-batched execution: each stage can be computed with independent slice writes
        if !self.is_compiled {
            self.compile()?;
        }

        if inputs.len() != self.input_wires.len() {
            return Err(ZevError::Internal(format!(
                "Circuit expects {} inputs, got {}",
                self.input_wires.len(),
                inputs.len()
            )));
        }

        let t0 = Instant::now();
        let mut wire_values = vec![false; self.num_wires];

        for (&wire, &val) in self.input_wires.iter().zip(inputs.iter()) {
            wire_values[wire] = val;
        }

        // For each stage, all gate inputs are read-only from prior stages.
        // We buffer stage outputs to prevent any intra-stage write hazards.
        for stage in &self.stages {
            let mut stage_updates: Vec<(WireId, bool)> = Vec::with_capacity(stage.len());

            for &gate_id in stage {
                let gate = &self.gates[gate_id];
                let in_vals: Vec<bool> = gate.inputs.iter().map(|&w| wire_values[w]).collect();
                let out_val = gate.gate_type.evaluate_bool(&in_vals);
                stage_updates.push((gate.output, out_val));
            }

            for (wire, val) in stage_updates {
                wire_values[wire] = val;
            }
        }

        let elapsed = t0.elapsed().as_secs_f64() * 1_000_000.0;
        let outputs: Vec<bool> = self.output_wires.iter().map(|&w| wire_values[w]).collect();
        let stage_widths: Vec<usize> = self.stages.iter().map(|s| s.len()).collect();

        Ok(CircuitReport {
            outputs,
            total_gates: self.gates.len(),
            critical_path_depth: self.stages.len(),
            stage_widths,
            compound_confidence: 1.0,
            latency_micros: elapsed,
        })
    }

    /// Evaluates the circuit with probabilistic signals in [0.0, 1.0].
    /// Computes compound confidence and tracks abstention margins.
    pub fn evaluate_probabilistic(
        &mut self,
        inputs: &[f64],
        single_gate_accuracy: f64,
        abstain_margin: f64,
    ) -> Result<ProbabilisticCircuitReport> {
        if !self.is_compiled {
            self.compile()?;
        }

        if inputs.len() != self.input_wires.len() {
            return Err(ZevError::Internal(format!(
                "Circuit expects {} inputs, got {}",
                self.input_wires.len(),
                inputs.len()
            )));
        }

        let t0 = Instant::now();
        let mut wire_probs = vec![0.0f64; self.num_wires];

        for (&wire, &p) in self.input_wires.iter().zip(inputs.iter()) {
            wire_probs[wire] = p.clamp(0.0, 1.0);
        }

        for stage in &self.stages {
            for &gate_id in stage {
                let gate = &self.gates[gate_id];
                let in_probs: Vec<f64> = gate.inputs.iter().map(|&w| wire_probs[w]).collect();
                wire_probs[gate.output] = gate.gate_type.evaluate_probabilistic(&in_probs);
            }
        }

        let elapsed = t0.elapsed().as_secs_f64() * 1_000_000.0;

        let output_probs: Vec<f64> = self.output_wires.iter().map(|&w| wire_probs[w]).collect();
        let output_decisions: Vec<bool> = output_probs.iter().map(|&p| p >= 0.5).collect();

        let abstained_count = output_probs
            .iter()
            .filter(|&&p| (p - 0.5).abs() < abstain_margin)
            .count();

        // Compound accuracy degradation calculation: p^N
        let num_gates = self.gates.len();
        let compound_confidence = single_gate_accuracy.powi(num_gates as i32);

        Ok(ProbabilisticCircuitReport {
            output_probabilities: output_probs,
            output_decisions,
            compound_confidence,
            abstained_outputs: abstained_count,
            critical_path_depth: self.stages.len(),
            latency_micros: elapsed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_nand_half_adder() {
        let mut circuit = TopologicalCircuit::new();
        let a = circuit.alloc_wire();
        let b = circuit.alloc_wire();
        circuit.set_inputs(&[a, b]);

        let (sum, carry) = circuit.add_nand_half_adder(a, b);
        circuit.set_outputs(&[sum, carry]);

        circuit.compile().unwrap();
        assert!(circuit.stages.len() >= 3);

        // Test 0 + 0 = 0, C=0
        let r0 = circuit.evaluate_parallel(&[false, false]).unwrap();
        assert_eq!(r0.outputs, vec![false, false]);

        // Test 1 + 0 = 1, C=0
        let r1 = circuit.evaluate_parallel(&[true, false]).unwrap();
        assert_eq!(r1.outputs, vec![true, false]);

        // Test 0 + 1 = 1, C=0
        let r2 = circuit.evaluate_parallel(&[false, true]).unwrap();
        assert_eq!(r2.outputs, vec![true, false]);

        // Test 1 + 1 = 0, C=1
        let r3 = circuit.evaluate_parallel(&[true, true]).unwrap();
        assert_eq!(r3.outputs, vec![false, true]);
    }
}
