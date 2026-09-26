//! Sequential Logic, NAND Flip-Flops & Register Memory
//!
//! Implements cross-coupled NAND SR latches, Master-Slave D Flip-Flops,
//! and N-bit synchronous register files. Models the "memory half-life" / decay
//! inherent in probabilistic LLM feedback loops vs. Zev's deterministic retention.

use crate::logic::gate::ZevLogicEngine;

/// Cross-coupled NAND SR Latch.
///
/// Built from two cross-coupled NAND gates:
/// - `Q = NAND(S_bar, Q_bar)`
/// - `Q_bar = NAND(R_bar, Q)`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrLatch {
    pub q: bool,
    pub q_bar: bool,
}

impl Default for SrLatch {
    fn default() -> Self {
        Self::new()
    }
}

impl SrLatch {
    pub fn new() -> Self {
        Self {
            q: false,
            q_bar: true,
        }
    }

    /// Evaluates the SR latch with active-low inputs (S_bar, R_bar).
    ///
    /// Truth Table:
    /// - `S_bar=1, R_bar=1`: Hold state (Memory)
    /// - `S_bar=0, R_bar=1`: Set (Q=1, Q_bar=0)
    /// - `S_bar=1, R_bar=0`: Reset (Q=0, Q_bar=1)
    /// - `S_bar=0, R_bar=0`: Meta-stable / Forbidden (Q=1, Q_bar=1)
    pub fn update(&mut self, s_bar: bool, r_bar: bool) -> (bool, bool) {
        let mut engine = ZevLogicEngine::new();

        if !s_bar && !r_bar {
            // Forbidden / Meta-stable state in physical NAND latch
            self.q = true;
            self.q_bar = true;
            return (self.q, self.q_bar);
        }

        // Iterative relaxation to find fixed point of cross-coupled gates
        for _ in 0..4 {
            let next_q = engine.nand(s_bar, self.q_bar).0;
            let next_q_bar = engine.nand(r_bar, self.q).0;
            if next_q == self.q && next_q_bar == self.q_bar {
                break;
            }
            self.q = next_q;
            self.q_bar = next_q_bar;
        }

        (self.q, self.q_bar)
    }
}

/// Gated D-Latch synthesized with 4 NAND gates.
///
/// When `clk = 1`, latch is transparent: `Q <= D`.
/// When `clk = 0`, latch is opaque: retains previous `Q`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedDLatch {
    pub q: bool,
    pub q_bar: bool,
    pub gate_evals: usize,
}

impl Default for GatedDLatch {
    fn default() -> Self {
        Self::new()
    }
}

impl GatedDLatch {
    pub fn new() -> Self {
        Self {
            q: false,
            q_bar: true,
            gate_evals: 0,
        }
    }

    pub fn update(&mut self, d: bool, clk: bool) -> bool {
        let mut engine = ZevLogicEngine::new();

        // 1. not_d = NAND(d, d)
        let not_d = engine.nand(d, d).0;

        // 2. s_bar = NAND(d, clk)
        let s_bar = engine.nand(d, clk).0;

        // 3. r_bar = NAND(not_d, clk)
        let r_bar = engine.nand(not_d, clk).0;

        // 4. Cross-coupled SR stage
        let mut latch = SrLatch {
            q: self.q,
            q_bar: self.q_bar,
        };
        let (q, q_bar) = latch.update(s_bar, r_bar);

        self.q = q;
        self.q_bar = q_bar;
        self.gate_evals += engine.gate_evaluations + 2; // ~5 NAND gates total

        self.q
    }
}

/// Positive Edge-Triggered Master-Slave D Flip-Flop.
///
/// Built from two gated D-latches (Master + Slave) connected in series
/// with inverted clock signals. State updates strictly on the rising clock edge (0 -> 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlipFlop {
    master: GatedDLatch,
    slave: GatedDLatch,
    last_clk: bool,
    pub q: bool,
    pub q_bar: bool,
    pub total_gate_evals: usize,
}

impl Default for DFlipFlop {
    fn default() -> Self {
        Self::new()
    }
}

impl DFlipFlop {
    pub fn new() -> Self {
        Self {
            master: GatedDLatch::new(),
            slave: GatedDLatch::new(),
            last_clk: false,
            q: false,
            q_bar: true,
            total_gate_evals: 0,
        }
    }

    /// Ticks the flip-flop with data `d` and clock signal `clk`.
    /// Returns the updated `Q` output.
    pub fn tick(&mut self, d: bool, clk: bool) -> bool {
        // Master latch is active when clk is LOW
        let master_out = self.master.update(d, !clk);

        // Slave latch is active when clk is HIGH
        let slave_out = self.slave.update(master_out, clk);

        self.q = slave_out;
        self.q_bar = !slave_out;
        self.last_clk = clk;
        self.total_gate_evals += self.master.gate_evals + self.slave.gate_evals;

        self.q
    }
}

/// N-bit Synchronous Register built from an array of D Flip-Flops.
#[derive(Debug, Clone)]
pub struct Register {
    pub bit_width: usize,
    flip_flops: Vec<DFlipFlop>,
    pub value: u64,
}

impl Register {
    pub fn new(bit_width: usize) -> Self {
        Self {
            bit_width,
            flip_flops: (0..bit_width).map(|_| DFlipFlop::new()).collect(),
            value: 0,
        }
    }

    /// Resets all flip-flops to zero.
    pub fn reset(&mut self) {
        for ff in &mut self.flip_flops {
            *ff = DFlipFlop::new();
        }
        self.value = 0;
    }

    /// Clock pulse: updates register value if `write_enable` is high.
    pub fn clock_pulse(&mut self, next_value: u64, write_enable: bool) -> u64 {
        let mask = if self.bit_width >= 64 {
            u64::MAX
        } else {
            (1u64 << self.bit_width) - 1
        };

        let target_data = if write_enable {
            next_value & mask
        } else {
            self.value
        };

        // Rising edge: clk 0 -> 1
        let mut new_val = 0u64;
        for i in 0..self.bit_width {
            let bit_in = ((target_data >> i) & 1) == 1;
            // Complete clock cycle (falling phase then rising phase)
            self.flip_flops[i].tick(bit_in, false);
            let bit_out = self.flip_flops[i].tick(bit_in, true);
            if bit_out {
                new_val |= 1u64 << i;
            }
        }

        self.value = new_val;
        self.value
    }
}

/// Analysis helper computing memory survival probability (retention vs decay)
/// across clock cycles for an LLM gate simulation vs Zev.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryHalfLifeReport {
    pub clock_cycles: usize,
    pub gates_per_bit: usize,
    pub single_gate_accuracy: f64,
    pub llm_retention_probability: f64,
    pub zev_retention_probability: f64,
    pub llm_half_life_cycles: f64,
}

impl MemoryHalfLifeReport {
    pub fn compute(
        bit_width: usize,
        clock_cycles: usize,
        single_gate_accuracy: f64,
    ) -> Self {
        // A master-slave D flip-flop uses ~9 NAND gates per bit
        let gates_per_bit = 9;
        let total_gates = bit_width * gates_per_bit;

        // P(register survives 1 cycle) = p^(total_gates)
        let p_per_cycle = single_gate_accuracy.powi(total_gates as i32);

        // P(register survives T cycles) = p^(total_gates * T)
        let llm_retention = p_per_cycle.powi(clock_cycles as i32);

        // Half-life: cycles T where p^(total_gates * T) = 0.5
        // T * total_gates * ln(p) = ln(0.5) => T = ln(0.5) / (total_gates * ln(p))
        let half_life = if single_gate_accuracy >= 1.0 {
            f64::INFINITY
        } else {
            (0.5f64).ln() / (total_gates as f64 * single_gate_accuracy.ln())
        };

        Self {
            clock_cycles,
            gates_per_bit,
            single_gate_accuracy,
            llm_retention_probability: llm_retention,
            zev_retention_probability: 1.0, // 100% deterministic calibration
            llm_half_life_cycles: half_life,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sr_latch_set_reset_hold() {
        let mut latch = SrLatch::new();

        // Initially Q=0, Q_bar=1
        assert_eq!((latch.q, latch.q_bar), (false, true));

        // SET: S_bar=0, R_bar=1
        latch.update(false, true);
        assert_eq!((latch.q, latch.q_bar), (true, false));

        // HOLD: S_bar=1, R_bar=1
        latch.update(true, true);
        assert_eq!((latch.q, latch.q_bar), (true, false));

        // RESET: S_bar=1, R_bar=0
        latch.update(true, false);
        assert_eq!((latch.q, latch.q_bar), (false, true));

        // HOLD: S_bar=1, R_bar=1
        latch.update(true, true);
        assert_eq!((latch.q, latch.q_bar), (false, true));
    }

    #[test]
    fn test_d_flip_flop_clocking() {
        let mut dff = DFlipFlop::new();

        // Feed D=1, clock low -> Q remains false
        dff.tick(true, false);
        assert_eq!(dff.q, false);

        // Clock rises -> Q latches to true
        dff.tick(true, true);
        assert_eq!(dff.q, true);

        // Change D=0 while clock high -> Q stays true until next cycle
        dff.tick(false, true);
        assert_eq!(dff.q, true);

        // Clock low then high with D=0 -> Q becomes false
        dff.tick(false, false);
        dff.tick(false, true);
        assert_eq!(dff.q, false);
    }

    #[test]
    fn test_4bit_register() {
        let mut reg = Register::new(4);
        assert_eq!(reg.value, 0);

        // Write 12 (1100b)
        reg.clock_pulse(12, true);
        assert_eq!(reg.value, 12);

        // Disable write enable and attempt writing 7 -> value must hold 12
        reg.clock_pulse(7, false);
        assert_eq!(reg.value, 12);

        // Enable write, write 7
        reg.clock_pulse(7, true);
        assert_eq!(reg.value, 7);
    }

    #[test]
    fn test_memory_half_life_decay() {
        let report = MemoryHalfLifeReport::compute(4, 10, 0.998);
        assert!(report.llm_retention_probability < 0.50); // After 10 cycles, LLM memory < 50%
        assert_eq!(report.zev_retention_probability, 1.0);
    }
}
