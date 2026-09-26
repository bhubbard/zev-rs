//! Minimal Turing-Complete Sequential Micro-Machine (Zev CPU)
//!
//! Integrates Combinational ALU + Sequential Flip-Flop Register + Probabilistic Clock
//! into a working sequential computer capable of running micro-programs.

use std::time::Instant;

use crate::error::Result;
use crate::logic::alu::{AluMode, AluOp, SemanticAluEngine};
use crate::logic::clock::{ClockMode, ProbabilisticClock};
use crate::logic::sequential::Register;

/// Micro-instructions supported by the minimal CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicroInstruction {
    /// Loads an immediate value into the accumulator: ACC <= imm
    LoadImm(u64),
    /// ACC <= ACC + imm
    Add(u64),
    /// ACC <= ACC - imm
    Sub(u64),
    /// ACC <= ACC AND imm
    And(u64),
    /// ACC <= ACC OR imm
    Or(u64),
    /// ACC <= ACC XOR imm
    Xor(u64),
    /// Jump to target PC if accumulator zero flag is set
    JumpIfZero(usize),
    /// Unconditional jump to target PC
    Jump(usize),
    /// Halts execution
    Halt,
}

/// Cycle report for one executed instruction step.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuCycleReport {
    pub cycle: u64,
    pub pc: usize,
    pub instruction: MicroInstruction,
    pub acc_before: u64,
    pub acc_after: u64,
    pub zero_flag: bool,
    pub carry_flag: bool,
    pub clock_rising: bool,
    pub confidence: f64,
}

/// Summary report after running a program to completion.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuRunReport {
    pub total_cycles: u64,
    pub final_acc: u64,
    pub zero_flag: bool,
    pub carry_flag: bool,
    pub halted: bool,
    pub execution_time_micros: f64,
    pub cycles_per_second: f64,
}

/// A minimal sequential computer integrating ALU, Registers (Flip-Flops), and Clock.
#[derive(Debug, Clone)]
pub struct ZevMicroCpu {
    pub bit_width: usize,
    pub acc: Register,
    pub pc: usize,
    pub clock: ProbabilisticClock,
    pub alu: SemanticAluEngine,
    pub program: Vec<MicroInstruction>,
    pub halted: bool,
    pub cycle_count: u64,
    pub zero_flag: bool,
    pub carry_flag: bool,
}

impl ZevMicroCpu {
    pub fn new(bit_width: usize, clock_mode: ClockMode, program: Vec<MicroInstruction>) -> Self {
        Self {
            bit_width,
            acc: Register::new(bit_width),
            pc: 0,
            clock: ProbabilisticClock::new(clock_mode),
            alu: SemanticAluEngine::new(),
            program,
            halted: false,
            cycle_count: 0,
            zero_flag: true,
            carry_flag: false,
        }
    }

    /// Resets the CPU state.
    pub fn reset(&mut self) {
        self.acc.reset();
        self.pc = 0;
        self.halted = false;
        self.cycle_count = 0;
        self.zero_flag = true;
        self.carry_flag = false;
    }

    /// Steps the CPU forward by one clock cycle.
    pub fn step(&mut self) -> Result<Option<CpuCycleReport>> {
        if self.halted {
            return Ok(None);
        }

        // Fetch instruction
        if self.pc >= self.program.len() {
            self.halted = true;
            return Ok(None);
        }
        let instr = self.program[self.pc];

        // Step clock
        let pulse = self.clock.step(Some(1.0));
        if pulse.suppressed_by_confidence {
            return Ok(None);
        }

        let acc_before = self.acc.value;
        let mut next_pc = self.pc + 1;
        let mut next_acc = acc_before;

        match instr {
            MicroInstruction::LoadImm(val) => {
                next_acc = val;
                self.zero_flag = (val & ((1 << self.bit_width) - 1)) == 0;
                self.carry_flag = false;
            }
            MicroInstruction::Add(imm) => {
                let alu_res = self.alu.execute(
                    AluOp::Add,
                    acc_before,
                    imm,
                    self.bit_width,
                    AluMode::MacroSemantic,
                )?;
                next_acc = alu_res.result;
                self.zero_flag = alu_res.zero_flag;
                self.carry_flag = alu_res.carry_out;
            }
            MicroInstruction::Sub(imm) => {
                let alu_res = self.alu.execute(
                    AluOp::Sub,
                    acc_before,
                    imm,
                    self.bit_width,
                    AluMode::MacroSemantic,
                )?;
                next_acc = alu_res.result;
                self.zero_flag = alu_res.zero_flag;
                self.carry_flag = alu_res.carry_out;
            }
            MicroInstruction::And(imm) => {
                let alu_res = self.alu.execute(
                    AluOp::And,
                    acc_before,
                    imm,
                    self.bit_width,
                    AluMode::MacroSemantic,
                )?;
                next_acc = alu_res.result;
                self.zero_flag = alu_res.zero_flag;
            }
            MicroInstruction::Or(imm) => {
                let alu_res = self.alu.execute(
                    AluOp::Or,
                    acc_before,
                    imm,
                    self.bit_width,
                    AluMode::MacroSemantic,
                )?;
                next_acc = alu_res.result;
                self.zero_flag = alu_res.zero_flag;
            }
            MicroInstruction::Xor(imm) => {
                let alu_res = self.alu.execute(
                    AluOp::Xor,
                    acc_before,
                    imm,
                    self.bit_width,
                    AluMode::MacroSemantic,
                )?;
                next_acc = alu_res.result;
                self.zero_flag = alu_res.zero_flag;
            }
            MicroInstruction::JumpIfZero(target) => {
                if self.zero_flag {
                    next_pc = target;
                }
            }
            MicroInstruction::Jump(target) => {
                next_pc = target;
            }
            MicroInstruction::Halt => {
                self.halted = true;
                return Ok(None);
            }
        }

        // Latch accumulator using flip-flop clock pulse
        let acc_after = self.acc.clock_pulse(next_acc, true);
        self.cycle_count += 1;
        let current_pc = self.pc;
        self.pc = next_pc;

        Ok(Some(CpuCycleReport {
            cycle: self.cycle_count,
            pc: current_pc,
            instruction: instr,
            acc_before,
            acc_after,
            zero_flag: self.zero_flag,
            carry_flag: self.carry_flag,
            clock_rising: pulse.rising_edge,
            confidence: 1.0,
        }))
    }

    /// Runs the CPU until a HALT instruction or maximum cycle count.
    pub fn run(&mut self, max_cycles: usize) -> Result<CpuRunReport> {
        let t0 = Instant::now();
        let mut steps = 0;

        while !self.halted && steps < max_cycles {
            if self.step()?.is_none() {
                break;
            }
            steps += 1;
        }

        let elapsed = t0.elapsed().as_secs_f64() * 1_000_000.0;
        let cps = if elapsed > 0.0 {
            (self.cycle_count as f64) / (elapsed / 1_000_000.0)
        } else {
            0.0
        };

        Ok(CpuRunReport {
            total_cycles: self.cycle_count,
            final_acc: self.acc.value,
            zero_flag: self.zero_flag,
            carry_flag: self.carry_flag,
            halted: self.halted,
            execution_time_micros: elapsed,
            cycles_per_second: cps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_countdown_to_zero() {
        // Program:
        // 0: LoadImm(5)
        // 1: Sub(1)
        // 2: JumpIfZero(4)
        // 3: Jump(1)
        // 4: Halt
        let program = vec![
            MicroInstruction::LoadImm(5),
            MicroInstruction::Sub(1),
            MicroInstruction::JumpIfZero(4),
            MicroInstruction::Jump(1),
            MicroInstruction::Halt,
        ];

        let mut cpu = ZevMicroCpu::new(
            4,
            ClockMode::Synchronous { period_micros: 1.0 },
            program,
        );

        let report = cpu.run(100).unwrap();
        assert!(report.halted);
        assert_eq!(report.final_acc, 0);
        assert!(report.zero_flag);
        assert_eq!(report.total_cycles, 15);
    }

    #[test]
    fn test_cpu_fibonacci() {
        // Program computing 1 + 2 + 3 + 4 = 10
        let program = vec![
            MicroInstruction::LoadImm(0),
            MicroInstruction::Add(1),
            MicroInstruction::Add(2),
            MicroInstruction::Add(3),
            MicroInstruction::Add(4),
            MicroInstruction::Halt,
        ];

        let mut cpu = ZevMicroCpu::new(
            8,
            ClockMode::Synchronous { period_micros: 1.0 },
            program,
        );

        let report = cpu.run(50).unwrap();
        assert!(report.halted);
        assert_eq!(report.final_acc, 10);
    }
}
