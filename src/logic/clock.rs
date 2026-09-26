//! Probabilistic Clock & Asynchronous Gated Oscillators
//!
//! Provides deterministic and stochastic clock sources:
//! - Synchronous Quartz-style oscillator
//! - Poisson / Markovian stochastic clock (probabilistic firing rates)
//! - Confidence-Gated Clock (ticks only when calibrated confidence passes threshold, preventing race conditions)
//! - Thermal / Jitter Clock (Gaussian phase noise and timing fluctuations)

use std::time::Instant;

/// Clock operation mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockMode {
    /// Pure synchronous oscillator with fixed period in microseconds.
    Synchronous { period_micros: f64 },

    /// Probabilistic clock based on a Poisson process with rate lambda (ticks/sec).
    ProbabilisticPoisson { lambda: f64 },

    /// Asynchronous wavefront clock: only triggers a tick when logic confidence >= threshold.
    ConfidenceGated { threshold: f64 },

    /// Thermal jitter oscillator with Gaussian variation around a nominal period.
    ThermalJitter {
        nominal_period_micros: f64,
        jitter_std_dev: f64,
    },
}

/// The result of a clock evaluation step.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockPulse {
    pub tick_number: u64,
    pub clk_level: bool,
    pub rising_edge: bool,
    pub falling_edge: bool,
    pub elapsed_micros: f64,
    pub confidence: f64,
    pub suppressed_by_confidence: bool,
}

/// A configurable clock engine supporting deterministic, probabilistic, and confidence-gated modes.
#[derive(Debug, Clone)]
pub struct ProbabilisticClock {
    pub mode: ClockMode,
    pub current_tick: u64,
    pub clk_level: bool,
    start_time: Instant,
    last_tick_time: Instant,
    rng_state: u64,
}

impl ProbabilisticClock {
    pub fn new(mode: ClockMode) -> Self {
        let now = Instant::now();
        Self {
            mode,
            current_tick: 0,
            clk_level: false,
            start_time: now,
            last_tick_time: now,
            // Simple XorShift64 PRNG state for zero-dependency stochastic sampling
            rng_state: 0x853c49e6748fea9b,
        }
    }

    /// Advances the PRNG and returns a uniform float in [0.0, 1.0).
    fn next_uniform(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }

    /// Samples standard normal using Box-Muller transform.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_uniform().max(1e-10);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Evaluates whether a clock tick should occur.
    ///
    /// - If `ConfidenceGated`, evaluates against `logic_confidence`.
    /// - If `ProbabilisticPoisson`, samples a Poisson probability transition.
    /// - If `ThermalJitter`, adds Gaussian timing delay.
    pub fn step(&mut self, logic_confidence: Option<f64>) -> ClockPulse {
        let conf = logic_confidence.unwrap_or(1.0);
        let now = Instant::now();
        let elapsed = now.duration_since(self.start_time).as_secs_f64() * 1_000_000.0;

        let (should_tick, suppressed) = match self.mode {
            ClockMode::Synchronous { .. } => (true, false),

            ClockMode::ConfidenceGated { threshold } => {
                if conf >= threshold {
                    (true, false)
                } else {
                    (false, true)
                }
            }

            ClockMode::ProbabilisticPoisson { lambda } => {
                // Probability of tick in short step: p = 1 - exp(-lambda * dt)
                let dt = now.duration_since(self.last_tick_time).as_secs_f64().max(1e-6);
                let p_tick = 1.0 - (-lambda * dt).exp();
                let sample = self.next_uniform();
                (sample < p_tick, false)
            }

            ClockMode::ThermalJitter {
                nominal_period_micros,
                jitter_std_dev,
            } => {
                let jitter = self.next_gaussian() * jitter_std_dev;
                let actual_period = (nominal_period_micros + jitter).max(0.1);
                let since_last = now.duration_since(self.last_tick_time).as_secs_f64() * 1_000_000.0;
                (since_last >= actual_period, false)
            }
        };

        if should_tick {
            self.last_tick_time = now;
            self.clk_level = !self.clk_level;
            if self.clk_level {
                self.current_tick += 1;
            }
        }

        let rising = should_tick && self.clk_level;
        let falling = should_tick && !self.clk_level;

        ClockPulse {
            tick_number: self.current_tick,
            clk_level: self.clk_level,
            rising_edge: rising,
            falling_edge: falling,
            elapsed_micros: elapsed,
            confidence: conf,
            suppressed_by_confidence: suppressed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronous_clock_edges() {
        let mut clk = ProbabilisticClock::new(ClockMode::Synchronous {
            period_micros: 10.0,
        });

        let p1 = clk.step(None);
        assert!(p1.rising_edge);
        assert_eq!(p1.tick_number, 1);

        let p2 = clk.step(None);
        assert!(p2.falling_edge);
        assert_eq!(p2.tick_number, 1);

        let p3 = clk.step(None);
        assert!(p3.rising_edge);
        assert_eq!(p3.tick_number, 2);
    }

    #[test]
    fn test_confidence_gated_clock() {
        let mut clk = ProbabilisticClock::new(ClockMode::ConfidenceGated { threshold: 0.90 });

        // Confidence 0.70 < 0.90 -> Tick suppressed
        let p1 = clk.step(Some(0.70));
        assert!(p1.suppressed_by_confidence);
        assert_eq!(p1.tick_number, 0);

        // Confidence 0.95 >= 0.90 -> Tick occurs
        let p2 = clk.step(Some(0.95));
        assert!(!p2.suppressed_by_confidence);
        assert!(p2.rising_edge);
        assert_eq!(p2.tick_number, 1);
    }
}
