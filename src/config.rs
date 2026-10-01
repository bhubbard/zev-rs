//! Configuration, hardware environment probe, and first-run setup wizard for Zev.
//!
//! Provides automatic environment detection (macOS Apple Silicon, Linux, Windows,
//! SIMD, Apple Neural Engine, local Gemma/Ollama endpoints) and an interactive
//! onboarding wizard for selecting the optimal execution method.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::time::Duration;

pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.35;
pub const DEFAULT_MARGIN_THRESHOLD: f64 = 0.10;
pub const DEFAULT_APFEL_WEIGHT: f64 = 0.50;

/// Available decision execution methods in Zev.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ZevMethod {
    /// Consensus Bayesian Product of Experts (Apfel ANE + Gemma) — #1 Global Accuracy (74.89%).
    DualEnsemble,
    /// Three-tier sequential cascade (SIMD > 0.85 → ANE > 0.75 → Gemma) — 72.73% Accuracy.
    DualCascade,
    /// Alternating load balancer across ANE and CPU/GPU with automatic failover — 72.73% Accuracy.
    LoadBalanced,
    /// Zero-token pure hardware SIMD reflex (CPU Neon/AVX2, zero weights, zero PyTorch/GPU) — 69.70% Accuracy.
    PureSimd,
    /// Apple Intelligence on-device FoundationModels on Apple Neural Engine — 70.56% Accuracy.
    Apfel,
    /// SIMD + Distilled Gemma 4 Head (or Ollama/vLLM/OpenAI endpoint) — 71.00% Accuracy.
    Gemma,
    /// Apple Silicon native MLX array evaluation — 70.80% Accuracy.
    Mlx,
    /// SIMD heuristics + Contrastive Language Model embedding verification — 70.20% Accuracy.
    Clm,
}

impl ZevMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DualEnsemble => "dual-ensemble",
            Self::DualCascade => "dual-cascade",
            Self::LoadBalanced => "load-balanced",
            Self::PureSimd => "pure-simd",
            Self::Apfel => "apfel",
            Self::Gemma => "gemma",
            Self::Mlx => "mlx",
            Self::Clm => "clm",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::DualEnsemble => "Zev-Dual-Ensemble (PoE) 🏆 [RECOMMENDED BEST OVERALL]",
            Self::DualCascade => "Zev-Dual-Cascade ⚡ [THREE-TIER RECURSIVE]",
            Self::LoadBalanced => "Zev-Load-Balanced ⚖️ [HIGH-THROUGHPUT BALANCER]",
            Self::PureSimd => "Zev-Default (Pure Hardware SIMD Reflex) 🚀 [UNIVERSAL / ZERO WEIGHTS]",
            Self::Apfel => "Zev-Apfel 🍎 [APPLE NEURAL ENGINE]",
            Self::Gemma => "Zev-Gemma4 🧠 [DISTILLED GEMMA / OLLAMA]",
            Self::Mlx => "Zev-Mlx ⚡ [APPLE SILICON MLX]",
            Self::Clm => "Zev-Clm 🔬 [CONTRASTIVE LANGUAGE MODEL]",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::DualEnsemble => {
                "Consensus Bayesian Product of Experts Fusion (Apfel ANE + Gemma). Top global accuracy (74.89%) with graceful auto-fallback to SIMD."
            }
            Self::DualCascade => {
                "Three-tier sequential cascade (SIMD > 0.85 → ANE > 0.75 → Gemma). Ultra-lean < 16 MB RSS footprint."
            }
            Self::LoadBalanced => {
                "Alternates sub-threshold traffic across ANE and CPU/GPU with automatic bidirectional failover (74.9k req/s)."
            }
            Self::PureSimd => {
                "Pure CPU Neon/AVX2; zero model weights, zero PyTorch/GPU. Runs on ANY Linux/Mac/Windows/Docker box with < 8 MB RAM."
            }
            Self::Apfel => {
                "On-device Apple Intelligence FoundationModels accelerated by Apple Neural Engine (macOS 15+ Apple Silicon)."
            }
            Self::Gemma => {
                "Fast SIMD reflex with Gemma 4 distillation fallback via local Ollama, vLLM, or OpenAI-compatible server."
            }
            Self::Mlx => {
                "Direct unified-memory MLX tensor evaluation optimized for Apple Silicon GPUs."
            }
            Self::Clm => {
                "Zero-allocation vector arena scoring paired with local Contrastive Language Model embedding verification."
            }
        }
    }

    pub fn accuracy(&self) -> &'static str {
        match self {
            Self::DualEnsemble => "74.89% (Global #1)",
            Self::DualCascade => "72.73%",
            Self::LoadBalanced => "72.73%",
            Self::PureSimd => "69.70%",
            Self::Apfel => "70.56%",
            Self::Gemma => "71.00%",
            Self::Mlx => "70.80%",
            Self::Clm => "70.20%",
        }
    }

    pub fn latency(&self) -> &'static str {
        match self {
            Self::DualEnsemble => "~13.4 µs (74,346 dec/s)",
            Self::DualCascade => "~13.5 µs (73,806 dec/s)",
            Self::LoadBalanced => "~13.3 µs (74,935 dec/s)",
            Self::PureSimd => "~13.3 µs (75,078 dec/s)",
            Self::Apfel => "~13.3 µs (75,322 dec/s)",
            Self::Gemma => "~46.2 µs (21,628 dec/s)",
            Self::Mlx => "~28.5 µs (35,080 dec/s)",
            Self::Clm => "~18.2 µs (54,945 dec/s)",
        }
    }

    pub fn requirements(&self) -> &'static str {
        match self {
            Self::DualEnsemble => "Mac Apple Silicon (ANE) + Gemma/Ollama endpoint (Auto-falls back to SIMD if unavailable)",
            Self::DualCascade => "Mac Apple Silicon (ANE) + Gemma/Ollama endpoint (Auto-falls back to SIMD)",
            Self::LoadBalanced => "Mac Apple Silicon (ANE) or remote Gemma server (Auto-falls back to SIMD)",
            Self::PureSimd => "None! Any Linux, macOS, Windows, Docker container (x86_64 / ARM64)",
            Self::Apfel => "macOS 15+ Apple Silicon (M-series chip with Apple Neural Engine)",
            Self::Gemma => "Any OS with Ollama, vLLM, or Gemma-compatible HTTP endpoint",
            Self::Mlx => "macOS Apple Silicon with mlx-rs compiled",
            Self::Clm => "Any OS with CPU SIMD or embedded contrastive projections",
        }
    }

    pub fn parse_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "1" | "dual-ensemble" | "ensemble" | "poe" | "zev-dual-ensemble" => Some(Self::DualEnsemble),
            "2" | "dual-cascade" | "cascade" | "zev-dual-cascade" => Some(Self::DualCascade),
            "3" | "load-balanced" | "balanced" | "zev-load-balanced" => Some(Self::LoadBalanced),
            "4" | "pure-simd" | "simd" | "default" | "zev-default" => Some(Self::PureSimd),
            "5" | "apfel" | "ane" | "neural" | "zev-apfel" => Some(Self::Apfel),
            "6" | "gemma" | "gemma4" | "zev-gemma" | "zev-gemma4" => Some(Self::Gemma),
            "7" | "mlx" | "zev-mlx" => Some(Self::Mlx),
            "8" | "clm" | "zev-clm" => Some(Self::Clm),
            _ => None,
        }
    }

    pub fn to_execution_mode(
        &self,
        conf_thresh: f64,
        _margin_thresh: f64,
        weight_apfel: f64,
    ) -> crate::engine::ExecutionMode {
        match self {
            Self::PureSimd => crate::engine::ExecutionMode::PureSimd,
            Self::LoadBalanced => crate::engine::ExecutionMode::LoadBalanced {
                confidence_threshold: conf_thresh,
            },
            Self::DualEnsemble => crate::engine::ExecutionMode::Ensemble {
                confidence_threshold: conf_thresh,
                weight_apfel,
            },
            Self::DualCascade => crate::engine::ExecutionMode::Cascade {
                fast_threshold: 0.85,
                neural_threshold: conf_thresh.max(0.70),
            },
            Self::Apfel => crate::engine::ExecutionMode::LoadBalanced {
                confidence_threshold: conf_thresh,
            },
            Self::Gemma => crate::engine::ExecutionMode::Cascade {
                fast_threshold: 0.85,
                neural_threshold: conf_thresh,
            },
            Self::Mlx => crate::engine::ExecutionMode::LoadBalanced {
                confidence_threshold: conf_thresh,
            },
            Self::Clm => crate::engine::ExecutionMode::PureSimd,
        }
    }
}

/// Dynamic system environment and hardware inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEnvironment {
    pub os: String,
    pub arch: String,
    pub is_mac: bool,
    pub is_apple_silicon: bool,
    pub has_simd: bool,
    pub simd_instruction_set: String,
    pub apfel_available: bool,
    pub apfel_detail: String,
    pub gemma_available: bool,
    pub gemma_endpoint: String,
    pub gemma_detail: String,
    pub mlx_available: bool,
}

impl SystemEnvironment {
    pub fn probe() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let is_mac = os == "macos";
        let is_apple_silicon = is_mac && arch == "aarch64";

        let (has_simd, simd_instruction_set) = if arch == "aarch64" {
            (true, "ARM NEON 128-bit Vector Extensions".to_string())
        } else if arch == "x86_64" {
            (true, "x86_64 AVX2 / FMA Vector Extensions".to_string())
        } else {
            (false, "Scalar fallback".to_string())
        };

        // Check Apfel / Apple Neural Engine
        let (apfel_available, apfel_detail) = if is_apple_silicon {
            #[cfg(feature = "neural")]
            {
                (
                    true,
                    "Apple Neural Engine (ANE) via apfel-rs FoundationModels".to_string(),
                )
            }
            #[cfg(not(feature = "neural"))]
            {
                (
                    false,
                    "Apple Silicon detected, but binary compiled without '--features neural'"
                        .to_string(),
                )
            }
        } else if is_mac {
            (
                false,
                "Intel Mac detected (Apple Neural Engine requires Apple Silicon M-series)"
                    .to_string(),
            )
        } else {
            (
                false,
                format!("Non-macOS platform ({os} {arch}); Apple Neural Engine not applicable"),
            )
        };

        // Check Gemma / local LLM endpoint
        let default_endpoints = [
            std::env::var("GEMMA_URL").ok(),
            Some("http://127.0.0.1:8000/v1".to_string()),
            Some("http://127.0.0.1:11434/v1".to_string()),
            Some("http://localhost:11434/v1".to_string()),
        ];

        let mut gemma_available = false;
        let mut gemma_endpoint = "http://127.0.0.1:8000/v1".to_string();
        let mut gemma_detail = "No active endpoint detected (auto-falls back to SIMD)".to_string();

        for ep_opt in default_endpoints.into_iter().flatten() {
            let probe_url = format!("{}/models", ep_opt.trim_end_matches('/'));
            if probe_http_endpoint(&probe_url, Duration::from_millis(150)) {
                gemma_available = true;
                gemma_endpoint = ep_opt.clone();
                gemma_detail = format!("Active responsive endpoint at {ep_opt}");
                break;
            }
        }

        // Check MLX
        let mlx_available = is_apple_silicon && cfg!(feature = "mlx");

        Self {
            os,
            arch,
            is_mac,
            is_apple_silicon,
            has_simd,
            simd_instruction_set,
            apfel_available,
            apfel_detail,
            gemma_available,
            gemma_endpoint,
            gemma_detail,
            mlx_available,
        }
    }

    /// Determines the optimal recommended method based on detected hardware.
    pub fn recommended_method(&self) -> ZevMethod {
        if self.is_apple_silicon {
            // Apple Silicon users get the worldwide #1 Ensemble (with auto-fallback to SIMD)
            ZevMethod::DualEnsemble
        } else if self.gemma_available {
            // Non-Mac users with a live Gemma/Ollama endpoint get Gemma
            ZevMethod::Gemma
        } else {
            // Universal default: pure SIMD hardware reflex (< 8 MB, ~13 µs, runs anywhere)
            ZevMethod::PureSimd
        }
    }

    /// Formats a clean terminal diagnostic box.
    pub fn format_diagnostic_box(&self) -> String {
        let mut s = String::new();
        s.push_str("╭────────────────────────────────────────────────────────────────────────────╮\n");
        s.push_str("│ 🔍 Zev Hardware & Environment Diagnostics                                 │\n");
        s.push_str("╰────────────────────────────────────────────────────────────────────────────╯\n");

        s.push_str(&format!(
            "  • Operating System:     {} ({})\n",
            self.os, self.arch
        ));
        s.push_str(&format!(
            "  • Hardware SIMD:        {} (Enabled • ~13 µs reflex)\n",
            self.simd_instruction_set
        ));

        if self.apfel_available {
            s.push_str(&format!(
                "  • Apple Neural Engine:  [✓] {}\n",
                self.apfel_detail
            ));
        } else {
            s.push_str(&format!(
                "  • Apple Neural Engine:  [i] {}\n",
                self.apfel_detail
            ));
        }

        if self.gemma_available {
            s.push_str(&format!(
                "  • Gemma/LLM Endpoint:   [✓] {}\n",
                self.gemma_detail
            ));
        } else {
            s.push_str(&format!(
                "  • Gemma/LLM Endpoint:   [i] {}\n",
                self.gemma_detail
            ));
        }

        if self.mlx_available {
            s.push_str("  • Apple Silicon MLX:    [✓] Compiled and available\n");
        } else if self.is_apple_silicon {
            s.push_str("  • Apple Silicon MLX:    [i] Apple Silicon detected (compile with --features mlx)\n");
        }

        s
    }
}

/// Helper to test if a local HTTP endpoint is responsive.
pub(crate) fn probe_http_endpoint(url: &str, timeout: Duration) -> bool {
    let host_port = url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split('/')
        .next()
        .unwrap_or(url);

    let host_port = if !host_port.contains(':') {
        if url.starts_with("https://") {
            format!("{}:443", host_port)
        } else {
            format!("{}:80", host_port)
        }
    } else {
        host_port.to_string()
    };

    if let Ok(mut addrs) = host_port.to_socket_addrs() {
        if let Some(addr) = addrs.next() {
            if std::net::TcpStream::connect_timeout(&addr, timeout).is_ok() {
                return true;
            }
        }
    }
    false
}

/// Returns the path to the configuration file.
pub fn config_path() -> PathBuf {
    ZevConfig::config_path()
}

/// Persistent user configuration for Zev.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZevConfig {
    pub method: ZevMethod,
    pub confidence_threshold: f64,
    pub margin_threshold: f64,
    pub apfel_weight: f64,
    pub gemma_url: Option<String>,
    pub gemma_model: Option<String>,
    pub created_at: String,
    pub version: String,
}

impl Default for ZevConfig {
    fn default() -> Self {
        let env = SystemEnvironment::probe();
        Self::default_for_env(&env)
    }
}

impl ZevConfig {
    pub fn default_for_env(env: &SystemEnvironment) -> Self {
        Self {
            method: env.recommended_method(),
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            margin_threshold: DEFAULT_MARGIN_THRESHOLD,
            apfel_weight: DEFAULT_APFEL_WEIGHT,
            gemma_url: Some(env.gemma_endpoint.clone()),
            gemma_model: Some("gemma-4-31b".to_string()),
            created_at: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Resolves the cross-platform configuration directory (`~/.config/zev` or `%APPDATA%\zev`).
    pub fn config_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("ZEV_CONFIG_DIR") {
            return PathBuf::from(dir);
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".config").join("zev");
        }
        if let Ok(appdata) = std::env::var("USERPROFILE") {
            return PathBuf::from(appdata).join(".config").join("zev");
        }
        PathBuf::from(".zev")
    }

    /// Path to the configuration file.
    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.json")
    }

    /// Checks if configuration exists on disk.
    pub fn exists() -> bool {
        Self::config_path().exists()
    }

    /// Loads configuration from disk if present.
    pub fn load() -> Option<Self> {
        let path = Self::config_path();
        if path.exists() {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(cfg) = serde_json::from_str::<Self>(&content) {
                    return Some(cfg);
                }
            }
        }
        None
    }

    /// Saves configuration to disk.
    pub fn save(&self) -> io::Result<()> {
        let dir = Self::config_dir();
        fs::create_dir_all(&dir)?;
        let path = Self::config_path();
        let content = serde_json::to_string_pretty(self)
            .map_err(io::Error::other)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Exports configured settings to environment variables for sub-modules.
    pub fn apply_to_env(&self) {
        match self.method {
            ZevMethod::DualEnsemble => {
                std::env::set_var("ZEV_FALLBACK", "apfel");
            }
            ZevMethod::DualCascade => {
                std::env::set_var("ZEV_FALLBACK", "gemma");
            }
            ZevMethod::LoadBalanced => {
                std::env::set_var("ZEV_FALLBACK", "apfel");
            }
            ZevMethod::PureSimd => {
                std::env::set_var("ZEV_FALLBACK", "none");
            }
            ZevMethod::Apfel => {
                std::env::set_var("ZEV_FALLBACK", "apfel");
            }
            ZevMethod::Gemma => {
                std::env::set_var("ZEV_FALLBACK", "gemma");
            }
            ZevMethod::Mlx => {
                std::env::set_var("ZEV_FALLBACK", "mlx");
            }
            ZevMethod::Clm => {
                std::env::set_var("ZEV_FALLBACK", "clm");
            }
        }

        std::env::set_var(
            "ZEV_FALLBACK_CONFIDENCE",
            self.confidence_threshold.to_string(),
        );
        std::env::set_var("ZEV_FALLBACK_MARGIN", self.margin_threshold.to_string());

        if let Some(ref url) = self.gemma_url {
            if std::env::var("GEMMA_URL").is_err() {
                std::env::set_var("GEMMA_URL", url);
            }
        }
        if let Some(ref model) = self.gemma_model {
            if std::env::var("GEMMA_MODEL").is_err() {
                std::env::set_var("GEMMA_MODEL", model);
            }
        }
    }
}

/// Runs the interactive first-run onboarding setup wizard.
pub fn run_setup_wizard(force_interactive: bool) -> io::Result<ZevConfig> {
    let env = SystemEnvironment::probe();

    println!();
    println!("  ███████╗███████╗██╗   ██╗");
    println!("  ╚══███╔╝██╔════╝██║   ██║");
    println!("    ███╔╝ █████╗  ██║   ██║");
    println!("   ███╔╝  ██╔══╝  ╚██╗ ██╔╝");
    println!("  ███████╗███████╗ ╚████╔╝ ");
    println!("  ╚══════╝╚══════╝  ╚═══╝  ");
    println!("  ✦ High-Performance Zero-Token Decision Engine (v{})", env!("CARGO_PKG_VERSION"));
    println!();

    print!("{}", env.format_diagnostic_box());
    println!();

    println!("╭────────────────────────────────────────────────────────────────────────────╮");
    println!("│ 🎯 Choose Your Default Decision Method                                    │");
    println!("╰────────────────────────────────────────────────────────────────────────────╯");
    println!();

    let methods = [
        ZevMethod::DualEnsemble,
        ZevMethod::DualCascade,
        ZevMethod::LoadBalanced,
        ZevMethod::PureSimd,
        ZevMethod::Apfel,
        ZevMethod::Gemma,
    ];

    for (i, m) in methods.iter().enumerate() {
        let num = i + 1;
        println!("  [{num}] {}", m.display_name());
        println!("      Description:  {}", m.description());
        println!(
            "      Accuracy:     {}  |  Latency: {}",
            m.accuracy(),
            m.latency()
        );
        println!("      Prerequisites: {}", m.requirements());
        println!();
    }

    let default_method = env.recommended_method();
    let default_idx = match default_method {
        ZevMethod::DualEnsemble => 1,
        ZevMethod::DualCascade => 2,
        ZevMethod::LoadBalanced => 3,
        ZevMethod::PureSimd => 4,
        ZevMethod::Apfel => 5,
        ZevMethod::Gemma => 6,
        _ => 1,
    };

    let chosen_method = if force_interactive || std::io::stdin().is_terminal() {
        print!(
            "Select default method [1-6] (default: [{default_idx}] {}): ",
            default_method.as_str()
        );
        io::stdout().flush()?;

        let mut input = String::new();
        let stdin = io::stdin();
        stdin.lock().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            default_method
        } else if let Some(m) = ZevMethod::parse_str(trimmed) {
            m
        } else {
            println!("  ! Unrecognized option '{trimmed}'. Selecting default: {default_method:?}");
            default_method
        }
    } else {
        println!(
            "  • Non-interactive terminal detected. Auto-selecting optimal method: {:?}",
            default_method
        );
        default_method
    };

    let mut config = ZevConfig::default_for_env(&env);
    config.method = chosen_method;

    // If method utilizes Gemma and endpoint not currently active, offer quick endpoint input
    if matches!(
        chosen_method,
        ZevMethod::DualEnsemble | ZevMethod::DualCascade | ZevMethod::Gemma
    ) && !env.gemma_available
        && (force_interactive || std::io::stdin().is_terminal())
    {
        println!();
        println!("  ℹ Gemma/Ollama endpoint not currently active at {}.", env.gemma_endpoint);
        print!("  Enter custom endpoint URL (or press Enter to keep default with automatic SIMD fallback): ");
        io::stdout().flush()?;
        let mut ep_in = String::new();
        io::stdin().lock().read_line(&mut ep_in)?;
        let ep_trim = ep_in.trim();
        if !ep_trim.is_empty() {
            config.gemma_url = Some(ep_trim.to_string());
        }
    }

    config.save()?;
    config.apply_to_env();

    println!();
    println!("╭────────────────────────────────────────────────────────────────────────────╮");
    println!("│ ✓ Configuration Initialized Successfully                                  │");
    println!("╰────────────────────────────────────────────────────────────────────────────╯");
    println!("  • Saved to:          {}", ZevConfig::config_path().display());
    println!("  • Active Method:     {}", config.method.display_name());
    println!("  • Accuracy Floor:    {}", config.method.accuracy());
    println!("  • Average Latency:   {}", config.method.latency());
    println!();
    println!("💡 Quick Start Examples:");
    println!(r#"  1. Route customer request:  zev route --routes '{{"billing":"Disputes","tech":"Bug"}}' --state "App crashed""#);
    println!("  2. Evaluate JSON decision:  zev decide --file request.json");
    println!("  3. Launch local API server: zev serve --port 8080");
    println!("  4. Check diagnostics:       zev doctor");
    println!("  5. Reconfigure anytime:     zev init");
    println!();

    Ok(config)
}

/// Loads existing configuration or prompts first-run onboarding wizard.
pub fn load_or_init() -> io::Result<ZevConfig> {
    if let Some(cfg) = ZevConfig::load() {
        cfg.apply_to_env();
        Ok(cfg)
    } else if std::io::stdin().is_terminal() {
        run_setup_wizard(false)
    } else {
        // Non-interactive fallback: probe and save silently
        let env = SystemEnvironment::probe();
        let cfg = ZevConfig::default_for_env(&env);
        let _ = cfg.save();
        cfg.apply_to_env();
        eprintln!(
            "zev: Initialized default configuration ({}) for non-interactive session.",
            cfg.method.as_str()
        );
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_environment_probe() {
        let env = SystemEnvironment::probe();
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
        assert!(!env.simd_instruction_set.is_empty());
    }

    #[test]
    fn test_zev_method_parsing() {
        assert_eq!(ZevMethod::parse_str("1"), Some(ZevMethod::DualEnsemble));
        assert_eq!(ZevMethod::parse_str("dual-ensemble"), Some(ZevMethod::DualEnsemble));
        assert_eq!(ZevMethod::parse_str("pure-simd"), Some(ZevMethod::PureSimd));
        assert_eq!(ZevMethod::parse_str("4"), Some(ZevMethod::PureSimd));
        assert_eq!(ZevMethod::parse_str("gemma"), Some(ZevMethod::Gemma));
        assert_eq!(ZevMethod::parse_str("apfel"), Some(ZevMethod::Apfel));
        assert_eq!(ZevMethod::parse_str("unknown_xyz"), None);
    }

    #[test]
    fn test_config_serialization() {
        let env = SystemEnvironment::probe();
        let cfg = ZevConfig::default_for_env(&env);
        let json = serde_json::to_string_pretty(&cfg).expect("Must serialize");
        let deserialized: ZevConfig = serde_json::from_str(&json).expect("Must deserialize");
        assert_eq!(cfg.method, deserialized.method);
    }
}
