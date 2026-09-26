# Security Policy

The `zev-rs` project takes the security of its zero-token decision engine, HTTP server, and model inference backends seriously. This document outlines our security commitment, supported versions, and procedure for reporting vulnerabilities.

---

## Supported Versions

Only the latest active minor release receives active security patches. We recommend all production deployments stay on the latest patch release.

| Version | Supported          |
| ------- | ------------------ |
| `0.3.x` | :white_check_mark: |
| `0.2.x` | :x:                |
| `0.1.x` | :x:                |
| `< 0.1` | :x:                |

---

## Reporting a Vulnerability

**Please DO NOT open public GitHub issues or discussions for suspected security vulnerabilities.** Public disclosure before a fix is available puts users and production services at risk.

### Private Reporting Channels

1. **GitHub Private Vulnerability Reporting (Preferred):**
   Navigate to the [Security Advisory page](https://github.com/bhubbard/zev-rs/security/advisories) on GitHub and select **"Report a vulnerability"**. This creates a confidential workspace where we can collaborate directly on the triage, validation, and fix.

2. **Direct Contact:**
   If GitHub Private Reporting is unavailable, send an encrypted or direct email to `hello@brandonhubbard.com` with the subject tag `[SECURITY: zev-rs]`.

### What to Include in Your Report

To help us investigate and patch quickly, please include:
- A clear description of the vulnerability and attack vector.
- Affected component(s): e.g., HTTP server (`/v1/systemone`), CLI, tokenization encoder, Candle/CUDA kernels, or conversion scripts.
- Step-by-step reproduction steps or a minimal Proof of Concept (PoC).
- Potential impact (e.g., Remote Code Execution, Denial of Service, timing side-channel, memory exhaustion).
- Any proposed mitigations or patch suggestions if you have them.

---

## Response Timeline

We adhere to the following SLA:

- **Initial Acknowledgment:** Within **24 to 48 hours** of receiving your report.
- **Triage & Severity Assessment:** Within **3 business days**.
- **Patch Development & Testing:** Priority based on severity (Critical/High addressed with urgent priority).
- **Public Advisory & Release:** Coordinated disclosure once the patch is released and users have an upgrade path.

---

## Security Architecture & Design Principles

`zev-rs` follows defense-in-depth principles across its codebase:

1. **Memory Safety & Zero Unsafe in Core Logic:**
   - Pure Rust execution by default, preventing buffer overflows, use-after-free, and memory corruption.
   - Any hardware acceleration operations (e.g., NVRTC CUDA custom ops) are strictly bounds-checked and encapsulated within verified custom ops.

2. **Safe Deserialization:**
   - Model weights and checkpoints are loaded via memory-mapped `safetensors`.
   - Native conversion utilities (`src/bin/convert_head.rs`) strictly reject arbitrary pickle classes and validate paths to prevent Remote Code Execution (RCE) and path traversal.

3. **Denial-of-Service (DoS) Defenses:**
   - Strict body size limits enforced on all HTTP ingestion endpoints (`DefaultBodyLimit::max(2MB)`).
   - In-memory inference worker queues are bounded (`mpsc::sync_channel`) with automatic `429 Too Many Requests` backpressure when saturated.

4. **Cryptographic & Auth Timing Resistance:**
   - Bearer token authentication compares API keys using constant-time byte comparisons (`constant_time_eq`) to eliminate timing side-channels.

5. **Prompt & Delimiter Injection Resistance:**
   - The tokenization encoder strictly sanitizes special control tags (`<|...|>`), preventing delimiter collision or jailbreaks in causal prompts.
