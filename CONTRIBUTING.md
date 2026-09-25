# Contributing to zev-rs

Thank you for your interest in contributing to `zev-rs`! We welcome contributions ranging from bug reports and documentation fixes to performance optimizations and new backend architectures.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand our community standards.

---

## Development Setup

`zev-rs` is built with modern Rust (Edition 2021). You will need:
- Rust toolchain (`stable`, version 1.79+ recommended)
- `cargo`, `rustfmt`, and `clippy`

### Clone and Build

```bash
git clone https://github.com/bhubbard/zev-rs.git
cd zev-rs

# Build the core library and CLI
cargo build

# Build with HTTP server support (default feature)
cargo build --features server

# Build with Candle / CUDA backend support
cargo build --features candle
cargo build --features cuda # Requires NVIDIA CUDA toolkit and nvrtc
```

---

## Running Tests

`zev-rs` includes comprehensive unit tests, integration tests, and 2,000 synthetic scale test cases. Always verify that tests pass before opening a pull request.

```bash
# Run unit and integration tests (2,000+ tests)
cargo test

# Run tests with Candle backend features enabled
cargo test --features candle

# Run tests with all compiler warnings treated as errors
cargo test -- --nocapture
```

---

## Code Style & Standards

We enforce strict formatting and linting standards across the codebase.

1. **Formatting:**
   ```bash
   cargo fmt --check
   # To fix formatting automatically:
   cargo fmt
   ```

2. **Clippy Linters:**
   We require clean clippy passes with zero warnings:
   ```bash
   cargo clippy -- -D warnings
   cargo clippy --features candle -- -D warnings
   ```

3. **Error Handling & Reliability:**
   - **No panics in server paths:** Never call `.unwrap()` or `.expect()` inside request handlers, background workers, or inference threads. Propagate errors using `Result` or return appropriate HTTP status codes.
   - **Bounds Checking:** Always validate slice and tensor indices before slicing.
   - **Type Safety:** Use strong typing and avoid unbounded memory allocations.

4. **Conventional Commits:**
   Please format your commit messages following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(...)`: New feature or capability
   - `fix(...)`: Bug fix or security remediation
   - `perf(...)`: Performance optimization
   - `refactor(...)`: Code improvement without behavioral change
   - `test(...)`: Adding or updating test cases
   - `docs(...)`: Documentation updates

---

## Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes:** Keep changes focused and self-contained. Add unit tests covering any new logic or bug reproduction.
3. **Verify:**
   - [ ] `cargo fmt --check` passes cleanly.
   - [ ] `cargo clippy -- -D warnings` and `cargo clippy --features candle -- -D warnings` pass with zero warnings.
   - [ ] `cargo test` passes.
4. **Submit PR:**
   - Fill out the PR template completely.
   - Link related issues (e.g. `Closes #12`).
   - Enable "Allow edits by maintainers" so maintainers can assist with minor fixes or rebasing.

---

## Reporting Issues

- **Bugs & Feature Requests:** Please use the [GitHub Issue Forms](https://github.com/bhubbard/zev-rs/issues/new/choose).
- **Security Vulnerabilities:** Follow the private reporting instructions in [SECURITY.md](SECURITY.md). Do **not** open public issues for security vulnerabilities.
