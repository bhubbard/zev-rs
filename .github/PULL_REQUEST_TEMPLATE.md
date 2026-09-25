## Description

Please include a summary of the change, the motivation behind it, and any relevant context or design decisions.

Fixes / Closes # (if applicable)

---

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] ⚡ Performance improvement
- [ ] 🛡️ Security hardening / vulnerability fix
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📚 Documentation update / examples
- [ ] 🧹 Refactoring / code hygiene (no functional changes)

---

## Verification & Testing

Please describe how you verified your changes:

- [ ] Ran standard test suite: `cargo test`
- [ ] Ran Candle backend tests (if touching inference/model): `cargo test --features candle`
- [ ] Checked formatting: `cargo fmt --check`
- [ ] Checked linter: `cargo clippy -- -D warnings` (and `--features candle`)
- [ ] Added new unit or integration tests covering the change

---

## Checklist

- [ ] My code adheres to the coding style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated relevant documentation where appropriate
- [ ] My changes generate no new compiler or clippy warnings
- [ ] Any dependent changes have been merged and published in downstream modules
