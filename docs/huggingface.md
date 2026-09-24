# Publishing Zev to Hugging Face: Complete Deployment Guide

This guide details how to publish and maintain **Zev** on Hugging Face as:
1. An interactive **Hugging Face Space** (Docker-based live microsecond playground and public API).
2. A **Hugging Face Dataset** (benchmark suite for decision engines and order-invariance testing).

---

## 1. Deploying the Hugging Face Space

The repository includes a ready-to-deploy Hugging Face Space configuration located in `hf-space/`. It runs a multi-stage Docker build that compiles `zev-rs` with the embedded interactive UI and REST API on port `7860`.

### Step-by-Step Deployment:

1. **Create a new Space on Hugging Face**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - **Space Name**: `zev` or `zev-rs`
   - **License**: MIT
   - **Space SDK**: Select **Docker** (Blank)
   - **Space Hardware**: Free CPU (Zev uses 0 MB model weights and 5.8 µs latency, so the free tier is blazingly fast).

2. **Push the Space to Hugging Face**:
   From your local terminal:
   ```bash
   # Add your Hugging Face Space as a git remote
   git remote add hf-space https://huggingface.co/spaces/bkhubbard/zev

   # Push using git subtree from the zev-rs repository
   # Or create a dedicated branch/repo containing the Space files:
   git subtree push --prefix hf-space hf-space main
   ```

   *Alternatively, using a fresh clone of the Space repo:*
   ```bash
   git clone https://huggingface.co/spaces/bkhubbard/zev /tmp/zev-space
   cp -r hf-space/* /tmp/zev-space/
   cp -r assets src Cargo.toml Cargo.lock README.md /tmp/zev-space/
   cd /tmp/zev-space
   git add .
   git commit -m "feat: Deploy Zev interactive zero-token playground and API"
   git push origin main
   ```

3. **Verify the Running Space**:
   Once built, your Space will be live at:
   `https://huggingface.co/spaces/bkhubbard/zev`
   
   It exposes:
   - Interactive Playground with order-invariance verifier, live latency gauge, and presets.
   - Public API endpoints: `/v1/decisions`, `/v1/tev1`, `/health`, `/v1/models`.

---

## 2. Publishing the Benchmark Dataset

The benchmark suite in `datasets/zev_benchmarks/` contains 1,200 structured test cases across intent routing, clinical differentials, infrastructure SRE triage, guardrail abstention, and order-invariance permutations.

### Step-by-Step Dataset Upload:

1. **Log in to Hugging Face**:
   ```bash
   hf auth login
   ```

2. **Upload using the `hf` CLI**:
   ```bash
   # Upload the benchmark directory directly (creates repo if not existing)
   hf upload bkhubbard/zev-benchmarks ./datasets/zev_benchmarks . --repo-type=dataset
   ```

3. **Verify Dataset Card**:
   Once uploaded, your dataset is live at:
   `https://huggingface.co/datasets/bkhubbard/zev-benchmarks`
   The uploaded `README.md` automatically sets up the Hugging Face dataset viewer with train/test splits, feature schemas, and interactive previews.

---

## 3. Regenerating or Customizing the Benchmark Dataset

To generate more test items or modify the task distribution:

```bash
# Generate 2,500 items into the dataset directory
python3 scripts/export_hf_dataset.py --output-dir datasets/zev_benchmarks --count 2500
```
