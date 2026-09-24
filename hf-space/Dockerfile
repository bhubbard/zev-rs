# Multi-stage Docker build for Hugging Face Spaces (Port 7860)
FROM rust:1.80-slim-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY assets/ ./assets/

# Build optimized release binary
RUN cargo build --release --bin zev --features server

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /home/appuser/app
COPY --from=builder /app/target/release/zev /usr/local/bin/zev

# Copy documentation and metadata
COPY README.md ./

USER appuser

EXPOSE 7860

ENV HOST=0.0.0.0
ENV PORT=7860

ENTRYPOINT ["zev", "serve", "--host", "0.0.0.0", "--port", "7860"]
