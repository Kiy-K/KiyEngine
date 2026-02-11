# ===========================================================================
# KiyEngine v6.0.0 Docker Image
#
# Multi-stage build: compile with full toolchain, deploy minimal runtime.
# Produces a lean image (~30 MB base + model) that runs the UCI engine
# on stdin/stdout, suitable for use with cutechess-cli or any UCI GUI.
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM rust:1.82-bookworm AS builder

WORKDIR /build

# Copy dependency manifests first for layer caching
COPY Cargo.toml Cargo.lock* ./
COPY .cargo .cargo

# Create a dummy main to pre-build dependencies
RUN mkdir -p src && \
    echo 'fn main() { println!("stub"); }' > src/main.rs && \
    echo 'pub fn stub() {}' > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Copy full source
COPY src/ src/

# Build the actual engine binary
# Override target-cpu=native with a portable baseline (AVX2 is widely supported)
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN cargo build --release --bin kiy_engine_v5_alpha && \
    strip target/release/kiy_engine_v5_alpha

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /engine

# Copy the compiled binary
COPY --from=builder /build/target/release/kiy_engine_v5_alpha ./kiyengine

# Copy model and book files (if present in the build context)
COPY kiyengine.gguf* ./
COPY book*.bin ./

# Metadata
LABEL maintainer="Khoi"
LABEL description="KiyEngine v6.0.0 -- UCI chess engine with BitNet 1.58-bit Transformer"
LABEL version="6.0.0"

# The engine communicates over stdin/stdout (UCI protocol)
ENTRYPOINT ["./kiyengine"]
