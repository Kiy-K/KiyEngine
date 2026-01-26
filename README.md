# KiyEngine V3

KiyEngine V3 is a high-performance UCI chess engine written in Rust. It combines a Mamba-MoE (Mixture of Experts) Neural Network for evaluation with a classic Alpha-Beta Search enhanced by a Massive Transposition Table.

## 🚀 Features

- **Mamba-MoE Neural Network**: State-of-the-art evaluation function for nuanced and powerful position analysis.
- **Massive Transposition Table**: A huge, 512MB default transposition table to cache search results and minimize NN evaluations.
- **Optimized for Performance**: Built with modern Rust and optimized for speed with a `lto = true`, `codegen-units = 1`, and `panic = 'abort'` release profile.
- **Modern Architecture**: Leverages the `candle` ML framework for its neural network backend.

## 🛠️ Installation & Build

Ensure you have the [Rust toolchain](https://rustup.rs/) installed.

```bash
# Build the optimized release binary
cargo build --release
```

The compiled binary will be located at `./target/release/kiy_engine_v3`.
