# KiyEngine v5.0.0
[![Build and Release](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml)

KiyEngine is a high-performance, open-source UCI chess engine written in Rust. It features a BitNet 1.58-bit Transformer architecture for evaluation and a highly optimized tactical search engine with Lazy SMP.

## What's New in V5

- **GGUF Format Support**: V5 introduces GGUF (GGML Universal File) format for faster loading and better quantization support
- **Memory-Mapped Loading**: Near-instant model loading via mmap
- **Backward Compatible**: Still supports legacy v4 `.safetensors` format
- **Auto-Format Detection**: Automatically detects and loads either format

See [V5_MIGRATION.md](V5_MIGRATION.md) for migration guide from v4.

---

## Documentation

For detailed documentation including architecture overview, module reference, and development guidelines, see **[DOCUMENTATION.md](DOCUMENTATION.md)**.

---

## Key Features
- BitNet 1.58-bit quantized transformer architecture for evaluation
- Efficient tactical search with Quiescence Search, Principal Variation Search, and aspiration windows
- Lazy SMP support where helper threads populate the Transposition Table
- Hybrid move ordering combining TT moves, policy suggestions, MVV-LVA captures, killer/history heuristics
- **GGUF format** with 4x smaller model size (~6MB vs ~24MB)
- **Memory-mapped loading** for instant startup

---

## Installation & Build

Build requires the Rust toolchain (rustup).

```bash
git clone https://github.com/Kiy-K/KiyEngine.git
cd KiyEngine
cargo build --release
```

The release binary is available at `./target/release/kiy_engine_v4_omega`.

### Converting v4 Models to V5 (GGUF)

```bash
python scripts/convert_to_gguf.py \
    --input kiyengine.safetensors \
    --output kiyengine.gguf
```

---

## UCI Usage

KiyEngine implements the UCI protocol and can be used with any UCI-compatible GUI.

Configuration options:
- `Hash` (spin) — default: 512 MB — Transposition Table size in megabytes
- `Threads` (spin) — default: 1 — Number of search threads
- `Move Overhead` (spin) — default: 30 ms — Time buffer to account for latency

Example:
```
uci
setoption name Hash value 1024
isready
position startpos
go depth 25
```

---

## Model Files

V5 uses GGUF format by default (`kiyengine.gguf`). The engine automatically:
1. Tries to load `kiyengine.gguf` (V5 format)
2. Falls back to `kiyengine.safetensors` (v4 format) if not found

To inspect a GGUF model:
```bash
cargo run --bin inspect_gguf -- kiyengine.gguf
```

---

## Troubleshooting
- If weights are missing, place `kiyengine.gguf` or `kiyengine.safetensors` in the executable directory
- Use the `release` build for best performance
- Reduce `Hash` if you encounter out-of-memory issues
- See [V5_MIGRATION.md](V5_MIGRATION.md) for migration issues

---

## License

KiyEngine is licensed under the Apache License 2.0. See `LICENSE` for details.

© 2026 Khoi
