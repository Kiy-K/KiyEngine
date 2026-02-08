# Contributing to KiyEngine

Thank you for your interest in contributing to KiyEngine! This document provides guidelines and standards for contributing.

## Development Setup

1. **Prerequisites**
   - Rust 1.70+ (install via [rustup](https://rustup.rs/))
   - Git

2. **Clone and Build**
   ```bash
   git clone https://github.com/Kiy-K/KiyEngine.git
   cd KiyEngine
   cargo build --release
   ```

## Code Standards

### Rust Style
- Follow standard Rust formatting (`cargo fmt`)
- Use `cargo clippy` to catch common issues
- All code must compile without warnings
- Document public APIs with doc comments

### Module Organization
```
src/
├── lib.rs               # Library exports
├── main.rs              # Binary entry point
├── constants.rs         # Global constants
├── bin/                 # Additional binaries (inspect_gguf, chess_bench)
├── book/                # Polyglot opening book
│   ├── mod.rs           # Book probing & Zobrist hashing
│   └── polyglot_keys.rs # Official 781-key Random64 array
├── engine/              # Neural network inference (GGUF/safetensors)
├── eval/                # Piece-square tables
├── search/
│   ├── mod.rs           # Alpha-beta, lock-free TT, move ordering, quiescence
│   ├── eval.rs          # Material + PST static evaluation
│   ├── time.rs          # Time management
│   ├── lazy_smp.rs      # Lazy SMP threading
│   ├── mcts*.rs         # MCTS implementations (experimental)
│   └── tt.rs            # Atomic TT (alternative implementation)
├── simd/                # AVX2/SIMD optimizations
└── uci/                 # UCI protocol handler
```

### Testing
- Add tests for new functionality
- Run `cargo test` before submitting PRs
- Benchmark critical paths with `cargo bench`

## Submitting Changes

1. Create a feature branch
2. Make focused, atomic commits
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request

## Architecture Guidelines

### Search Algorithms
- Alpha-Beta: `src/search/lazy_smp.rs`
- MCTS: `src/search/mcts_arena.rs` (preferred)
- Keep search logic separate from UCI/protocol

### Neural Network
- Use BitNet 1.58-bit quantization
- Batch inference for efficiency
- SIMD optimizations where applicable

## Contact

- Open an issue for bug reports
- Start a discussion for feature requests
- Join our community Discord (link in README)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
