# ♟️ KiyEngine 1.0

KiyEngine is a professional-grade, high-performance UCI chess engine written in Rust. It combines modern bitboard techniques with advanced search algorithms to deliver a powerful and stable competitive experience.

## 🚀 Key Features

- **Magic Bitboards**: Ultra-fast $O(1)$ move generation for sliding pieces (Rooks, Bishops, Queens) using precomputed lookup tables.
- **GM-Level Search**: Optimized Minimax with Alpha-Beta pruning, reaching depths of 15+ in seconds.
- **Deep Tactical Vision**: Advanced heuristics including:
  - **Transposition Table (TT)** with Zobrist Hashing.
  - **Null Move Pruning (NMP)** & Late Move Reductions (LMR).
  - **Killer Move** & History Heuristics for superior move ordering.
  - **Check Extensions** for tactical stability.
- **Professional UCI Protocol**: Full compatibility with standard GUIs (En Croissant, Arena, ChessBase), featuring:
  - Dynamic `Hash` memory management.
  - Periodic progress reporting (NPS, Nodes, Time).
  - Responsive `stop` command support.

## 🛠️ Installation & Build

Ensure you have the [Rust toolchain](https://rustup.rs/) installed.

```bash
# Clone the repository
git clone https://github.com/Kiy-K/KiyEngine.git
cd KiyEngine

# Build the optimized release binary
cargo build --release
```

The compiled binary will be located at `./target/release/kiy_engine`.

## 🖥️ GUI Integration (e.g., En Croissant)

1. Open your favorite Chess GUI.
2. Navigate to **Engines > Add New Engine**.
3. Select **Local Executable** and browse to the `kiy_engine` binary created in the build step.
4. The engine will be identified as **KiyEngine 1.0**.

## 📊 Technical Specs

- **Language**: Rust
- **Architecture**: Bitboards (Magic)
- **Evaluation**: Tapered PeSTo-style PST + Bishop Pair/Material.
- **NPS**: ~3,000,000+ nodes/second (hardware dependent).

## 📄 License

This project is open-source. See the repository for details.
