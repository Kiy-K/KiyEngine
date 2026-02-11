# KiyEngine NNUE Training

Train an NNUE evaluation network for KiyEngine using the [Bullet](https://github.com/jw1912/bullet) framework with Lichess broadcast data.

## Architecture

```
(768 → 512)×2 → SCReLU → 1024 → 1 (×8 output buckets)
```

- **768 inputs**: Chess768 (64 squares × 6 pieces × 2 colors)
- **512 hidden**: Dual perspective (STM + NSTM accumulators)
- **SCReLU**: Squared Clipped ReLU activation
- **1024 → 1**: Output layer after perspective concatenation
- **8 buckets**: MaterialCount8 output bucketing
- **Quantization**: i16 × 255 (feature layer), i16 × 64 (output layer)

## Data Source

[Lichess Broadcasting Database](https://database.lichess.org/) — free, open-source, high-quality GM/IM games from official broadcasts (FIDE events, super-tournaments, etc.). ~900K+ games available.

### Why Broadcast Data?

- **High quality**: Games played by GMs/IMs in official tournaments
- **Free & open**: No license restrictions, updated monthly
- **Well-annotated**: Clean PGN with proper results
- **Diverse**: Covers all openings and game phases

## Quick Start (Lightning.AI L4)

```bash
# 1. Clone the repo
git clone https://github.com/Kiy-K/KiyEngine.git
cd KiyEngine/training

# 2. Setup environment (installs Rust, CUDA, Python deps, Bullet)
chmod +x scripts/*.sh
./scripts/setup_lightning.sh

# 3. Download data & convert to bulletformat
./scripts/download_and_convert.sh

# 4. Train!
./scripts/train.sh --epochs 400 --lr 0.001
```

## Manual Steps

### 1. Prepare Data

```bash
# Install Python deps
pip install python-chess zstandard requests tqdm

# Download specific months
python scripts/prepare_data.py --download 2025-01 --output data/train.txt --min-elo 2400

# Or download all available broadcasts
python scripts/prepare_data.py --download-all --output data/train.txt

# Or use a local PGN file
python scripts/prepare_data.py --input /path/to/games.pgn.zst --output data/train.txt
```

### 2. Convert to Bulletformat

```bash
# Build bullet-utils (one time)
git clone https://github.com/jw1912/bullet.git ~/bullet
cd ~/bullet && cargo build --release --package bullet-utils

# Convert text → bulletformat
~/bullet/target/release/bullet-utils text-to-bullet data/train.txt data/train.bullet
```

### 3. Train

```bash
# Build trainer
cargo build --release

# Run training
./target/release/train \
    --data data/train.bullet \
    --test data/test.bullet \
    --id kiyengine-nnue-v1 \
    --epochs 400 \
    --lr 0.001 \
    --batch-size 16384
```

### 4. Use in KiyEngine

Copy the quantized checkpoint to the engine directory:

```bash
cp checkpoints/kiyengine-nnue-v1_epoch400.bin ../kiyengine.nnue
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 400 | Number of superbatches |
| `--lr` | 0.001 | Initial learning rate (AdamW) |
| `--batch-size` | 16384 | Positions per batch |
| `--superbatch-size` | 6104 | Batches per superbatch (~100M positions) |
| `--min-elo` | 2400 | Minimum player ELO filter |
| `--id` | kiyengine-nnue | Network identifier |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA GPU | NVIDIA L4 (24GB) |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB (for all broadcasts) |
| Time | ~2h (L4, 100M pos) | ~8h (L4, full data) |

## Advanced: Re-scoring with Stockfish

For stronger networks, re-score positions with Stockfish instead of using WDL-only:

```bash
# Install Stockfish
sudo apt install stockfish

# Re-score (add to prepare_data.py or use separate script)
# This replaces score=0 with Stockfish evaluation at depth 8-12
# Dramatically improves network quality but takes much longer
```

## File Structure

```
training/
├── Cargo.toml              # Rust project (bullet_lib dependency)
├── src/
│   └── main.rs             # Bullet NNUE trainer
├── scripts/
│   ├── prepare_data.py     # Lichess PGN → text format
│   ├── setup_lightning.sh  # Lightning.AI environment setup
│   ├── download_and_convert.sh  # Full data pipeline
│   └── train.sh            # Training launcher
├── data/                   # Training data (gitignored)
├── checkpoints/            # Saved networks (gitignored)
└── README.md
```
