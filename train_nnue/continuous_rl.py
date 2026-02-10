#!/usr/bin/env python3
"""
Continuous RL training loop for KiyEngine NNUE.

Alternates between:
  1. Self-play data generation (fast Rust binary, ~4 games/s)
  2. NNUE training (PyTorch, warm-started from current weights)
  3. Engine rebuild (picks up new NNUE file)

Usage:
  python continuous_rl.py [OPTIONS]

Options:
  --iterations N     Number of RL iterations (default: 20)
  --games N          Self-play games per iteration (default: 200)
  --depth D          Search depth for self-play (default: 5)
  --epochs N         Training epochs per iteration (default: 5)
  --batch-size N     Training batch size (default: 4096)
  --lr F             Learning rate (default: 5e-4)
  --nnue PATH        Starting NNUE file (default: kiyengine.nnue)
  --kd-data PATH     Knowledge distillation data to mix in (optional)
  --kd-ratio F       Fraction of KD data to include (default: 0.3)
  --max-buffer N     Max RL positions to keep in buffer (default: 2000000)
"""

import argparse
import math
import os
import random
import struct
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ============================================================================
# CONSTANTS (must match train.py and Rust engine)
# ============================================================================

NUM_FEATURES = 768
QA = 255
QB = 64
SCALE = 400
WEIGHT_CLIP = 1.98
NNUE_MAGIC = b"KYNN"
NNUE_VERSION = 1

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_BIN = os.path.join(ENGINE_DIR, "target", "release", "kiy_engine_v5_alpha")
SELFPLAY_BIN = os.path.join(ENGINE_DIR, "target", "release", "selfplay")

# ============================================================================
# FEATURE ENCODING (duplicated from train.py to keep this self-contained)
# ============================================================================

def white_feature(color: int, piece_type: int, square: int) -> int:
    return color * 384 + piece_type * 64 + square

def black_feature(color: int, piece_type: int, square: int) -> int:
    flipped_color = 1 - color
    flipped_sq = square ^ 56
    return flipped_color * 384 + piece_type * 64 + flipped_sq

def fen_to_features(fen: str):
    parts = fen.split()
    board_str = parts[0]
    stm = 0 if parts[1] == "w" else 1

    white_indices = []
    black_indices = []

    rank = 7
    file = 0

    piece_map = {
        'P': (0, 0), 'N': (0, 1), 'B': (0, 2), 'R': (0, 3), 'Q': (0, 4), 'K': (0, 5),
        'p': (1, 0), 'n': (1, 1), 'b': (1, 2), 'r': (1, 3), 'q': (1, 4), 'k': (1, 5),
    }

    for ch in board_str:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            color, piece_type = piece_map[ch]
            sq = rank * 8 + file
            white_indices.append(white_feature(color, piece_type, sq))
            black_indices.append(black_feature(color, piece_type, sq))
            file += 1

    return white_indices, black_indices, stm

# ============================================================================
# NNUE MODEL
# ============================================================================

class SCReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0).square()

class NNUEModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ft_weight = nn.Parameter(torch.zeros(NUM_FEATURES, hidden_size))
        self.ft_bias = nn.Parameter(torch.zeros(hidden_size))
        self.out_weight = nn.Parameter(torch.zeros(2 * hidden_size))
        self.out_bias = nn.Parameter(torch.zeros(1))
        nn.init.kaiming_normal_(self.ft_weight, nonlinearity='relu')
        nn.init.normal_(self.out_weight, std=0.01)
        self.screlu = SCReLU()

    def forward(self, white_features, black_features, stm):
        batch_size = white_features.size(0)
        white_acc = self._accumulate(white_features)
        black_acc = self._accumulate(black_features)
        white_act = self.screlu(white_acc)
        black_act = self.screlu(black_acc)
        stm_mask = stm.unsqueeze(1).float()
        us = white_act * (1 - stm_mask) + black_act * stm_mask
        them = black_act * (1 - stm_mask) + white_act * stm_mask
        combined = torch.cat([us, them], dim=1)
        output = (combined * self.out_weight).sum(dim=1) + self.out_bias.squeeze()
        return output

    def _accumulate(self, features):
        batch_size = features.size(0)
        acc = self.ft_bias.unsqueeze(0).expand(batch_size, -1).clone()
        valid_mask = features >= 0
        safe_indices = features.clamp(min=0)
        weights = self.ft_weight[safe_indices]
        weights = weights * valid_mask.unsqueeze(2).float()
        acc = acc + weights.sum(dim=1)
        return acc

    def export_nnue(self, path: str):
        with torch.no_grad():
            ft_w = (self.ft_weight.data * QA).round().clamp(-32768, 32767).short()
            ft_b = (self.ft_bias.data * QA).round().clamp(-32768, 32767).short()
            clipped_w = self.out_weight.data.clamp(-WEIGHT_CLIP, WEIGHT_CLIP)
            out_w = (clipped_w * QB).round().clamp(-32768, 32767).short()
            out_b = (self.out_bias.data * QB).round().clamp(-32768, 32767).short()

        with open(path, "wb") as f:
            f.write(NNUE_MAGIC)
            f.write(struct.pack("<I", NNUE_VERSION))
            f.write(struct.pack("<I", self.hidden_size))
            for feat_idx in range(NUM_FEATURES):
                for h in range(self.hidden_size):
                    f.write(struct.pack("<h", int(ft_w[feat_idx, h].item())))
            for h in range(self.hidden_size):
                f.write(struct.pack("<h", int(ft_b[h].item())))
            for h in range(2 * self.hidden_size):
                f.write(struct.pack("<h", int(out_w[h].item())))
            f.write(struct.pack("<h", int(out_b[0].item())))

        size_kb = os.path.getsize(path) / 1024
        print(f"  Exported NNUE to {path} ({size_kb:.1f} KB)")

# ============================================================================
# LOAD NNUE WEIGHTS INTO PYTORCH MODEL
# ============================================================================

def load_nnue_into_model(model: NNUEModel, nnue_path: str):
    """Load quantized NNUE binary weights into a float PyTorch model."""
    with open(nnue_path, "rb") as f:
        magic = f.read(4)
        assert magic == NNUE_MAGIC, f"Invalid magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        hidden = struct.unpack("<I", f.read(4))[0]
        assert hidden == model.hidden_size, f"Hidden mismatch: {hidden} vs {model.hidden_size}"

        # FT weights: quantized by QA
        ft_data = []
        for _ in range(NUM_FEATURES * hidden):
            ft_data.append(struct.unpack("<h", f.read(2))[0])
        ft_w = torch.tensor(ft_data, dtype=torch.float32).reshape(NUM_FEATURES, hidden) / QA
        model.ft_weight.data.copy_(ft_w)

        # FT biases: quantized by QA
        ft_b = []
        for _ in range(hidden):
            ft_b.append(struct.unpack("<h", f.read(2))[0])
        model.ft_bias.data.copy_(torch.tensor(ft_b, dtype=torch.float32) / QA)

        # Output weights: quantized by QB
        out_w = []
        for _ in range(2 * hidden):
            out_w.append(struct.unpack("<h", f.read(2))[0])
        model.out_weight.data.copy_(torch.tensor(out_w, dtype=torch.float32) / QB)

        # Output bias: quantized by QB
        out_b = struct.unpack("<h", f.read(2))[0]
        model.out_bias.data.copy_(torch.tensor([out_b], dtype=torch.float32) / QB)

    print(f"  Loaded NNUE weights from {nnue_path} (hidden={hidden})")

# ============================================================================
# DATASET
# ============================================================================

class BinaryDataset(Dataset):
    """Dataset from binary (fen_len, fen, eval_cp) entries."""
    def __init__(self, path: str, max_active: int = 32):
        self.max_active = max_active
        self.entries = []
        with open(path, "rb") as f:
            while True:
                len_bytes = f.read(2)
                if not len_bytes or len(len_bytes) < 2:
                    break
                fen_len = struct.unpack("<H", len_bytes)[0]
                fen = f.read(fen_len).decode("utf-8")
                eval_bytes = f.read(2)
                if not eval_bytes or len(eval_bytes) < 2:
                    break
                eval_cp = struct.unpack("<h", eval_bytes)[0]
                self.entries.append((fen, eval_cp))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fen, eval_cp = self.entries[idx]
        white_idx, black_idx, stm = fen_to_features(fen)
        white_padded = white_idx + [-1] * (self.max_active - len(white_idx))
        black_padded = black_idx + [-1] * (self.max_active - len(black_idx))
        target = 1.0 / (1.0 + math.exp(-eval_cp / SCALE))
        return (
            torch.tensor(white_padded[:self.max_active], dtype=torch.long),
            torch.tensor(black_padded[:self.max_active], dtype=torch.long),
            torch.tensor(stm, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(eval_cp, dtype=torch.float32),
        )

# ============================================================================
# SELF-PLAY DATA GENERATION (via Rust binary)
# ============================================================================

def run_selfplay(nnue_path: str, output_path: str, num_games: int, depth: int) -> int:
    """Run Rust selfplay binary and return number of positions generated."""
    cmd = [
        SELFPLAY_BIN,
        "--games", str(num_games),
        "--depth", str(depth),
        "--output", output_path,
        "--nnue", nnue_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE_DIR)

    if result.returncode != 0:
        print(f"  ERROR: selfplay failed: {result.stderr}")
        return 0

    # Parse output for position count
    for line in result.stderr.split('\n'):
        if 'Positions:' in line:
            try:
                return int(line.split('Positions:')[1].strip())
            except:
                pass

    # Fallback: estimate from file size
    try:
        size = os.path.getsize(output_path)
        return size // 50  # rough estimate
    except:
        return 0

# ============================================================================
# TRAINING STEP
# ============================================================================

def train_one_iteration(
    model: NNUEModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    rl_data_paths: list,
    kd_data_path: str | None,
    kd_ratio: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Train model on combined KD + RL data for a few epochs. Returns best val loss."""

    # Build combined dataset
    datasets = []
    total_rl = 0
    for path in rl_data_paths:
        if os.path.exists(path):
            ds = BinaryDataset(path)
            if len(ds) > 0:
                datasets.append(ds)
                total_rl += len(ds)

    if kd_data_path and os.path.exists(kd_data_path) and kd_ratio > 0:
        kd_ds = BinaryDataset(kd_data_path)
        # Subsample KD data to maintain ratio
        kd_target = int(total_rl * kd_ratio / (1 - kd_ratio))
        kd_target = min(kd_target, len(kd_ds))
        if kd_target > 0:
            indices = random.sample(range(len(kd_ds)), kd_target)
            kd_subset = torch.utils.data.Subset(kd_ds, indices)
            datasets.append(kd_subset)
            print(f"  Mixed in {kd_target} KD positions ({kd_ratio:.0%} ratio)")

    if not datasets:
        print("  WARNING: No training data available!")
        return float('inf')

    combined = ConcatDataset(datasets)
    total = len(combined)
    print(f"  Training on {total} positions ({total_rl} RL + {total - total_rl} KD)")

    # Split 95/5
    train_size = int(0.95 * total)
    val_size = total - train_size
    train_ds, val_ds = torch.utils.data.random_split(combined, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        for w, b, s, tgt, _ in train_loader:
            w, b, s, tgt = w.to(device), b.to(device), s.to(device), tgt.to(device)
            output = model(w, b, s)
            pred = torch.sigmoid(output / SCALE)
            loss = F.mse_loss(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for w, b, s, tgt, _ in val_loader:
                w, b, s, tgt = w.to(device), b.to(device), s.to(device), tgt.to(device)
                output = model(w, b, s)
                pred = torch.sigmoid(output / SCALE)
                loss = F.mse_loss(pred, tgt)
                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / max(train_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        best_val = min(best_val, avg_val)

        if scheduler:
            scheduler.step()

        print(f"    Epoch {epoch+1}/{epochs}: train={avg_train:.6f} val={avg_val:.6f}")

    return best_val

# ============================================================================
# REBUILD ENGINE
# ============================================================================

def rebuild_engine():
    """Rebuild the engine and selfplay binaries in release mode."""
    print("  Rebuilding engine (cargo build --release)...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        capture_output=True, text=True, cwd=ENGINE_DIR
    )
    if result.returncode != 0:
        print(f"  WARNING: Build failed: {result.stderr[-500:]}")
        return False
    return True

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Continuous RL for KiyEngine NNUE")
    parser.add_argument("--iterations", type=int, default=20, help="RL iterations")
    parser.add_argument("--games", type=int, default=200, help="Self-play games per iter")
    parser.add_argument("--depth", type=int, default=5, help="Search depth for selfplay")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per iter")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--nnue", default="kiyengine.nnue", help="Starting NNUE file")
    parser.add_argument("--kd-data", default=None, help="KD data file to mix in")
    parser.add_argument("--kd-ratio", type=float, default=0.3, help="KD data ratio")
    parser.add_argument("--max-buffer", type=int, default=2000000, help="Max RL buffer size")
    args = parser.parse_args()

    nnue_path = os.path.join(ENGINE_DIR, args.nnue)
    rl_data_dir = os.path.join(ENGINE_DIR, "train_nnue", "data", "rl_iterations")
    os.makedirs(rl_data_dir, exist_ok=True)

    kd_path = None
    if args.kd_data:
        kd_path = os.path.join(ENGINE_DIR, args.kd_data) if not os.path.isabs(args.kd_data) else args.kd_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("KiyEngine Continuous RL Training")
    print("=" * 60)
    print(f"  Iterations: {args.iterations}")
    print(f"  Games/iter: {args.games} (depth {args.depth})")
    print(f"  Epochs/iter: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  NNUE: {nnue_path}")
    print(f"  KD data: {kd_path or 'none'}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load current NNUE to determine hidden size
    with open(nnue_path, "rb") as f:
        f.read(4)  # magic
        f.read(4)  # version
        hidden_size = struct.unpack("<I", f.read(4))[0]
    print(f"  Hidden size: {hidden_size}")

    # Create model and load current weights
    model = NNUEModel(hidden_size=hidden_size).to(device)
    load_nnue_into_model(model, nnue_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Track all RL data files
    rl_data_paths = []

    # Also include any pre-existing RL data
    pre_existing_rl = os.path.join(ENGINE_DIR, "train_nnue", "data", "rl.bin")
    if os.path.exists(pre_existing_rl):
        rl_data_paths.append(pre_existing_rl)
        print(f"  Found pre-existing RL data: {pre_existing_rl}")

    total_start = time.time()
    best_overall_loss = float('inf')

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")

        # 1. Self-play data generation
        rl_output = os.path.join(rl_data_dir, f"rl_iter_{iteration:03d}.bin")
        print(f"\n[1/3] Self-play ({args.games} games, depth {args.depth})...")
        num_pos = run_selfplay(nnue_path, rl_output, args.games, args.depth)
        print(f"  Generated {num_pos} positions")
        rl_data_paths.append(rl_output)

        # Trim old RL data if buffer is too large
        total_rl_positions = 0
        for p in rl_data_paths:
            if os.path.exists(p):
                total_rl_positions += os.path.getsize(p) // 50
        if total_rl_positions > args.max_buffer:
            # Remove oldest RL iteration files (keep recent ones)
            while total_rl_positions > args.max_buffer and len(rl_data_paths) > 1:
                removed = rl_data_paths.pop(0)
                if os.path.exists(removed):
                    total_rl_positions -= os.path.getsize(removed) // 50
                    print(f"  Trimmed old RL data: {removed}")

        # 2. Train
        print(f"\n[2/3] Training ({args.epochs} epochs)...")
        val_loss = train_one_iteration(
            model, optimizer, None,
            rl_data_paths, kd_path, args.kd_ratio,
            args.epochs, args.batch_size, device
        )

        # 3. Export and rebuild
        print(f"\n[3/3] Export and rebuild...")
        if val_loss <= best_overall_loss * 1.05:  # Allow 5% regression
            model.export_nnue(nnue_path)
            best_overall_loss = min(best_overall_loss, val_loss)
            rebuild_engine()
        else:
            print(f"  Skipped export (val_loss {val_loss:.6f} > best {best_overall_loss:.6f} + 5%)")
            # Reload best weights to avoid divergence
            load_nnue_into_model(model, nnue_path)

        iter_elapsed = time.time() - iter_start
        total_elapsed = time.time() - total_start
        eta = (total_elapsed / iteration) * (args.iterations - iteration)

        print(f"\n  Iter {iteration} complete: val_loss={val_loss:.6f} "
              f"best={best_overall_loss:.6f} "
              f"time={iter_elapsed:.0f}s ETA={eta:.0f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Continuous RL complete!")
    print(f"  Iterations: {args.iterations}")
    print(f"  Best val_loss: {best_overall_loss:.6f}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  NNUE: {nnue_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
