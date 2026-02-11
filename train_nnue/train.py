#!/usr/bin/env python3
"""
KiyEngine NNUE Training Script

Trains a small NNUE network via:
  1. Knowledge distillation from v5.2 KiyNet transformer model
  2. Self-play reinforcement learning (optional phase 2)

Architecture: (768 → HIDDEN)×2 → SCReLU → 2*HIDDEN → 1
  - Input: 768 binary features (6 pieces × 2 colors × 64 squares)
  - Feature transformer: 768 → HIDDEN per perspective (float32 during training)
  - SCReLU activation: clamp(x, 0, 1)² (quantized to [0, QA] at export)
  - Output: dot product → single scalar eval

Usage:
  # Phase 1: Generate training data from v5.2 model
  python train.py generate --model kiyengine.gguf --output data/train.bin --positions 2000000

  # Phase 2: Train NNUE from generated data
  python train.py train --data data/train.bin --output kiyengine.nnue --epochs 50

  # Phase 3: Self-play RL refinement (optional)
  python train.py selfplay --nnue kiyengine.nnue --games 10000 --output kiyengine_rl.nnue
"""

import argparse
import struct
import os
import sys
import random
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONSTANTS
# ============================================================================

NUM_FEATURES = 768       # 6 pieces × 2 colors × 64 squares
HIDDEN_SIZE = 512        # Feature transformer hidden size
QA = 255                 # Accumulator quantization factor
QB = 64                  # Output weight quantization factor (for Lizard SCReLU SIMD)
SCALE = 400              # Sigmoid scale: sigmoid(eval / SCALE) ≈ win_prob
WEIGHT_CLIP = 1.98       # Clip output weights so v*w fits in i16: 255*round(1.98*64)=32385<32767
NNUE_MAGIC = b"KYNN"
NNUE_VERSION = 1

# Piece mapping
PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
PIECE_IDX = {p: i for i, p in enumerate(PIECES)}


# ============================================================================
# FEATURE ENCODING
# ============================================================================

def white_feature(color: int, piece_type: int, square: int) -> int:
    """White perspective feature index. color: 0=white, 1=black."""
    return color * 384 + piece_type * 64 + square


def black_feature(color: int, piece_type: int, square: int) -> int:
    """Black perspective feature index (mirror colors + flip ranks)."""
    flipped_color = 1 - color
    flipped_sq = square ^ 56  # Flip rank
    return flipped_color * 384 + piece_type * 64 + flipped_sq


def fen_to_features(fen: str):
    """
    Convert FEN string to (white_features, black_features, side_to_move).

    Returns:
        white_indices: list of active feature indices for white perspective
        black_indices: list of active feature indices for black perspective
        stm: 0 for white, 1 for black
    """
    parts = fen.split()
    board_str = parts[0]
    stm = 0 if parts[1] == "w" else 1

    white_indices = []
    black_indices = []

    rank = 7  # FEN starts from rank 8 (index 7)
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
# NNUE MODEL (PyTorch)
# ============================================================================

class SCReLU(nn.Module):
    """Squared Clipped ReLU: clamp(x, 0, 1)²"""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0).square()


class NNUEModel(nn.Module):
    """
    NNUE model: (768 → HIDDEN)×2 → SCReLU → 2*HIDDEN → 1

    During training, weights are float32. At export, quantized to i16.
    """
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size

        # Feature transformer (shared weights, separate accumulators per perspective)
        self.ft_weight = nn.Parameter(torch.zeros(NUM_FEATURES, hidden_size))
        self.ft_bias = nn.Parameter(torch.zeros(hidden_size))

        # Output layer
        self.out_weight = nn.Parameter(torch.zeros(2 * hidden_size))
        self.out_bias = nn.Parameter(torch.zeros(1))

        # Initialize
        nn.init.kaiming_normal_(self.ft_weight, nonlinearity='relu')
        nn.init.zeros_(self.ft_bias)
        nn.init.normal_(self.out_weight, std=0.01)
        nn.init.zeros_(self.out_bias)

        self.screlu = SCReLU()

    def forward(self, white_features, black_features, stm):
        """
        Forward pass.

        Args:
            white_features: sparse feature indices [batch, max_active] (padded with -1)
            black_features: sparse feature indices [batch, max_active] (padded with -1)
            stm: side to move [batch] (0=white, 1=black)

        Returns:
            eval: [batch] scalar evaluation
        """
        batch_size = white_features.size(0)

        # Compute accumulators by gathering and summing feature columns
        white_acc = self._accumulate(white_features)  # [batch, hidden]
        black_acc = self._accumulate(black_features)   # [batch, hidden]

        # Apply SCReLU
        white_act = self.screlu(white_acc)  # [batch, hidden]
        black_act = self.screlu(black_acc)  # [batch, hidden]

        # Concatenate based on side to move: [us, them]
        stm_mask = stm.unsqueeze(1).float()  # [batch, 1]

        # When stm=0 (white): us=white, them=black
        # When stm=1 (black): us=black, them=white
        us = white_act * (1 - stm_mask) + black_act * stm_mask
        them = black_act * (1 - stm_mask) + white_act * stm_mask

        # Concatenate and dot with output weights
        combined = torch.cat([us, them], dim=1)  # [batch, 2*hidden]
        output = (combined * self.out_weight).sum(dim=1) + self.out_bias.squeeze()

        return output

    def _accumulate(self, features):
        """Accumulate feature columns for sparse input."""
        batch_size = features.size(0)
        acc = self.ft_bias.unsqueeze(0).expand(batch_size, -1).clone()

        # Mask out padding (-1)
        valid_mask = features >= 0  # [batch, max_active]

        # Clamp indices to valid range for gather (padding positions won't contribute)
        safe_indices = features.clamp(min=0)  # [batch, max_active]

        # Gather weights: [batch, max_active, hidden]
        weights = self.ft_weight[safe_indices]  # Advanced indexing

        # Zero out invalid positions
        weights = weights * valid_mask.unsqueeze(2).float()

        # Sum over active features
        acc = acc + weights.sum(dim=1)

        return acc

    def export_nnue(self, path: str):
        """Export quantized weights to binary NNUE format."""
        with torch.no_grad():
            # Quantize feature transformer: float → i16 (scale by QA)
            ft_w = (self.ft_weight.data * QA).round().clamp(-32768, 32767).short()
            ft_b = (self.ft_bias.data * QA).round().clamp(-32768, 32767).short()

            # Quantize output weights: scale by QB with clipping for SIMD safety
            clipped_w = self.out_weight.data.clamp(-WEIGHT_CLIP, WEIGHT_CLIP)
            out_w = (clipped_w * QB).round().clamp(-32768, 32767).short()
            out_b = (self.out_bias.data * QB).round().clamp(-32768, 32767).short()

        with open(path, "wb") as f:
            f.write(NNUE_MAGIC)
            f.write(struct.pack("<I", NNUE_VERSION))
            f.write(struct.pack("<I", self.hidden_size))

            # Feature transformer weights: [768, hidden] → row-major i16
            for feat_idx in range(NUM_FEATURES):
                for h in range(self.hidden_size):
                    f.write(struct.pack("<h", int(ft_w[feat_idx, h].item())))

            # Feature transformer biases
            for h in range(self.hidden_size):
                f.write(struct.pack("<h", int(ft_b[h].item())))

            # Output weights
            for h in range(2 * self.hidden_size):
                f.write(struct.pack("<h", int(out_w[h].item())))

            # Output bias
            f.write(struct.pack("<h", int(out_b[0].item())))

        file_size = os.path.getsize(path)
        print(f"Exported NNUE to {path} ({file_size / 1024:.1f} KB)")


# ============================================================================
# TRAINING DATASET
# ============================================================================

class NNUEDataset(Dataset):
    """
    Dataset of (FEN, eval) pairs stored in a simple binary format.

    Format per entry:
      - fen_len: u16
      - fen: utf8 bytes
      - eval_cp: i16 (centipawns, clamped to [-10000, 10000])
    """
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

        print(f"Loaded {len(self.entries)} positions from {path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fen, eval_cp = self.entries[idx]
        white_idx, black_idx, stm = fen_to_features(fen)

        # Pad to max_active with -1
        white_padded = white_idx + [-1] * (self.max_active - len(white_idx))
        black_padded = black_idx + [-1] * (self.max_active - len(black_idx))

        # Target: sigmoid(eval / SCALE) for WDL-like loss
        target = 1.0 / (1.0 + math.exp(-eval_cp / SCALE))

        return (
            torch.tensor(white_padded[:self.max_active], dtype=torch.long),
            torch.tensor(black_padded[:self.max_active], dtype=torch.long),
            torch.tensor(stm, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(eval_cp, dtype=torch.float32),
        )


# ============================================================================
# DATA GENERATION (from v5.2 model via subprocess)
# ============================================================================

def generate_random_fen():
    """Generate a random legal FEN by making random moves from the starting position."""
    import chess
    board = chess.Board()
    num_moves = random.randint(4, 80)

    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        board.push(random.choice(legal_moves))

    return board.fen() if not board.is_game_over() else None


def generate_training_data(model_path: str, output_path: str, num_positions: int):
    """
    Generate training data by evaluating random positions with the v5.2 model.

    This creates positions via random playout and labels them with the v5.2
    model's value head output (converted to centipawns).
    """
    try:
        import chess
    except ImportError:
        print("ERROR: python-chess is required. Install with: pip install chess")
        sys.exit(1)

    print(f"Generating {num_positions} training positions...")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    positions = []
    attempts = 0
    while len(positions) < num_positions and attempts < num_positions * 3:
        fen = generate_random_fen()
        if fen:
            positions.append(fen)
        attempts += 1

        if len(positions) % 10000 == 0:
            print(f"  Generated {len(positions)}/{num_positions} positions...")

    print(f"Generated {len(positions)} valid positions")

    # For now, use random evals as placeholder (real labels come from v5.2 inference)
    # TODO: Integrate with KiyEngine v5.2 model for actual evaluation
    print("WARNING: Using random evaluations as placeholder.")
    print("For real training, run the Rust label_positions binary to get v5.2 evals.")

    with open(output_path, "wb") as f:
        for fen in positions:
            fen_bytes = fen.encode("utf-8")
            # Placeholder eval: small random value centered at 0
            eval_cp = random.randint(-200, 200)
            f.write(struct.pack("<H", len(fen_bytes)))
            f.write(fen_bytes)
            f.write(struct.pack("<h", max(-10000, min(10000, eval_cp))))

    file_size = os.path.getsize(output_path)
    print(f"Saved {len(positions)} entries to {output_path} ({file_size / 1024 / 1024:.1f} MB)")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_nnue(data_path: str, output_path: str, epochs: int = 50,
               batch_size: int = 16384, lr: float = 0.001, hidden_size: int = HIDDEN_SIZE):
    """Train NNUE from labeled position data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Architecture: (768 → {hidden_size})×2 → SCReLU → {2*hidden_size} → 1")

    dataset = NNUEDataset(data_path)
    if len(dataset) == 0:
        print("ERROR: No training data found.")
        return

    # Split: 95% train, 5% validation
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=use_cuda)

    model = NNUEModel(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss: BCE on sigmoid(output) vs sigmoid(target_cp)
    # This is the standard NNUE training loss
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for white_f, black_f, stm, target, eval_cp in train_loader:
            white_f = white_f.to(device)
            black_f = black_f.to(device)
            stm = stm.to(device)
            target = target.to(device)

            output = model(white_f, black_f, stm)

            # Sigmoid loss: compare sigmoid(output/SCALE) with target (already sigmoid'd)
            pred = torch.sigmoid(output / SCALE)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for white_f, black_f, stm, target, eval_cp in val_loader:
                white_f = white_f.to(device)
                black_f = black_f.to(device)
                stm = stm.to(device)
                target = target.to(device)

                output = model(white_f, black_f, stm)
                pred = torch.sigmoid(output / SCALE)
                loss = F.mse_loss(pred, target)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{epochs}: "
              f"train_loss={avg_train_loss:.6f} "
              f"val_loss={avg_val_loss:.6f} "
              f"lr={lr_now:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.export_nnue(output_path)
            print(f"  → Saved best model (val_loss={best_val_loss:.6f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")
    print(f"NNUE exported to: {output_path}")


# ============================================================================
# SELF-PLAY RL (Phase 2)
# ============================================================================

def selfplay_rl(nnue_path: str, num_games: int, output_path: str):
    """
    Refine NNUE via self-play reinforcement learning.

    Uses the current NNUE for evaluation during self-play games,
    then updates weights based on game outcomes (WDL).
    """
    try:
        import chess
    except ImportError:
        print("ERROR: python-chess is required. Install with: pip install chess")
        sys.exit(1)

    print(f"Self-play RL: {num_games} games")
    print(f"Starting NNUE: {nnue_path}")
    print(f"Output: {output_path}")

    # Load existing NNUE weights into PyTorch model
    model = NNUEModel(hidden_size=HIDDEN_SIZE)

    # Read binary weights
    with open(nnue_path, "rb") as f:
        magic = f.read(4)
        assert magic == NNUE_MAGIC, f"Invalid NNUE magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        hidden = struct.unpack("<I", f.read(4))[0]
        assert hidden == HIDDEN_SIZE, f"Hidden size mismatch: {hidden} vs {HIDDEN_SIZE}"

        # Load FT weights
        ft_data = []
        for _ in range(NUM_FEATURES * hidden):
            ft_data.append(struct.unpack("<h", f.read(2))[0])
        ft_w = torch.tensor(ft_data, dtype=torch.float32).reshape(NUM_FEATURES, hidden) / QA
        model.ft_weight.data.copy_(ft_w)

        # Load FT biases
        ft_b = []
        for _ in range(hidden):
            ft_b.append(struct.unpack("<h", f.read(2))[0])
        model.ft_bias.data.copy_(torch.tensor(ft_b, dtype=torch.float32) / QA)

        # Load output weights
        out_w = []
        for _ in range(2 * hidden):
            out_w.append(struct.unpack("<h", f.read(2))[0])
        model.out_weight.data.copy_(torch.tensor(out_w, dtype=torch.float32) / QB)

        # Load output bias
        out_b = struct.unpack("<h", f.read(2))[0]
        model.out_bias.data.copy_(torch.tensor([out_b], dtype=torch.float32) / QB)

    print("Loaded NNUE weights into PyTorch model")

    # Self-play loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    game_positions = []  # Collect (fen, game_result) tuples

    for game_idx in range(num_games):
        board = chess.Board()
        positions = []

        # Play a game using NNUE eval for move selection (simple 1-ply greedy)
        while not board.is_game_over() and board.fullmove_number < 200:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Evaluate all moves, pick best
            best_move = None
            best_eval = float("-inf")

            for mv in legal_moves:
                board.push(mv)
                fen = board.fen()
                w_idx, b_idx, stm = fen_to_features(fen)

                # Quick eval
                w_t = torch.tensor([w_idx + [-1] * (32 - len(w_idx))], dtype=torch.long)
                b_t = torch.tensor([b_idx + [-1] * (32 - len(b_idx))], dtype=torch.long)
                s_t = torch.tensor([stm], dtype=torch.long)

                with torch.no_grad():
                    ev = -model(w_t, b_t, s_t).item()  # Negamax

                if ev > best_eval:
                    best_eval = ev
                    best_move = mv

                board.pop()

            positions.append(board.fen())
            board.push(best_move)

        # Determine game result
        result = board.result()
        if result == "1-0":
            wdl = 1.0
        elif result == "0-1":
            wdl = 0.0
        else:
            wdl = 0.5

        # Add positions with result
        for fen in positions:
            _, _, stm = fen_to_features(fen)
            # From white's perspective: wdl as-is
            # Training target: adjust for side to move
            target = wdl if stm == 0 else 1.0 - wdl
            game_positions.append((fen, target))

        if (game_idx + 1) % 100 == 0:
            print(f"  Game {game_idx + 1}/{num_games}: "
                  f"result={result}, positions={len(game_positions)}")

        # Train on accumulated positions every 500 games
        if len(game_positions) >= 50000 or game_idx == num_games - 1:
            print(f"  Training on {len(game_positions)} positions...")
            random.shuffle(game_positions)

            model.train()
            batch_size = 4096
            total_loss = 0.0
            batches = 0

            for start in range(0, len(game_positions), batch_size):
                batch = game_positions[start:start + batch_size]
                w_batch, b_batch, s_batch, t_batch = [], [], [], []

                for fen, target in batch:
                    w_idx, b_idx, stm = fen_to_features(fen)
                    w_batch.append(w_idx + [-1] * (32 - len(w_idx)))
                    b_batch.append(b_idx + [-1] * (32 - len(b_idx)))
                    s_batch.append(stm)
                    t_batch.append(target)

                w_t = torch.tensor(w_batch, dtype=torch.long)
                b_t = torch.tensor(b_batch, dtype=torch.long)
                s_t = torch.tensor(s_batch, dtype=torch.long)
                tgt = torch.tensor(t_batch, dtype=torch.float32)

                output = model(w_t, b_t, s_t)
                pred = torch.sigmoid(output / SCALE)
                loss = F.mse_loss(pred, tgt)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / max(batches, 1)
            print(f"  RL training loss: {avg_loss:.6f}")
            game_positions.clear()

    # Export final model
    model.export_nnue(output_path)
    print(f"\nSelf-play RL complete. NNUE exported to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KiyEngine NNUE Training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate training data
    gen_parser = subparsers.add_parser("generate", help="Generate training data from v5.2 model")
    gen_parser.add_argument("--model", default="kiyengine.gguf", help="Path to v5.2 GGUF model")
    gen_parser.add_argument("--output", default="data/train.bin", help="Output data file")
    gen_parser.add_argument("--positions", type=int, default=2000000, help="Number of positions")

    # Train NNUE
    train_parser = subparsers.add_parser("train", help="Train NNUE from labeled data")
    train_parser.add_argument("--data", default="data/train.bin", help="Training data file")
    train_parser.add_argument("--output", default="kiyengine.nnue", help="Output NNUE file")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16384, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE, help="Hidden size")

    # Self-play RL
    sp_parser = subparsers.add_parser("selfplay", help="Self-play RL refinement")
    sp_parser.add_argument("--nnue", default="kiyengine.nnue", help="Input NNUE file")
    sp_parser.add_argument("--games", type=int, default=10000, help="Number of games")
    sp_parser.add_argument("--output", default="kiyengine_rl.nnue", help="Output NNUE file")

    # Quick export (for testing)
    exp_parser = subparsers.add_parser("export-random", help="Export random NNUE (for testing)")
    exp_parser.add_argument("--output", default="kiyengine.nnue", help="Output NNUE file")
    exp_parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE, help="Hidden size")

    args = parser.parse_args()

    if args.command == "generate":
        generate_training_data(args.model, args.output, args.positions)
    elif args.command == "train":
        train_nnue(args.data, args.output, args.epochs, args.batch_size, args.lr, args.hidden)
    elif args.command == "selfplay":
        selfplay_rl(args.nnue, args.games, args.output)
    elif args.command == "export-random":
        model = NNUEModel(hidden_size=args.hidden)
        model.export_nnue(args.output)
        print("Exported random NNUE (for testing inference pipeline)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
