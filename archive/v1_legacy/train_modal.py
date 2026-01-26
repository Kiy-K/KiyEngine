import modal
import os
import subprocess
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine

# --- Modal Setup ---
stub = modal.Stub("kiy-engine-nnue-training")
volume = modal.Volume.from_name("fen-dataset", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "python-chess")
    .apt_install("stockfish", "git")
)

# --- PyTorch Student Model ---
class StudentNNUE(nn.Module):
    def __init__(self, input_features=768, hidden_neurons=32768):
        super(StudentNNUE, self).__init__()
        # A single, shared embedding layer for both perspectives
        self.embedding = nn.Linear(input_features, hidden_neurons, bias=False)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, x_white, x_black):
        # Process each perspective through the shared embedding layer
        hidden_white = self.embedding(x_white)
        hidden_black = self.embedding(x_black)

        # Apply Clipped ReLU activation, mimicking the Rust engine
        activated_white = torch.clamp(hidden_white, 0, 127)
        activated_black = torch.clamp(hidden_black, 0, 127)

        # Combine the activated accumulators before the final layer
        combined_activation = activated_white + activated_black

        return self.output(combined_activation)

# --- Feature Extraction (Correct Half-KAP Implementation) ---
def get_halfkp_features_from_board(board: chess.Board):
    """
    Generates two sparse feature vectors for a board state, one for each player's perspective.
    This is the correct Half-KAP (King-Agnostic-Piece) implementation.
    """
    white_indices = []
    black_indices = []

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            # Feature index calculation: (color * 6 + piece_type) * 64 + square
            color_idx = 0 if piece.color == chess.WHITE else 1
            piece_idx = piece.piece_type - 1

            # White's perspective index
            white_sq = sq
            white_feature_idx = (color_idx * 6 + piece_idx) * 64 + white_sq
            white_indices.append(white_feature_idx)

            # Black's perspective index (board is flipped vertically)
            black_sq = sq ^ 56
            black_feature_idx = (color_idx * 6 + piece_idx) * 64 + black_sq
            black_indices.append(black_feature_idx)

    return white_indices, black_indices

def create_batch_tensors(batch_fens, device):
    """Creates dense tensors for a batch of FENs efficiently."""
    batch_size = len(batch_fens)
    white_batch = torch.zeros(batch_size, 768, device=device)
    black_batch = torch.zeros(batch_size, 768, device=device)

    all_white_indices = []
    all_black_indices = []

    for i, fen in enumerate(batch_fens):
        board = chess.Board(fen.strip())
        white_indices, black_indices = get_halfkp_features_from_board(board)
        # Add the batch index to each feature index
        all_white_indices.extend([(i, idx) for idx in white_indices])
        all_black_indices.extend([(i, idx) for idx in black_indices])

    # Unzip indices for tensor assignment
    if all_white_indices:
        white_rows, white_cols = zip(*all_white_indices)
        white_batch[white_rows, white_cols] = 1.0

    if all_black_indices:
        black_rows, black_cols = zip(*all_black_indices)
        black_batch[black_rows, black_cols] = 1.0

    return white_batch, black_batch


# --- Helper Functions ---
def get_stockfish_eval(engine, fen):
    """Gets the NNUE evaluation from Stockfish for a given FEN."""
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=0))
    # Score is from white's perspective.
    score = info["score"].white().score(mate_score=10000)
    return score

def export_weights_to_bin(model, filename="weights.bin"):
    """Exports the trained and quantized weights to the specified binary format."""
    input_features = 768
    hidden_neurons = 32768
    output_neurons = 1
    quantization_scale = 256.0

    # Header
    magic_string = b"KIY-NNUE-V2"
    header_format = f"<16sIII f 32x" # Magic(16), Dims(12), Scale(4), Padding(32)

    # Quantize weights
    # Transpose embedding weight to match (Input, Output) layout expected by Rust
    feature_weights_f32 = model.embedding.weight.data.T.numpy()
    output_weights_f32 = model.output.weight.data.T.numpy()
    output_bias_f32 = model.output.bias.data.numpy()

    feature_weights_i16 = (feature_weights_f32 * quantization_scale).astype(np.int16)
    output_weights_i16 = (output_weights_f32 * quantization_scale).astype(np.int16)
    output_bias_i16 = (output_bias_f32 * quantization_scale).astype(np.int16)

    with open(filename, "wb") as f:
        # Write Header
        header_packed = struct.pack(
            header_format,
            magic_string,
            input_features,
            hidden_neurons,
            output_neurons,
            quantization_scale,
        )
        f.write(header_packed)
        f.write(feature_weights_i16.tobytes())
        f.write(output_weights_i16.tobytes())
        f.write(output_bias_i16.tobytes())
    print(f"Successfully exported quantized weights to {filename}")


# --- Modal Training Function ---
@stub.function(
    image=image,
    gpu="T4",
    timeout=18000, # 5 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("github-token")]
)
def train():
    print("Starting NNUE training process with corrected architecture...")
    stockfish_path = "/usr/games/stockfish"
    dataset_path = "/data/train.fen"

    # --- 1. Setup Teacher and Student ---
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    model = StudentNNUE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000) # T_max is a placeholder, will be updated

    # --- 2. Training Loop ---
    batch_size = 1024
    num_epochs = 1

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        with open(dataset_path, "w") as f:
            f.write("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")

    with open(dataset_path, "r") as f:
        fens = f.readlines()

    print(f"Training on {len(fens)} positions...")
    model.train()

    # Update T_max for the scheduler
    num_batches = len(fens) // batch_size
    scheduler.T_max = num_batches * num_epochs

    for epoch in range(num_epochs):
        for i in range(0, len(fens), batch_size):
            batch_fens = fens[i:i+batch_size]
            if not batch_fens:
                continue

            inputs_white, inputs_black = create_batch_tensors(batch_fens, device)
            # Stockfish score is from white's perspective, which is what we train against
            targets = torch.tensor([get_stockfish_eval(engine, fen) for fen in batch_fens], dtype=torch.float32).view(-1, 1).to(device)

            optimizer.zero_grad()
            # Pass both perspectives to the model
            outputs = model(inputs_white, inputs_black)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % (10 * batch_size) == 0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    engine.quit()
    print("Training finished.")

    # --- 3. Export and Push to Git ---
    export_weights_to_bin(model.cpu()) # Move model to CPU before exporting

    repo_url = "https://github.com/khoi/KiyEngine.git"
    branch = "tactical-nnue"
    github_pat = os.environ["GITHUB_TOKEN"]

    subprocess.run(["git", "config", "--global", "user.email", "jules-bot@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Jules the AI Engineer"])

    clone_url = f"https://x-access-token:{github_pat}@{repo_url.split('//')[1]}"
    subprocess.run(["git", "clone", "--branch", branch, clone_url, "KiyEngine"])

    subprocess.run(["mv", "weights.bin", "KiyEngine/"])

    os.chdir("KiyEngine")
    subprocess.run(["git", "add", "weights.bin"])
    subprocess.run(["git", "commit", "-m", "Distilled 50M Weights (Corrected Arch) - T4 Run"])
    subprocess.run(["git", "push", "origin", branch])

    print("Successfully pushed corrected weights.bin to GitHub.")
