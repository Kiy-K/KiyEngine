# KiyEngine V3

KiyEngine V3 is a high-performance, single-core optimized chess engine written in Rust. It features a novel evaluation and move ordering system based on a Mixture-of-Experts (MoE) Mamba (State Space Model) architecture, integrated into a classic Alpha-Beta search framework.

## Architecture

- **Model:** Mixture of Experts (MoE) with Mamba (SSM) blocks.
- **Scale:** ~25M parameters, designed to be compact and fit within CPU L3 cache.
- **Structure:** 4 layers, 8 Mamba block experts per layer.
- **Routing:** Top-k = 2 routing, activating the two best experts per token.
- **Search:** Iterative deepening Alpha-Beta search with Mamba state push/pop.

---

## How to Run

### Prerequisites
- Rust toolchain (latest stable version recommended)
- Git

### Compilation
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd kiy_engine_v3
   ```

2. **Compile in release mode:**
   This command builds the optimized binary.
   ```bash
   cargo build --release
   ```
   The executable will be located at `target/release/kiy_engine_v3`.

### Running the Engine
You can run the engine directly to interact with it via the Universal Chess Interface (UCI).
```bash
cargo run --release
```

---

## Usage Guide

KiyEngine V3 uses the standard UCI protocol. You can connect it to any UCI-compatible chess GUI (like Arena, Cute Chess, or Scid vs. PC) or interact with it directly from the command line.

### UCI Commands

- `uci`: Displays engine information and supported options.
- `isready`: Confirms the engine is ready to receive commands.
- `position startpos moves e2e4`: Sets the board to the starting position and plays the move e2e4.
- `position fen <fen_string>`: Sets the board to the given FEN string.
- `go depth 8`: Starts a search for the best move at a fixed depth of 8 plies.

### Custom `bench` Command
To measure the engine's performance (Nodes Per Second), use the `bench` command. This runs a fixed-depth search on the starting position.

```
bench
```
The engine will output the NPS count upon completion.

---

## Model Weights

The engine is designed to load its neural network weights from a `model.safetensors` file located in the root of the executable's directory.

### Weight Naming Convention
The `safetensors` file must follow a strict naming convention for the engine to load the tensors correctly.

- **Embeddings:** `embeddings`
- **Normalization:** `norm_w`
- **Output Heads:** `policy_head`, `value_head`
- **MoE Layers:** Tensors are named hierarchically:
  - Router Weights: `layer.{l}.router_w`
  - Mamba Block Weights: `layer.{l}.expert.{e}.{param_name}`
    - Where `{l}` is the layer index (0-3), `{e}` is the expert index (0-7), and `{param_name}` is one of the following: `in_proj_w`, `conv1d_w`, `conv1d_b`, `x_proj_w`, `dt_proj_w`, `dt_proj_b`, `a_log`, `d`, `out_proj_w`.

If `model.safetensors` is not found, the engine will initialize with random weights and print a warning.

---

## Deployment Guide

### Pushing to GitHub
1. **Initialize a new Git repository (if you haven't already):**
   ```bash
   git init
   git add .
   git commit -m "Initial commit of KiyEngine V3"
   ```
2. **Add a remote origin:**
   Replace `<your-username>` and `<your-repo-name>` with your GitHub details.
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   ```
3. **Push the code to the main branch:**
   ```bash
   git branch -M main
   git push -u origin main
   ```

### Uploading Model Weights to Hugging Face Hub

Hugging Face Hub is an excellent platform for hosting model weights.

1. **Install the Hugging Face CLI:**
   ```bash
   pip install -U "huggingface_hub[cli]"
   ```
2. **Log in to your Hugging Face account:**
   You will need to provide an access token.
   ```bash
   huggingface-cli login
   ```
3. **Create a new model repository:**
   You can do this on the Hugging Face website or via the CLI.
   ```bash
   huggingface-cli repo create your-model-name
   ```
4. **Upload your `model.safetensors` file:**
   ```bash
   huggingface-cli upload your-model-name /path/to/your/model.safetensors model.safetensors
   ```
   This command uploads your local file to the root of your Hugging Face model repository. Users can then download it to use with the engine.
