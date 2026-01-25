# training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from safetensors.torch import save_file
import os

from model import KiyEngineV3
from dataset import ChessDataset

def save_model_as_safetensors(model: KiyEngineV3, path: str):
    """ Saves the model weights in .safetensors format with the Rust core's naming convention. """
    tensors = {}

    # Embeddings and final normalization
    tensors["embeddings"] = model.embedding.weight
    tensors["norm_w"] = model.norm.weight

    # Policy and Value heads
    tensors["policy_head"] = model.policy_head.weight
    tensors["value_head"] = model.value_head[2].weight # Accessing the final linear layer

    # MoE Layers
    for l, layer in enumerate(model.layers):
        tensors[f"layer.{l}.router_w"] = layer.router.weight
        for e, expert in enumerate(layer.experts):
            prefix = f"layer.{l}.expert.{e}"
            tensors[f"{prefix}.in_proj_w"] = expert.in_proj.weight
            tensors[f"{prefix}.conv1d_w"] = expert.conv1d.weight
            tensors[f"{prefix}.conv1d_b"] = expert.conv1d.bias
            tensors[f"{prefix}.x_proj_w"] = expert.x_proj.weight
            tensors[f"{prefix}.dt_proj_w"] = expert.dt_proj.weight
            tensors[f"{prefix}.dt_proj_b"] = expert.dt_proj.bias
            tensors[f"{prefix}.a_log"] = expert.A_log
            tensors[f"{prefix}.d"] = expert.D
            tensors[f"{prefix}.out_proj_w"] = expert.out_proj.weight

    print(f"Saving model to {path}...")
    save_file(tensors, path)
    print("Model saved successfully.")

def train(config: dict):
    """ Main training loop. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Initialization
    model_config = config['model']
    model_config['noise_sigma'] = config['training']['noise_sigma']
    model = KiyEngineV3(model_config).to(device)

    # Data Loading (with a placeholder check for the data file)
    if not os.path.exists(config['paths']['train_data_path']):
        print(f"Error: Training data not found at {config['paths']['train_data_path']}")
        print("Please download the dataset and update the path in config.yaml.")
        # Create a dummy file to allow the script to run without erroring
        with open("dummy_dataset.pgn", "w") as f:
            f.write('[Event "Dummy"]\n[Result "1-0"]\n\n1. e4 e5 *')
        dataset = ChessDataset(pgn_file_path="dummy_dataset.pgn")
    else:
        dataset = ChessDataset(pgn_file_path=config['paths']['train_data_path'])

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Optimizer and Loss Functions
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Training Loop
    model.train()
    for epoch in range(config['training']['epochs']):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")

        for batch in pbar:
            input_seq, policy_target, value_target = [b.to(device) for b in batch]

            optimizer.zero_grad()

            policy_logits, value_pred, aux_loss = model(input_seq)

            # Composite Loss Calculation
            policy_loss = policy_loss_fn(policy_logits, policy_target)
            value_loss = value_loss_fn(value_pred.squeeze(), value_target.squeeze())

            loss = (config['training']['policy_weight'] * policy_loss +
                    config['training']['value_weight'] * value_loss +
                    config['training']['aux_loss_lambda'] * aux_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Policy": f"{policy_loss.item():.4f}",
                "Value": f"{value_loss.item():.4f}"
            })

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # Save the final model
    save_dir = config['paths']['save_path']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, config['paths']['model_save_name'])
    save_model_as_safetensors(model, save_path)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
