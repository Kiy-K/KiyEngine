# training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm
from safetensors.torch import save_file
import os

from model import KiyEngineV3
from dataset import ChessDataset

def setup_ddp(rank: int, world_size: int, backend: str):
    """ Initializes the distributed process group. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_ddp():
    """ Cleans up the distributed process group. """
    dist.destroy_process_group()

def save_model_as_safetensors(model: KiyEngineV3, path: str):
    """ Saves the model weights in .safetensors format with the Rust core's naming convention. """
    # When using DDP, the model is wrapped. We need to access the underlying module.
    model_to_save = model.module if isinstance(model, DDP) else model

    tensors = {}
    # ... (rest of the saving logic is identical)
    tensors["embeddings"] = model_to_save.embedding.weight
    tensors["norm_w"] = model_to_save.norm.weight
    tensors["policy_head"] = model_to_save.policy_head.weight
    # Correctly access the final layer in the sequential value head
    tensors["value_head"] = model_to_save.value_head[-1].weight

    for l, layer in enumerate(model_to_save.layers):
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

    print(f"Rank 0: Saving model to {path}...")
    save_file(tensors, path)
    print("Rank 0: Model saved successfully.")

def train_ddp(rank: int, world_size: int, config: dict):
    """ The core training function to be run on each GPU process. """
    setup_ddp(rank, world_size, config['ddp']['backend'])

    # --- Model Initialization ---
    model_config = config['model']
    model_config['noise_sigma'] = config['training']['noise_sigma']
    model = KiyEngineV3(model_config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # --- Data Loading ---
    dataset = ChessDataset(pgn_file_path=config['paths']['train_data_path'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # --- Optimizer, Loss, and Scaler ---
    optimizer = optim.Adam(ddp_model.parameters(), lr=config['training']['learning_rate'])
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    ddp_model.train()
    for epoch in range(config['training']['epochs']):
        sampler.set_epoch(epoch) # Important for shuffling in DDP

        # tqdm progress bar only on the primary process (rank 0)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", disable=(rank != 0))

        for batch in pbar:
            input_seq, policy_target, value_target = [b.to(rank) for b in batch]
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                policy_logits, value_pred, aux_loss = ddp_model(input_seq)
                policy_loss = policy_loss_fn(policy_logits, policy_target)
                value_loss = value_loss_fn(value_pred.squeeze(), value_target.squeeze())

                # Synchronize the aux loss across all GPUs
                dist.all_reduce(aux_loss, op=dist.ReduceOp.SUM)
                avg_aux_loss = aux_loss / world_size

                loss = (config['training']['policy_weight'] * policy_loss +
                        config['training']['value_weight'] * value_loss +
                        config['training']['aux_loss_lambda'] * avg_aux_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # --- Save Model (Rank 0 only) ---
    if rank == 0:
        save_dir = config['paths']['save_path']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, config['paths']['model_save_name'])
        save_model_as_safetensors(ddp_model, save_path)

    cleanup_ddp()

def main():
    """ Main function to handle DDP spawning or single-GPU training. """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Check if we should use DDP
    use_ddp = torch.cuda.is_available() and torch.cuda.device_count() > 1 and config['ddp']['world_size'] > 1

    if use_ddp:
        world_size = config['ddp']['world_size']
        print(f"Spawning {world_size} DDP processes.")
        mp.spawn(train_ddp,
                 args=(world_size, config),
                 nprocs=world_size,
                 join=True)
    else:
        print("Starting single-GPU training.")
        # Fallback to a simplified single-GPU training loop
        # This part is similar to the original train.py for flexibility
        rank = 0 # Single process is always rank 0
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        model_config = config['model']
        model_config['noise_sigma'] = config['training']['noise_sigma']
        model = KiyEngineV3(model_config).to(device)

        dataset = ChessDataset(pgn_file_path=config['paths']['train_data_path'])
        dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        policy_loss_fn = nn.CrossEntropyLoss()
        value_loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(config['training']['epochs']):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
            for batch in pbar:
                input_seq, policy_target, value_target = [b.to(device) for b in batch]
                optimizer.zero_grad()
                policy_logits, value_pred, aux_loss = model(input_seq)
                policy_loss = policy_loss_fn(policy_logits, policy_target)
                value_loss = value_loss_fn(value_pred.squeeze(), value_target.squeeze())
                loss = (config['training']['policy_weight'] * policy_loss +
                        config['training']['value_weight'] * value_loss +
                        config['training']['aux_loss_lambda'] * aux_loss)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        save_dir = config['paths']['save_path']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, config['paths']['model_save_name'])
        save_model_as_safetensors(model, save_path)

if __name__ == "__main__":
    main()
