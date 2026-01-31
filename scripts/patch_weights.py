import argparse
import os
from safetensors.numpy import load_file, save_file
import numpy as np

def patch_weights(input_path, output_path=None):
    if not output_path:
        output_path = input_path

    print(f"[Patch] Loading weights from {input_path}...")
    try:
        tensors = load_file(input_path)
    except FileNotFoundError:
        print(f"[Patch] File {input_path} not found. Skipping patch.")
        return

    new_tensors = {}
    patched = False

    for k, v in tensors.items():
        new_key = k
        
        # Patch 1: embeddings -> embedding (singular)
        if k.startswith("embeddings."):
            new_key = k.replace("embeddings.", "embedding.")
            print(f"[Patch] Renaming {k} -> {new_key}")
            patched = True
            
        # Patch 2: router -> skill_emb (Maia-style)
        if "router.weight" in k:
            new_key = k.replace("router.weight", "skill_emb.weight")
            print(f"[Patch] Renaming {k} -> {new_key}")
            patched = True
        if "router.bias" in k:
            new_key = k.replace("router.bias", "skill_emb.bias")
            print(f"[Patch] Renaming {k} -> {new_key}")
            patched = True
            
        # Patch 3: layers.N.experts.E -> layers.N.experts.E (No change needed usually)
        # But ensure we keep all other keys
        
        new_tensors[new_key] = v

    if patched:
        print(f"[Patch] Saving patched weights to {output_path}...")
        save_file(new_tensors, output_path, metadata={"format": "pt"})
        print("[Patch] Done.")
    else:
        print("[Patch] No changes needed. Weights are already compliant.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.safetensors")
    args = parser.parse_args()
    
    patch_weights(args.model)
