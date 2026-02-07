// src/bin/inspect_model.rs
//! Inspect safetensors model file to see available tensor keys (legacy v4 format)

use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("kiyengine.gguf");

    println!(
        "Inspecting model: {} (legacy v4 safetensors format)",
        model_path
    );
    println!("========================================");

    let file = File::open(model_path)?;
    // SAFETY: Memory mapping is safe for read-only files
    let mmap = unsafe { Mmap::map(&file)? };

    let tensors = candle_core::safetensors::load_buffer(&mmap, &candle_core::Device::Cpu)?;

    println!("\nAvailable tensors ({} total):\n", tensors.len());

    let mut keys: Vec<_> = tensors.keys().collect();
    keys.sort();

    for key in keys {
        if let Some(tensor) = tensors.get(key) {
            let shape: Vec<usize> = tensor.dims().iter().copied().collect();
            let dtype = format!("{:?}", tensor.dtype());
            println!("  {}: {:?} (dtype: {})", key, shape, dtype);
        }
    }

    println!("\n========================================");
    println!("Expected keys for current loader:");
    println!("  - embed.weight");
    println!("  - pos_embed");
    println!("  - layers.0.linear.weight");
    println!("  - layers.0.linear.norm.weight");
    println!("  - policy_head.*");
    println!("  - value_head.*");

    Ok(())
}
