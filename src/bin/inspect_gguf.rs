// src/bin/inspect_gguf.rs
//! Inspect GGUF model file to see metadata and tensor information
//!
//! Usage: cargo run --bin inspect_gguf -- path/to/model.gguf

use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("kiyengine.gguf");

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           KiyEngine - GGUF Model Inspector                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load GGUF file
    let gguf = kiyengine::engine::gguf::GgufFile::from_path(model_path)?;

    // Display header info
    println!("📁 File: {}", model_path);
    println!("📊 Version: GGUF v{}", gguf.version);
    println!();

    // Display general metadata
    println!("📋 General Metadata:");
    println!("   ─────────────────────────────────────────────────────────────");
    if let Some(arch) = gguf.get_string("general.architecture") {
        println!("   Architecture: {}", arch);
    }
    if let Some(name) = gguf.get_string("general.name") {
        println!("   Name: {}", name);
    }
    if let Some(version) = gguf.get_string("general.version") {
        println!("   Version: {}", version);
    }
    if let Some(author) = gguf.get_string("general.author") {
        println!("   Author: {}", author);
    }
    println!();

    // Display KiyEngine-specific metadata
    println!("🔧 KiyEngine Architecture:");
    println!("   ─────────────────────────────────────────────────────────────");
    if let Some(vocab) = gguf.get_u32("kiyengine.vocab_size") {
        println!("   Vocab Size: {}", vocab);
    }
    if let Some(ctx) = gguf.get_u32("kiyengine.context_length") {
        println!("   Context Length: {}", ctx);
    }
    if let Some(d_model) = gguf.get_u32("kiyengine.embedding_length") {
        println!("   D-Model: {}", d_model);
    }
    if let Some(layers) = gguf.get_u32("kiyengine.block_count") {
        println!("   Num Layers: {}", layers);
    }
    if let Some(ff) = gguf.get_u32("kiyengine.feed_forward_length") {
        println!("   Feed Forward: {}", ff);
    }
    if let Some(heads) = gguf.get_u32("kiyengine.attention.head_count") {
        println!("   Attention Heads: {}", heads);
    }
    if let Some(eps) = gguf.get_f32("kiyengine.attention.layer_norm_rms_epsilon") {
        println!("   RMS Norm Epsilon: {:.1e}", eps);
    }
    println!();

    // Display layer scale factors
    println!("⚖️  BitNet Scale Factors:");
    println!("   ─────────────────────────────────────────────────────────────");
    let mut has_scales = false;
    for i in 0..10 {
        // Check first 10 layers
        let key = format!("kiyengine.layers.{}.linear.scale", i);
        if let Some(scale) = gguf.get_f32(&key) {
            println!("   Layer {}: scale = {:.6}", i, scale);
            has_scales = true;
        }
    }
    if !has_scales {
        println!("   (No scale factors found - using default 1.0)");
    }
    println!();

    // Display tensor info
    println!("📦 Tensors ({} total):", gguf.tensors.len());
    println!("   ─────────────────────────────────────────────────────────────");
    println!("   {:<35} {:>18} {:>12}", "Name", "Shape", "Type");
    println!("   ─────────────────────────────────────────────────────────────");

    let mut tensors: Vec<_> = gguf.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    for tensor in &tensors {
        let shape_str = tensor
            .dimensions
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let qtype_str = format!("{:?}", tensor.quantization);
        println!(
            "   {:<35} {:>18} {:>12}",
            if tensor.name.len() > 34 {
                format!("{}…", &tensor.name[..33])
            } else {
                tensor.name.clone()
            },
            shape_str,
            qtype_str
        );
    }
    println!();

    // Calculate total size
    let total_params: u64 = tensors.iter().map(|t| t.n_elements).sum();
    let total_bytes: u64 = tensors
        .iter()
        .map(|t| t.n_elements * t.quantization.type_size() as u64)
        .sum();

    println!("📊 Summary:");
    println!("   ─────────────────────────────────────────────────────────────");
    println!("   Total Parameters: {:>15}", format_number(total_params));
    println!("   Total Size: {:>21}", format_bytes(total_bytes));
    println!(
        "   Tensor Data Offset: {:>15} bytes",
        gguf.tensor_data_offset
    );
    println!();

    Ok(())
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_bytes(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024 * 1024 {
        format!("{:.2} MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.2} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}
