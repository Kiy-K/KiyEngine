//! KiyEngine V3 - Main Entry Point
//!
//! This file initializes the core components of the engine, including the
//! transposition table and the Mamba-MoE neural network model.

// Declare the modules
mod model;
mod tt;

use crate::model::MambaMoEModel;
use crate::tt::TranspositionTable;
use anyhow::Result;

fn main() -> Result<()> {
    println!("Initializing KiyEngine V3...");

    // 1. Initialize the global Transposition Table
    let tt = TranspositionTable::default();
    let hash_size = tt.size_mb();

    // 2. Attempt to load the Mamba-MoE model
    let model_status = match MambaMoEModel::load("model.safetensors") {
        Ok(_) => "Loaded successfully",
        Err(e) => {
            // Check if the error was due to the fallback (file not found)
            if e.to_string().contains("not found") {
                "Initialized with random weights (model.safetensors not found)"
            } else {
                // A more serious error occurred during loading
                eprintln!("Critical Error: Failed to load or initialize model: {}", e);
                "Error"
            }
        }
    };

    // 3. Print the system ready message
    println!("\n--== System Ready ==--");
    println!("- Transposition Table: {} MB", hash_size);
    println!("- Mamba-MoE Model: {}", model_status);
    println!("--==================--");

    // The engine would then proceed to the UCI loop or other commands.
    // For now, we just exit.
    Ok(())
}
