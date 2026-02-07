// src/bin/model_benchmark.rs
//! Comprehensive benchmark for the fine-tuned KiyEngine model
//!
//! Tests model loading, inference speed, and evaluation accuracy

use candle_core::Device;
use kiy_engine_v5_alpha::constants::{DEFAULT_MODEL_PATH, LEGACY_MODEL_PATH};
use kiy_engine_v5_alpha::engine::Engine;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("KiyEngine Model Benchmark Suite v5.0.0");
    println!("========================================\n");

    let device = Device::Cpu;

    // Test 1: Model Loading
    println!("[Test 1] Model Loading");
    println!("-----------------------");

    let start = Instant::now();
    let finetuned_engine =
        Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device.clone()).map(Arc::new);
    let finetuned_load_time = start.elapsed();

    match &finetuned_engine {
        Ok(_) => println!(
            "Fine-tuned model:        Loaded in {:?}",
            finetuned_load_time
        ),
        Err(e) => println!("Fine-tuned model:        FAILED - {}", e),
    }

    let start = Instant::now();
    let legacy_engine =
        Engine::load_from_safetensors(LEGACY_MODEL_PATH, device.clone()).map(Arc::new);
    let legacy_load_time = start.elapsed();

    match &legacy_engine {
        Ok(_) => println!("Legacy model (v4):       Loaded in {:?}", legacy_load_time),
        Err(e) => println!("Legacy model (v4):       Not available - {}", e),
    }
    println!();

    // Test 2: Inference Speed Benchmark
    if finetuned_engine.is_ok() {
        println!("[Test 2] Inference Speed Benchmark");
        println!("-----------------------------------");

        let engine = finetuned_engine.as_ref().unwrap();

        // Warm-up runs
        let warmup_tokens: Vec<usize> = (0..10).map(|i| (i * 100) % 4608).collect();
        for _ in 0..5 {
            let _ = engine.forward(&warmup_tokens);
        }

        // Benchmark with varying sequence lengths
        let seq_lengths = vec![4, 8, 16, 32];
        let iterations = 100;

        for seq_len in seq_lengths {
            let test_tokens: Vec<usize> = (0..seq_len).map(|i| (i * 150) % 4608).collect();

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = engine.forward(&test_tokens);
            }
            let total_time = start.elapsed();
            let avg_time = total_time / iterations;
            let throughput = 1000.0 / avg_time.as_millis() as f64;

            println!(
                "Sequence length {}: avg={:?} ({:.1} inferences/sec)",
                seq_len, avg_time, throughput
            );
        }
        println!();
    }

    // Test 3: Position Evaluation Tests
    if finetuned_engine.is_ok() {
        println!("[Test 3] Position Evaluation Tests");
        println!("-----------------------------------");

        let engine = finetuned_engine.as_ref().unwrap();

        // Standard test positions
        let test_positions = vec![
            ("Start Position", vec![], 0.0), // Neutral start
            ("White Advantage", vec![100, 100, 100, 100, 100], 0.3), // Some moves made
            ("Tactical Complex", vec![1234, 567, 890, 123, 456, 789], 0.0),
        ];

        for (name, tokens, _expected_range) in test_positions {
            let start = Instant::now();
            let result = engine.forward(&tokens);
            let inference_time = start.elapsed();

            match result {
                Ok((policy, value)) => {
                    let policy_shape = policy.shape().clone();
                    println!(
                        "{}: value={:.3}, policy_shape={:?}, time={:?}",
                        name, value, policy_shape, inference_time
                    );
                }
                Err(e) => {
                    println!("{}: ERROR - {}", name, e);
                }
            }
        }
        println!();
    }

    // Test 4: Comparison Test (if legacy model available)
    if legacy_engine.is_ok() && finetuned_engine.is_ok() {
        println!("[Test 4] Model Comparison (Fine-tuned vs Legacy)");
        println!("--------------------------");

        let legacy = legacy_engine.as_ref().unwrap();
        let finetuned = finetuned_engine.as_ref().unwrap();

        let test_tokens: Vec<usize> = (0..16).map(|i| (i * 300) % 4608).collect();

        // Run both models
        let start = Instant::now();
        let legacy_result = legacy.forward(&test_tokens);
        let legacy_time = start.elapsed();

        let start = Instant::now();
        let finetuned_result = finetuned.forward(&test_tokens);
        let finetuned_time = start.elapsed();

        match (legacy_result, finetuned_result) {
            (Ok((_, legacy_val)), Ok((_, finetuned_val))) => {
                let diff = (legacy_val - finetuned_val).abs();
                println!(
                    "Value difference: {:.4} (legacy={:.3}, finetuned={:.3})",
                    diff, legacy_val, finetuned_val
                );
                println!(
                    "Inference time - Legacy: {:?}, Fine-tuned: {:?}",
                    legacy_time, finetuned_time
                );

                if finetuned_time < legacy_time {
                    let speedup = legacy_time.as_nanos() as f64 / finetuned_time.as_nanos() as f64;
                    println!("Fine-tuned model is {:.2}x faster!", speedup);
                }
            }
            _ => println!("Comparison failed - one or both models returned errors"),
        }
    }

    // Test 5: Stress Test (batch inference simulation)
    if finetuned_engine.is_ok() {
        println!("\n[Test 5] Stress Test (1000 inferences)");
        println!("---------------------------------------");

        let engine = finetuned_engine.as_ref().unwrap();
        let iterations = 1000;
        let mut success_count = 0;
        let mut error_count = 0;
        let mut total_time = Duration::ZERO;

        for i in 0..iterations {
            // Vary token sequence length and content
            let seq_len = 4 + (i % 28); // 4 to 32
            let tokens: Vec<usize> = (0..seq_len)
                .map(|j| ((i * 100 + j * 50) % 4608) as usize)
                .collect();

            let start = Instant::now();
            match engine.forward(&tokens) {
                Ok(_) => success_count += 1,
                Err(_) => error_count += 1,
            }
            total_time += start.elapsed();
        }

        let avg_time = total_time / iterations;
        let throughput = iterations as f64 / total_time.as_secs_f64();

        println!(
            "Success rate: {}/{} ({} errors)",
            success_count, iterations, error_count
        );
        println!("Average inference time: {:?}", avg_time);
        println!("Throughput: {:.1} inferences/sec", throughput);
    }

    println!("\n========================================");
    println!("Benchmark Complete");
    println!("========================================");

    Ok(())
}
