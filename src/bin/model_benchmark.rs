// src/bin/model_benchmark.rs
//! v5.2.0 KiyNet_Ultimate benchmark

use candle_core::Device;
use kiyengine::constants::DEFAULT_MODEL_PATH;
use kiyengine::engine::Engine;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("KiyEngine Model Benchmark Suite v5.2.0");
    println!("========================================\n");

    let device = Device::Cpu;

    // Test 1: Model Loading
    println!("[Test 1] Model Loading");
    println!("-----------------------");

    let start = Instant::now();
    let engine = Engine::load_from_gguf(DEFAULT_MODEL_PATH, device)?.into();
    let load_time = start.elapsed();
    let engine: Arc<Engine> = engine;
    println!("Model loaded in {:?}", load_time);
    println!();

    // Test 2: Inference Speed Benchmark
    println!("[Test 2] Inference Speed Benchmark");
    println!("-----------------------------------");

    // Warm-up
    let warmup_tokens: Vec<usize> = (0..10).map(|i| (i * 100) % 4608).collect();
    for _ in 0..3 {
        let _ = engine.forward(&warmup_tokens);
    }

    let seq_lengths = vec![4, 8, 16, 32, 64];
    let iterations = 50u32;

    for seq_len in seq_lengths {
        let test_tokens: Vec<usize> = (0..seq_len).map(|i| (i * 150) % 4608).collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = engine.forward(&test_tokens);
        }
        let total_time = start.elapsed();
        let avg_time = total_time / iterations;
        let throughput = iterations as f64 / total_time.as_secs_f64();

        println!(
            "seq_len={:>2}: avg={:?} ({:.1} inf/sec)",
            seq_len, avg_time, throughput
        );
    }
    println!();

    // Test 3: Position Evaluation
    println!("[Test 3] Position Evaluation");
    println!("----------------------------");

    let test_positions: Vec<(&str, Vec<usize>)> = vec![
        ("Start Position", vec![]),
        ("Short game", vec![100, 200, 300, 400, 500]),
        ("Tactical", vec![1234, 567, 890, 123, 456, 789]),
    ];

    for (name, tokens) in &test_positions {
        let start = Instant::now();
        match engine.forward(tokens) {
            Ok((policy, value)) => {
                println!(
                    "{}: value={:.3}, policy_shape={:?}, time={:?}",
                    name,
                    value,
                    policy.shape(),
                    start.elapsed()
                );
            }
            Err(e) => println!("{}: ERROR - {}", name, e),
        }
    }
    println!();

    // Test 4: Stress Test
    println!("[Test 4] Stress Test (500 inferences)");
    println!("--------------------------------------");

    let stress_iters = 500u32;
    let mut success = 0u32;
    let mut total_time = Duration::ZERO;

    for i in 0..stress_iters {
        let seq_len = 4 + (i as usize % 60);
        let tokens: Vec<usize> = (0..seq_len)
            .map(|j| (i as usize * 100 + j * 50) % 4608)
            .collect();

        let start = Instant::now();
        if engine.forward(&tokens).is_ok() {
            success += 1;
        }
        total_time += start.elapsed();
    }

    let avg = total_time / stress_iters;
    let throughput = stress_iters as f64 / total_time.as_secs_f64();
    println!("Success: {}/{}", success, stress_iters);
    println!("Avg: {:?} ({:.1} inf/sec)", avg, throughput);

    // Test 5: KV Cache Incremental Inference
    println!("\n[Test 5] KV Cache Incremental Inference");
    println!("----------------------------------------");

    {
        let mut cache = engine.create_kv_cache();

        // Simulate a game: start with 4 tokens, add 1-2 each "move"
        let base_tokens: Vec<usize> = (0..4).map(|i| (i * 150) % 4608).collect();

        // Cold start (cache miss) — full forward
        let start = Instant::now();
        let (_, val_cold) = engine.forward_with_cache(&base_tokens, &mut cache)?;
        let cold_time = start.elapsed();
        println!(
            "Cold  (4 tokens, cache miss):  {:?}  value={:.3}",
            cold_time, val_cold
        );

        // Verify cache populated
        assert_eq!(cache.seq_len, 4);

        // Incremental: add 1 token at a time (cache hit)
        let mut tokens = base_tokens.clone();
        let mut cache_times = vec![];
        for step in 0..10 {
            tokens.push((1000 + step * 77) % 4608);
            let start = Instant::now();
            let (_, _val) = engine.forward_with_cache(&tokens, &mut cache)?;
            cache_times.push(start.elapsed());
        }
        let avg_cached = cache_times.iter().sum::<Duration>() / cache_times.len() as u32;
        println!(
            "Cached (+1 token, 10 steps):   avg={:?}  (seq 5→14)",
            avg_cached
        );

        // Compare: full forward at same seq_len (no cache)
        let start = Instant::now();
        let iters = 10u32;
        for _ in 0..iters {
            let _ = engine.forward(&tokens);
        }
        let avg_full = start.elapsed() / iters;
        println!("Full  (14 tokens, no cache):   avg={:?}", avg_full);

        let speedup = avg_full.as_secs_f64() / avg_cached.as_secs_f64();
        println!("KV Cache speedup: {:.1}x", speedup);

        // Verify correctness: cached vs non-cached should produce same value
        let mut fresh_cache = engine.create_kv_cache();
        let (_, val_cached) = engine.forward_with_cache(&tokens, &mut fresh_cache)?;
        let (_, val_full) = engine.forward(&tokens).map(|(p, v)| (p, v))?;
        let diff = (val_cached - val_full).abs();
        println!(
            "Correctness: cached={:.6} full={:.6} diff={:.6} {}",
            val_cached,
            val_full,
            diff,
            if diff < 1e-4 { "✓" } else { "✗ MISMATCH" }
        );
    }

    println!("\n========================================");
    println!("Benchmark Complete");
    println!("========================================");

    Ok(())
}
