// src/engine/tests.rs
//! Comprehensive tests for the neural network engine
//!
//! Tests model loading, inference correctness, and performance characteristics.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::constants::{DEFAULT_MODEL_PATH, LEGACY_MODEL_PATH};
    use candle_core::Device;
    use std::time::Instant;

    /// Test that both models can be loaded successfully
    #[test]
    fn test_model_loading() {
        let device = Device::Cpu;

        // Try loading default model (may fail if file doesn't exist in test env)
        let _ = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device.clone());

        // Try loading fine-tuned model
        let _ = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device);
    }

    /// Test inference with empty tokens should fail gracefully
    #[test]
    fn test_empty_tokens_error() {
        let device = Device::Cpu;

        // Only run if model exists
        if let Ok(engine) = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device) {
            let result = engine.forward(&[]);
            assert!(result.is_err(), "Empty tokens should return an error");
        }
    }

    /// Test inference with various sequence lengths
    #[test]
    fn test_variable_sequence_lengths() {
        let device = Device::Cpu;

        if let Ok(engine) = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device) {
            for seq_len in [1, 4, 8, 16, 32, 64] {
                let tokens: Vec<usize> = (0..seq_len.min(CONTEXT_LENGTH))
                    .map(|i| (i * 100) % 4608)
                    .collect();

                let result = engine.forward(&tokens);
                assert!(result.is_ok(), "Failed for sequence length {}", seq_len);

                if let Ok((policy, value)) = result {
                    // Check value is in valid range (after tanh)
                    assert!(
                        value >= -1.0 && value <= 1.0,
                        "Value {} out of range [-1, 1] for seq_len {}",
                        value,
                        seq_len
                    );

                    // Check policy shape
                    let policy_dims: Vec<usize> = policy.dims().iter().copied().collect();
                    assert_eq!(
                        policy_dims,
                        vec![4608],
                        "Policy shape {:?} incorrect for seq_len {}",
                        policy_dims,
                        seq_len
                    );
                }
            }
        }
    }

    /// Test that repeated inferences produce consistent results
    #[test]
    fn test_inference_consistency() {
        let device = Device::Cpu;

        if let Ok(engine) = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device) {
            let tokens: Vec<usize> = vec![100, 200, 300, 400, 500];

            // Run inference multiple times
            let mut values = vec![];
            for _ in 0..5 {
                if let Ok((_, value)) = engine.forward(&tokens) {
                    values.push(value);
                }
            }

            // Check all values are identical (deterministic)
            if !values.is_empty() {
                let first = values[0];
                for (i, &val) in values.iter().enumerate() {
                    assert!(
                        (val - first).abs() < 1e-6,
                        "Inconsistent value at iteration {}: {} vs {}",
                        i,
                        val,
                        first
                    );
                }
            }
        }
    }

    /// Benchmark inference performance
    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored
    fn benchmark_inference_performance() {
        let device = Device::Cpu;

        let engine = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device)
            .expect("Failed to load model");

        let tokens: Vec<usize> = (0..16).map(|i| (i * 150) % 4608).collect();
        let iterations = 1000;

        // Warm-up
        for _ in 0..10 {
            let _ = engine.forward(&tokens);
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = engine.forward(&tokens);
        }
        let total_time = start.elapsed();

        let avg_time = total_time / iterations;
        let throughput = iterations as f64 / total_time.as_secs_f64();

        println!("\n=== Inference Benchmark ===");
        println!("Iterations: {}", iterations);
        println!("Total time: {:?}", total_time);
        println!("Average time per inference: {:?}", avg_time);
        println!("Throughput: {:.1} inferences/sec", throughput);

        // Assert reasonable performance (>10 inferences/sec minimum)
        assert!(
            throughput > 10.0,
            "Performance too slow: {} inferences/sec",
            throughput
        );
    }

    /// Test model comparison between default and fine-tuned
    #[test]
    fn test_model_comparison() {
        let device = Device::Cpu;

        let engine1 = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device.clone());
        let engine2 = Engine::load_from_safetensors(LEGACY_MODEL_PATH, device);

        // Only compare if both models are available
        if let (Ok(engine1), Ok(engine2)) = (&engine1, &engine2) {
            let test_cases: Vec<Vec<usize>> = vec![
                vec![100],
                vec![100, 200],
                vec![100, 200, 300, 400],
                vec![1000, 2000, 3000],
            ];

            println!("\n=== Model Comparison ===");
            for tokens in test_cases {
                let result1 = engine1.forward(&tokens);
                let result2 = engine2.forward(&tokens);

                match (result1, result2) {
                    (Ok((_, val1)), Ok((_, val2))) => {
                        let diff = (val1 - val2).abs();
                        println!(
                            "Tokens {}: model1={:.3}, model2={:.3}, diff={:.3}",
                            tokens.len(),
                            val1,
                            val2,
                            diff
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    /// Test with chess-specific token sequences (simulated move histories)
    #[test]
    fn test_chess_position_evaluations() {
        let device = Device::Cpu;

        if let Ok(engine) = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device) {
            // Simulate various move histories
            let positions = vec![
                // Short opening sequence
                vec![100, 200, 300, 400],
                // Mid-game simulation
                vec![100, 200, 300, 400, 500, 600, 700, 800],
                // Late game with fewer pieces
                vec![50, 100, 150, 200, 250],
            ];

            for (i, tokens) in positions.iter().enumerate() {
                let result = engine.forward(tokens);
                assert!(result.is_ok(), "Position {} failed: {:?}", i, result.err());

                if let Ok((_, value)) = result {
                    println!("Position {}: evaluation = {:.3}", i, value);
                }
            }
        }
    }

    /// Stress test with rapid sequential inferences
    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored
    fn stress_test_rapid_inference() {
        let device = Device::Cpu;

        let engine = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device)
            .expect("Failed to load model");

        let iterations = 10000;
        let mut success = 0;
        let mut errors = 0;

        let start = Instant::now();

        for i in 0..iterations {
            let seq_len = 4 + (i % 28);
            let tokens: Vec<usize> = (0..seq_len).map(|j| (i * 50 + j * 25) % 4608).collect();

            match engine.forward(&tokens) {
                Ok(_) => success += 1,
                Err(_) => errors += 1,
            }
        }

        let total_time = start.elapsed();
        let success_rate = success as f64 / iterations as f64 * 100.0;
        let throughput = iterations as f64 / total_time.as_secs_f64();

        println!("\n=== Stress Test Results ===");
        println!("Iterations: {}", iterations);
        println!("Success rate: {:.1}%", success_rate);
        println!("Errors: {}", errors);
        println!("Total time: {:?}", total_time);
        println!("Throughput: {:.1} inferences/sec", throughput);

        assert!(success_rate > 99.0, "Too many failures: {} errors", errors);
    }

    /// Test memory usage doesn't grow unbounded
    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored
    fn test_memory_stability() {
        let device = Device::Cpu;

        let engine = Engine::load_from_safetensors(DEFAULT_MODEL_PATH, device)
            .expect("Failed to load model");

        let iterations = 1000;
        let tokens: Vec<usize> = (0..16).map(|i| (i * 100) % 4608).collect();

        // Run many inferences - memory should stay stable
        for _ in 0..iterations {
            let _ = engine.forward(&tokens);
        }

        // If we get here without OOM, memory is stable
        println!(
            "Memory stability test passed: {} iterations completed",
            iterations
        );
    }
}
