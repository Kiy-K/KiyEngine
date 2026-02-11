// src/engine/tests.rs
//! Tests for the KiyNet neural network engine

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::constants::DEFAULT_MODEL_PATH;
    use candle_core::Device;
    use std::time::Instant;

    fn try_load_engine() -> Option<Engine> {
        if !std::path::Path::new(DEFAULT_MODEL_PATH).exists() {
            return None;
        }
        Engine::load_from_gguf(DEFAULT_MODEL_PATH, Device::Cpu).ok()
    }

    #[test]
    fn test_model_loading() {
        let _ = try_load_engine();
    }

    #[test]
    fn test_variable_sequence_lengths() {
        if let Some(engine) = try_load_engine() {
            for seq_len in [1, 4, 8, 16, 32, 64] {
                let tokens: Vec<usize> = (0..seq_len).map(|i| (i * 100) % 4608).collect();
                let result = engine.forward(&tokens);
                assert!(result.is_ok(), "Failed for seq_len {}", seq_len);

                if let Ok((policy, value)) = result {
                    assert!(value >= -1.0 && value <= 1.0);
                    assert_eq!(policy.dims(), &[4608]);
                }
            }
        }
    }

    #[test]
    fn test_inference_consistency() {
        if let Some(engine) = try_load_engine() {
            let tokens: Vec<usize> = vec![100, 200, 300, 400, 500];
            let mut values = vec![];
            for _ in 0..5 {
                if let Ok((_, value)) = engine.forward(&tokens) {
                    values.push(value);
                }
            }
            if !values.is_empty() {
                let first = values[0];
                for (i, &val) in values.iter().enumerate() {
                    assert!((val - first).abs() < 1e-6, "Inconsistent at iter {}", i);
                }
            }
        }
    }

    #[test]
    fn test_chess_position_evaluations() {
        if let Some(engine) = try_load_engine() {
            let positions = vec![
                vec![100, 200, 300, 400],
                vec![100, 200, 300, 400, 500, 600, 700, 800],
                vec![50, 100, 150, 200, 250],
            ];
            for (i, tokens) in positions.iter().enumerate() {
                let result = engine.forward(tokens);
                assert!(result.is_ok(), "Position {} failed", i);
            }
        }
    }

    #[test]
    #[ignore]
    fn benchmark_inference_performance() {
        let engine = try_load_engine().expect("No model");
        let tokens: Vec<usize> = (0..16).map(|i| (i * 150) % 4608).collect();
        let iterations = 100u32;

        for _ in 0..5 {
            let _ = engine.forward(&tokens);
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = engine.forward(&tokens);
        }
        let total = start.elapsed();
        let avg = total / iterations;
        println!(
            "Avg inference: {:?} ({:.1}/sec)",
            avg,
            iterations as f64 / total.as_secs_f64()
        );
    }
}
