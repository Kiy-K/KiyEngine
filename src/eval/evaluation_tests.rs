// src/eval/evaluation_tests.rs
//! Chess position evaluation tests

#[cfg(test)]
mod tests {
    use crate::constants::DEFAULT_MODEL_PATH;
    use crate::engine::Engine;
    use crate::uci::MoveCodec;
    use candle_core::Device;
    use chess::{Board, ChessMove};
    use std::str::FromStr;
    use std::time::Instant;

    fn try_load_engine() -> Option<Engine> {
        if !std::path::Path::new(DEFAULT_MODEL_PATH).exists() {
            return None;
        }
        Engine::load_from_gguf(DEFAULT_MODEL_PATH, Device::Cpu).ok()
    }

    fn moves_to_tokens(moves: &[&str], board: &mut Board) -> Vec<usize> {
        let mut tokens = Vec::new();
        for move_str in moves {
            if let Ok(mv) = ChessMove::from_str(move_str) {
                let token = MoveCodec::move_to_token(&mv);
                tokens.push(token as usize);
                *board = board.make_move_new(mv);
            }
        }
        tokens
    }

    #[test]
    fn test_opening_evaluations() {
        if let Some(engine) = try_load_engine() {
            let openings: Vec<(&str, Vec<&str>)> = vec![
                ("1.e4", vec!["e2e4"]),
                ("1.d4", vec!["d2d4"]),
                ("1.e4 e5", vec!["e2e4", "e7e5"]),
                ("1.e4 c5", vec!["e2e4", "c7c5"]),
            ];

            for (name, moves) in openings {
                let mut board = Board::default();
                let tokens = moves_to_tokens(&moves, &mut board);
                match engine.forward(&tokens) {
                    Ok((_, value)) => {
                        assert!(value >= -1.0 && value <= 1.0, "{} out of range", name);
                    }
                    Err(e) => panic!("{}: {}", name, e),
                }
            }
        }
    }

    #[test]
    fn test_evaluation_consistency() {
        if let Some(engine) = try_load_engine() {
            let paths = vec![
                vec!["e2e4", "e7e5", "g1f3", "g8f6"],
                vec!["g1f3", "g8f6", "e2e4", "e7e5"],
            ];

            let mut evals = Vec::new();
            for path in &paths {
                let mut board = Board::default();
                let tokens = moves_to_tokens(path, &mut board);
                if let Ok((_, value)) = engine.forward(&tokens) {
                    evals.push(value);
                }
            }

            if evals.len() >= 2 {
                println!(
                    "Same position, different paths: {:.3} vs {:.3}",
                    evals[0], evals[1]
                );
            }
        }
    }

    #[test]
    fn test_policy_head_output() {
        if let Some(engine) = try_load_engine() {
            let tokens = vec![100, 200, 300, 400, 500];
            match engine.forward(&tokens) {
                Ok((policy, _)) => {
                    assert_eq!(policy.dims(), &[4608]);
                    if let Ok(values) = policy.to_vec1::<f32>() {
                        let non_nan = values.iter().filter(|v| !v.is_nan()).count();
                        assert_eq!(non_nan, 4608, "Policy contains NaN");
                    }
                }
                Err(e) => panic!("Policy test failed: {}", e),
            }
        }
    }

    #[test]
    #[ignore]
    fn test_realistic_game_throughput() {
        let engine = try_load_engine().expect("No model");
        let games = 50;

        for length in [10, 20, 40, 64] {
            let start = Instant::now();
            for _ in 0..games {
                let tokens: Vec<usize> = (0..length).map(|i| (i * 97 + 53) % 4096).collect();
                let _ = engine.forward(&tokens);
            }
            let total = start.elapsed();
            println!(
                "{}-move: {:.1} games/sec",
                length,
                games as f64 / total.as_secs_f64()
            );
        }
    }
}
