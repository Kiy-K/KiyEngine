// src/bin/label_positions.rs
//! Generate NNUE training data by playing random games and labeling positions
//! with the v5.2 KiyNet transformer's value head.
//!
//! Uses KV cache for incremental inference within each game (~5ms per position
//! instead of ~10ms), achieving ~150-200 positions/second.
//!
//! Usage: cargo run --release --bin label_positions -- --positions 200000 --output data/train.bin

use chess::{Board, ChessMove, MoveGen};
use kiy_engine_v5_alpha::engine::Engine;
use kiy_engine_v5_alpha::uci::MoveCodec;
use std::io::{BufWriter, Write};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut num_positions: usize = 200_000;
    let mut output_path = String::from("train_nnue/data/train.bin");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--positions" | "-n" => {
                i += 1;
                num_positions = args[i].parse()?;
            }
            "--output" | "-o" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--help" | "-h" => {
                println!("Usage: label_positions [--positions N] [--output PATH]");
                println!("  --positions N   Number of positions to generate (default: 200000)");
                println!(
                    "  --output PATH   Output binary file (default: train_nnue/data/train.bin)"
                );
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== KiyEngine NNUE Data Generator ===");
    println!("Target positions: {}", num_positions);
    println!("Output: {}", output_path);

    // Load the v5.2 model
    let engine = Engine::new()?;

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(&output_path)?;
    let mut writer = BufWriter::new(file);
    let mut total_positions = 0usize;
    let mut total_games = 0usize;
    let start_time = std::time::Instant::now();

    // Use a simple RNG (xorshift64)
    let mut rng_state: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Create a KV cache for incremental inference
    let mut kv_cache = engine.create_kv_cache();

    println!("Generating positions via random self-play with KV cache...");

    while total_positions < num_positions {
        let mut board = Board::default();
        let mut move_tokens: Vec<usize> = Vec::with_capacity(128);

        // Clear KV cache for each new game
        kv_cache.clear();

        // Play a random game (8-120 plies)
        let max_plies = 8 + (xorshift64(&mut rng_state) % 113) as usize;
        // Limit to context length
        let max_plies = max_plies.min(engine.context_length - 1);

        for ply in 0..max_plies {
            if total_positions >= num_positions {
                break;
            }

            let legal: Vec<ChessMove> = MoveGen::new_legal(&board).collect();
            if legal.is_empty() {
                break;
            }

            // Pick a random legal move (with slight bias toward captures for interesting positions)
            let mv = pick_random_move(&legal, &board, &mut rng_state);

            // Convert to token and record
            let token = MoveCodec::move_to_token(&mv) as usize;
            move_tokens.push(token);

            // Make the move
            board = board.make_move_new(mv);

            // Skip early positions (first 6 plies = 3 full moves)
            if ply < 6 {
                continue;
            }

            // Run NN forward pass with KV cache (incremental — very fast!)
            match engine.forward_with_cache(&move_tokens, &mut kv_cache) {
                Ok((_policy, value)) => {
                    // Convert value [-1, 1] to centipawns
                    // value > 0 means good for side that just moved
                    // Scale: atanh-style mapping, ±0.5 → ±300cp, ±1.0 → ±600cp
                    let eval_cp = (value * 600.0).round() as i16;
                    let eval_cp = eval_cp.max(-10000).min(10000);

                    // Write entry: fen_len(u16) + fen(bytes) + eval_cp(i16)
                    let fen = format!("{}", board);
                    let fen_bytes = fen.as_bytes();
                    writer.write_all(&(fen_bytes.len() as u16).to_le_bytes())?;
                    writer.write_all(fen_bytes)?;
                    writer.write_all(&eval_cp.to_le_bytes())?;

                    total_positions += 1;
                }
                Err(e) => {
                    // KV cache may be stale, clear and retry next game
                    eprintln!("  Forward error at ply {}: {}", ply, e);
                    kv_cache.clear();
                    break;
                }
            }
        }

        total_games += 1;

        if total_games % 200 == 0 || total_positions >= num_positions {
            writer.flush()?;
            let elapsed = start_time.elapsed().as_secs_f64();
            let pps = total_positions as f64 / elapsed;
            let eta = if pps > 0.0 {
                (num_positions - total_positions) as f64 / pps
            } else {
                0.0
            };
            eprintln!(
                "  Games: {} | Positions: {}/{} | {:.0} pos/s | ETA: {:.0}s",
                total_games, total_positions, num_positions, pps, eta
            );
        }
    }

    writer.flush()?;
    let elapsed = start_time.elapsed().as_secs_f64();
    let file_size = std::fs::metadata(&output_path)?.len();

    eprintln!("\n=== Done ===");
    eprintln!("  Total games: {}", total_games);
    eprintln!("  Total positions: {}", total_positions);
    eprintln!(
        "  Time: {:.1}s ({:.0} pos/s)",
        elapsed,
        total_positions as f64 / elapsed
    );
    eprintln!(
        "  Output: {} ({:.1} MB)",
        output_path,
        file_size as f64 / 1024.0 / 1024.0
    );

    Ok(())
}

/// Pick a random legal move with slight bias toward captures (more interesting positions)
fn pick_random_move(legal: &[ChessMove], board: &Board, rng: &mut u64) -> ChessMove {
    // 30% chance: prefer a capture if available
    if xorshift64(rng) % 100 < 30 {
        let enemy = *board.color_combined(!board.side_to_move());
        let captures: Vec<ChessMove> = legal
            .iter()
            .filter(|mv| (chess::BitBoard::from_square(mv.get_dest()) & enemy).0 != 0)
            .copied()
            .collect();
        if !captures.is_empty() {
            let idx = (xorshift64(rng) % captures.len() as u64) as usize;
            return captures[idx];
        }
    }
    let idx = (xorshift64(rng) % legal.len() as u64) as usize;
    legal[idx]
}

/// xorshift64 PRNG
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
