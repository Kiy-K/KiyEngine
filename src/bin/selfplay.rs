// src/bin/selfplay.rs
//! Self-play RL data generator using the actual search engine with NNUE.
//!
//! Plays games at low depth using alpha-beta search with NNUE evaluation,
//! records positions with game outcomes (W/D/L), and outputs training data.
//!
//! Usage: cargo run --release --bin selfplay -- --games 500 --depth 4 --output train_nnue/data/rl.bin

use chess::{Board, BoardStatus, Color, MoveGen};
use kiy_engine::nnue::NnueNetwork;
use kiy_engine::search::{SearchWorker, TranspositionTable};
use std::io::{BufWriter, Write};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut num_games: usize = 500;
    let mut search_depth: u8 = 4;
    let mut output_path = String::from("train_nnue/data/rl.bin");
    let mut nnue_path = String::from("kiyengine.nnue");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" | "-g" => {
                i += 1;
                num_games = args[i].parse()?;
            }
            "--depth" | "-d" => {
                i += 1;
                search_depth = args[i].parse()?;
            }
            "--output" | "-o" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--nnue" => {
                i += 1;
                nnue_path = args[i].clone();
            }
            "--help" | "-h" => {
                println!("Usage: selfplay [OPTIONS]");
                println!("  --games N     Number of games to play (default: 500)");
                println!("  --depth D     Search depth per move (default: 4)");
                println!("  --output PATH Output binary file (default: train_nnue/data/rl.bin)");
                println!("  --nnue PATH   NNUE weights file (default: kiyengine.nnue)");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!("=== KiyEngine Self-Play RL Data Generator ===");
    eprintln!("Games: {}, Depth: {}", num_games, search_depth);
    eprintln!("NNUE: {}", nnue_path);
    eprintln!("Output: {}", output_path);

    // Load NNUE
    let nnue = Arc::new(NnueNetwork::load(&nnue_path)?);
    eprintln!(
        "  NNUE loaded: hidden_size={}, {:.1}KB",
        nnue.hidden_size,
        std::fs::metadata(&nnue_path)?.len() as f64 / 1024.0
    );

    // Create TT (small — we clear between games)
    let tt = Arc::new(TranspositionTable::new(32)); // 32 MB

    // Create output
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    let start_time = std::time::Instant::now();
    let mut total_positions = 0usize;
    let mut results = [0usize; 3]; // [white_wins, draws, black_wins]

    let mut rng_state: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    for game_idx in 0..num_games {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let mut board = Board::default();
        let mut game_fens: Vec<String> = Vec::new();

        // Random opening: play 2-6 random moves for diversity
        let random_plies = 2 + (xorshift64(&mut rng_state) % 5) as usize;
        for _ in 0..random_plies {
            let legal: Vec<_> = MoveGen::new_legal(&board).collect();
            if legal.is_empty() {
                break;
            }
            let idx = (xorshift64(&mut rng_state) % legal.len() as u64) as usize;
            board = board.make_move_new(legal[idx]);
        }

        // Play the game using alpha-beta + NNUE
        let max_game_plies = 300;
        for _ply in 0..max_game_plies {
            if board.status() != BoardStatus::Ongoing {
                break;
            }

            let legal: Vec<_> = MoveGen::new_legal(&board).collect();
            if legal.is_empty() {
                break;
            }

            // Record this position
            game_fens.push(format!("{}", board));

            // Create a fresh worker for each move (clean history)
            // This is fine for low depth since history tables barely matter
            let mut worker = SearchWorker::new(
                Arc::clone(&tt),
                board,
                Arc::clone(&stop_flag),
                Some(Arc::clone(&nnue)),
                None, // No Syzygy TB for selfplay
            );
            worker.nnue_refresh(0);

            // Search at fixed depth
            let (score, best_move) = worker.alpha_beta_root(search_depth);

            if let Some(mv) = best_move {
                board = board.make_move_new(mv);
            } else {
                // No move found (shouldn't happen with legal moves)
                break;
            }

            // Early termination: actual mate detected by search
            if score.abs() > 29500 {
                break;
            }
        }

        // Determine game result
        let result = match board.status() {
            BoardStatus::Checkmate => {
                // Side to move is checkmated
                if board.side_to_move() == Color::White {
                    0.0 // Black wins
                } else {
                    1.0 // White wins
                }
            }
            BoardStatus::Stalemate => 0.5,
            _ => 0.5, // Draw by repetition/50-move/max plies
        };

        match result as u8 {
            1 => results[0] += 1,
            0 => results[2] += 1,
            _ => results[1] += 1,
        }

        // Write positions with game result
        for fen in &game_fens {
            let fen_bytes = fen.as_bytes();
            // Determine eval from white's perspective: result is already from white's POV
            // Convert to centipawns: 1.0 → +600, 0.5 → 0, 0.0 → -600
            let eval_cp = ((result - 0.5) * 1200.0_f64).round() as i16;

            writer.write_all(&(fen_bytes.len() as u16).to_le_bytes())?;
            writer.write_all(fen_bytes)?;
            writer.write_all(&eval_cp.to_le_bytes())?;
            total_positions += 1;
        }

        // Clear TT between games
        tt.clear();

        if (game_idx + 1) % 50 == 0 || game_idx == num_games - 1 {
            writer.flush()?;
            let elapsed = start_time.elapsed().as_secs_f64();
            let gps = (game_idx + 1) as f64 / elapsed;
            let eta = if gps > 0.0 {
                (num_games - game_idx - 1) as f64 / gps
            } else {
                0.0
            };
            eprintln!(
                "  Game {}/{}: W={} D={} L={} | positions={} | {:.1} games/s | ETA: {:.0}s",
                game_idx + 1,
                num_games,
                results[0],
                results[1],
                results[2],
                total_positions,
                gps,
                eta
            );
        }
    }

    writer.flush()?;
    let elapsed = start_time.elapsed().as_secs_f64();
    let file_size = std::fs::metadata(&output_path)?.len();

    eprintln!("\n=== Self-Play Complete ===");
    eprintln!(
        "  Games: {} (W={} D={} L={})",
        num_games, results[0], results[1], results[2]
    );
    eprintln!("  Positions: {}", total_positions);
    eprintln!(
        "  Time: {:.1}s ({:.2} games/s)",
        elapsed,
        num_games as f64 / elapsed
    );
    eprintln!(
        "  Output: {} ({:.1} MB)",
        output_path,
        file_size as f64 / 1024.0 / 1024.0
    );

    Ok(())
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
