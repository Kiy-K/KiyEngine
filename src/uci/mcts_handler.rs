// src/uci/mcts_handler.rs
//! UCI Handler for MCTS Search Engine
//!
//! Provides a UCI-compatible interface that uses AlphaZero-style MCTS
//! instead of traditional alpha-beta search.

use crate::engine::Engine;
use crate::search::mcts::{MCTSConfig, MCTSSearcher};
use crate::uci::MoveCodec;
use chess::{Board, ChessMove, MoveGen};
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct MCTSUciHandler {
    searcher: MCTSSearcher,
    board: Board,
    move_history: Vec<usize>,
    position_history: Vec<u64>,
    stop_flag: Arc<AtomicBool>,
    config: MCTSConfig,
}

impl MCTSUciHandler {
    pub fn new() -> anyhow::Result<Self> {
        let engine = Arc::new(Engine::new()?);
        let config = MCTSConfig::default();
        let searcher = MCTSSearcher::new(engine, config.clone());

        Ok(Self {
            searcher,
            board: Board::default(),
            move_history: Vec::new(),
            position_history: Vec::new(),
            stop_flag: Arc::new(AtomicBool::new(false)),
            config,
        })
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(cmd) = line else { break };
            let cmd = cmd.trim();
            if cmd.is_empty() {
                continue;
            }
            self.handle_command(cmd);
        }
    }

    fn handle_command(&mut self, command: &str) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        match parts.get(0).copied() {
            Some("uci") => {
                println!("id name KiyEngine V5 MCTS");
                println!("id author Khoi");
                println!("option name Hash type spin default 512 min 1 max 65536");
                println!("option name Threads type spin default 4 min 1 max 256");
                println!("option name Simulations type spin default 16000 min 100 max 1000000");
                println!("option name Cpuct type spin default 125 min 50 max 500");
                println!("option name DirichletAlpha type spin default 30 min 1 max 100");
                println!("option name DirichletEpsilon type spin default 25 min 0 max 100");
                println!("option name FPUReduction type spin default 50 min 0 max 100");
                println!("uciok");
            }
            Some("isready") => {
                println!("readyok");
            }
            Some("ucinewgame") => {
                self.board = Board::default();
                self.move_history.clear();
                self.position_history.clear();
            }
            Some("position") => {
                self.handle_position(&parts[1..]);
            }
            Some("go") => {
                self.handle_go(&parts[1..]);
            }
            Some("stop") => {
                self.stop_flag.store(true, Ordering::SeqCst);
            }
            Some("setoption") => {
                self.handle_setoption(&parts[1..]);
            }
            Some("quit") => {
                std::process::exit(0);
            }
            _ => {}
        }
    }

    fn handle_setoption(&mut self, parts: &[&str]) {
        let mut value_idx = 0;
        for (i, &part) in parts.iter().enumerate() {
            if part == "value" {
                value_idx = i;
                break;
            }
        }

        if value_idx == 0 || value_idx == parts.len() - 1 {
            return;
        }

        if parts[0] != "name" {
            return;
        }

        let name = parts[1..value_idx].join(" ").to_lowercase();
        let value = parts[(value_idx + 1)..].join(" ");

        match name.as_str() {
            "threads" => {
                if let Ok(num) = value.parse::<usize>() {
                    self.config.threads = num.max(1);
                    self.searcher.set_threads(self.config.threads);
                }
            }
            "simulations" => {
                if let Ok(sim) = value.parse::<usize>() {
                    self.config.num_simulations = sim.clamp(100, 1_000_000);
                }
            }
            "cpuct" => {
                if let Ok(cp) = value.parse::<f32>() {
                    self.config.cpuct_base = cp / 100.0; // Stored as hundredths
                }
            }
            "dirichletepsilon" => {
                if let Ok(de) = value.parse::<f32>() {
                    self.config.dirichlet_epsilon = de / 100.0;
                }
            }
            "dirichletalpha" => {
                if let Ok(da) = value.parse::<f32>() {
                    self.config.dirichlet_alpha = da / 100.0;
                }
            }
            "fpureduction" => {
                if let Ok(fpu) = value.parse::<f32>() {
                    self.config.fpu_reduction = fpu / 100.0;
                }
            }
            _ => {}
        }
    }

    fn handle_position(&mut self, parts: &[&str]) {
        self.move_history.clear();
        self.position_history.clear();
        let mut i = 0;

        if parts.get(0) == Some(&"startpos") {
            self.board = Board::default();
            i = 1;
        } else if parts.get(0) == Some(&"fen") {
            let mut fen_parts = Vec::new();
            i = 1;
            while i < parts.len() && parts[i] != "moves" {
                fen_parts.push(parts[i]);
                i += 1;
            }
            let fen = fen_parts.join(" ");
            if let Ok(b) = Board::from_str(&fen) {
                self.board = b;
            }
        }

        self.position_history.push(self.board.get_hash());

        if parts.get(i) == Some(&"moves") {
            i += 1;
            for move_str in &parts[i..] {
                if let Ok(mv) = ChessMove::from_str(move_str) {
                    let token = MoveCodec::move_to_token(&mv);
                    self.move_history.push(token as usize);
                    self.board = self.board.make_move_new(mv);
                    self.position_history.push(self.board.get_hash());
                }
            }
        }
    }

    fn handle_go(&mut self, parts: &[&str]) {
        eprintln!("[MCTS Handler] handle_go called with: {:?}", parts);
        self.stop_flag.store(false, Ordering::SeqCst);

        let mut wtime: u64 = 0;
        let mut btime: u64 = 0;
        let mut winc: u64 = 0;
        let mut binc: u64 = 0;
        let mut _movestogo: u64 = 0;
        let mut depth: u64 = 0;
        let mut nodes: u64 = 0;
        let mut infinite = false;

        for i in 0..parts.len() {
            match parts[i] {
                "wtime" => wtime = parts.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0),
                "btime" => btime = parts.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0),
                "winc" => winc = parts.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0),
                "binc" => binc = parts.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0),
                "movestogo" => {
                    _movestogo = parts.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0)
                }
                "depth" => depth = parts.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0),
                "nodes" => nodes = parts.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0),
                "infinite" => infinite = true,
                _ => {}
            }
        }

        // Calculate time to use
        let time_ms = if infinite {
            u64::MAX
        } else if nodes > 0 {
            u64::MAX // Use node limit instead
        } else if depth > 0 {
            u64::MAX // Use depth/simulation limit instead
        } else {
            // Time management similar to alpha-beta version
            let side = self.board.side_to_move();
            let (time_left, inc) = if side == chess::Color::White {
                (wtime, winc)
            } else {
                (btime, binc)
            };

            let mut search_time = time_left / 20 + inc;
            if search_time > time_left {
                search_time = time_left.saturating_sub(50);
            }
            if search_time < 50 {
                search_time = 50;
            }
            search_time
        };

        eprintln!("[MCTS Handler] Calculated time_ms: {}", time_ms);

        // Override simulations if specified
        let mut config = self.config.clone();
        if nodes > 0 {
            config.num_simulations = nodes as usize;
        }
        if depth > 0 {
            // Convert depth to simulations: depth 5 = 10000 sims
            config.num_simulations = (depth as usize * 2000).max(100);
        }

        eprintln!(
            "[MCTS Handler] Spawning search thread with {} simulations",
            config.num_simulations
        );

        // Update the searcher's config before searching
        self.searcher.set_config(config.clone());

        // Use parallel search if threads > 1, otherwise use sequential
        let (best_move, _) = if config.threads > 1 {
            eprintln!(
                "[MCTS Handler] Using parallel search with {} threads",
                config.threads
            );
            self.searcher
                .search_parallel(&self.board, &self.move_history, time_ms)
        } else {
            eprintln!("[MCTS Handler] Using sequential search");
            self.searcher
                .search(&self.board, &self.move_history, time_ms)
        };
        println!("bestmove {}", best_move);

        eprintln!("[MCTS Handler] Search complete");
    }

    /// Get a fallback move when search fails
    #[allow(dead_code)]
    fn get_fallback_move(&self) -> Option<ChessMove> {
        let legal = MoveGen::new_legal(&self.board);
        legal.into_iter().next()
    }
}
