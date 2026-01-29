// src/uci.rs
//
// UCI (Universal Chess Interface) protocol handler for KiyEngine V3.
// Enhanced with proper time management for Stockfish match.

use crate::engine::Engine;
use crate::tutor::Tutor;
use crate::search::{SearchHandler, TimeControl};
use crate::tt::TranspositionTable;
use chess::{Board, ChessMove, Color};
use std::str::FromStr;
use std::time::Instant;

const TT_SIZE_MB: usize = 512;

pub struct UciHandler {
    engine: Engine,
    tutor: Tutor,
    board: Board,
    tt: TranspositionTable,
    move_history: Vec<usize>, // Token history for the model
    debug_mode: bool,
}

impl UciHandler {
    pub fn new() -> anyhow::Result<Self> {
        let engine = Engine::new()?;
        let tutor = Tutor::new();
        Ok(Self {
            engine,
            tutor,
            board: Board::default(),
            tt: TranspositionTable::new(TT_SIZE_MB),
            move_history: Vec::new(),
            debug_mode: true, // Enable debug by default for match monitoring
        })
    }

    pub fn handle_command(&mut self, command: &str) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        match parts.first() {
            Some(&"uci") => self.handle_uci(),
            Some(&"debug") => self.handle_debug(&parts[1..]),
            Some(&"isready") => self.handle_isready(),
            Some(&"setoption") => self.handle_setoption(&parts[1..]),
            Some(&"ucinewgame") => self.handle_new_game(),
            Some(&"position") => self.handle_position(&parts[1..]),
            Some(&"go") => self.handle_go(&parts[1..]),
            Some(&"bench") => self.handle_bench(),
            Some(&"eval") => self.handle_eval(),
            Some(&"quit") => std::process::exit(0),
            _ => (), // Ignore unknown commands
        }
    }

    fn handle_uci(&self) {
        println!("id name KiyEngine V3 Mamba-MoE");
        println!("id author Khoi");
        println!();
        println!("option name Hash type spin default 512 min 1 max 4096");
        println!("option name Debug type check default true");
        println!("option name ScoutDepth type spin default 4 min 2 max 8");
        println!("option name BlunderThreshold type spin default 150 min 50 max 500");
        println!("option name OverheadMs type spin default 50 min 0 max 500");
        println!("uciok");
    }

    fn handle_debug(&mut self, parts: &[&str]) {
        if let Some(&"on") = parts.first() {
            self.debug_mode = true;
            println!("info string Debug mode enabled");
        } else if let Some(&"off") = parts.first() {
            self.debug_mode = false;
            println!("info string Debug mode disabled");
        }
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_setoption(&mut self, parts: &[&str]) {
        // Parse "setoption name X value Y"
        let name_idx = parts.iter().position(|&p| p == "name");
        let value_idx = parts.iter().position(|&p| p == "value");

        if let (Some(ni), Some(vi)) = (name_idx, value_idx) {
            let name: String = parts[ni + 1..vi].join(" ");
            let value = parts.get(vi + 1).copied().unwrap_or("");

            match name.as_str() {
                "Hash" => {
                    if let Ok(size) = value.parse::<usize>() {
                        self.tt = TranspositionTable::new(size);
                        println!("info string Hash table resized to {} MB", size);
                    }
                }
                "Debug" => {
                    self.debug_mode = value == "true";
                }
                _ => {}
            }
        }
    }

    fn handle_new_game(&mut self) {
        self.tt.clear();
        self.move_history.clear();
        self.board = Board::default();
        println!("info string New game started, tables cleared");
    }

    fn handle_position(&mut self, parts: &[&str]) {
        self.move_history.clear();

        let mut current_board = match parts.first() {
            Some(&"startpos") => Board::default(),
            Some(&"fen") => {
                let fen_parts: Vec<&str> = parts.iter().skip(1).take_while(|&&p| p != "moves").copied().collect();
                let fen_string = fen_parts.join(" ");
                Board::from_str(&fen_string).unwrap_or(Board::default())
            }
            _ => return,
        };

        if let Some(moves_idx) = parts.iter().position(|&p| p == "moves") {
            for move_str in &parts[moves_idx + 1..] {
                if let Ok(mv) = ChessMove::from_str(move_str) {
                    // Add token to history before making the move
                    let token = crate::search::move_to_token(&mv, &current_board);
                    self.move_history.push(token);

                    current_board = current_board.make_move_new(mv);
                }
            }
        }

        self.board = current_board;

        if self.debug_mode {
            let side = if self.board.side_to_move() == Color::White { "White" } else { "Black" };
            println!("info string Position set: {} to move, {} moves in history", side, self.move_history.len());
        }
    }

    fn handle_go(&mut self, parts: &[&str]) {
        let start_time = Instant::now();

        // Parse time control parameters
        let mut tc = TimeControl::default();

        let mut i = 0;
        while i < parts.len() {
            match parts[i] {
                "wtime" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        tc.wtime = Some(v);
                    }
                    i += 2;
                }
                "btime" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        tc.btime = Some(v);
                    }
                    i += 2;
                }
                "winc" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        tc.winc = Some(v);
                    }
                    i += 2;
                }
                "binc" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        tc.binc = Some(v);
                    }
                    i += 2;
                }
                "movestogo" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        tc.movestogo = Some(v);
                    }
                    i += 2;
                }
                "movetime" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        tc.movetime = Some(v);
                    }
                    i += 2;
                }
                "depth" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        tc.depth = Some(v);
                    }
                    i += 2;
                }
                "infinite" => {
                    tc.infinite = true;
                    i += 1;
                }
                "nodes" => {
                    // Skip nodes limit for now
                    i += 2;
                }
                _ => i += 1,
            }
        }

        // Default depth if not specified and no time control
        if tc.depth.is_none() && tc.wtime.is_none() && tc.btime.is_none() && tc.movetime.is_none() && !tc.infinite {
            tc.depth = Some(8);
        }

        if self.debug_mode {
            let is_white = self.board.side_to_move() == Color::White;
            let allocated = tc.calculate_time(is_white);
            println!("info string Time allocated: {}ms", allocated.as_millis());
        }

        // Create search handler with current state
        let mut searcher = SearchHandler::new(&self.engine, &self.tutor, &mut self.tt, self.board);

        // Copy move history to game state
        for &token in &self.move_history {
            searcher.game_state.push_token(token);
        }

        // Run search with time control
        let (best_move, best_score) = searcher.search_with_time(&tc);

        let elapsed = start_time.elapsed();

        if let Some(mv) = best_move {
            if self.debug_mode {
                println!(
                    "info string Search complete: {} ({}cp) in {}ms, {} nodes",
                    mv, best_score, elapsed.as_millis(), searcher.nodes_searched
                );
            }
            println!("bestmove {}", mv);
        } else {
            // If no move found, try to find any legal move
            let legal_moves = chess::MoveGen::new_legal(&self.board);
            if let Some(mv) = legal_moves.into_iter().next() {
                println!("info string WARNING: No best move found, selecting first legal move");
                println!("bestmove {}", mv);
            }
        }
    }

    fn handle_bench(&mut self) {
        println!("info string Running benchmark (depth 8)...");

        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];

        let mut total_nodes = 0u64;
        let start_time = Instant::now();

        for fen in &positions {
            let board = Board::from_str(fen).unwrap_or(Board::default());
            let mut searcher = SearchHandler::new(&self.engine, &self.tutor, &mut self.tt, board);

            let tc = TimeControl { depth: Some(8), ..Default::default() };
            searcher.search_with_time(&tc);
            total_nodes += searcher.nodes_searched;
        }

        let duration = start_time.elapsed();
        let nps = if duration.as_secs_f64() > 0.0 {
            (total_nodes as f64 / duration.as_secs_f64()) as u64
        } else {
            0
        };

        println!("info string Benchmark complete:");
        println!("info string   Positions: {}", positions.len());
        println!("info string   Total nodes: {}", total_nodes);
        println!("info string   Time: {:.2}s", duration.as_secs_f64());
        println!("info string   NPS: {}", nps);
    }

    fn handle_eval(&self) {
        let eval = self.tutor.evaluate(&self.board);
        let side = if self.board.side_to_move() == Color::White { "White" } else { "Black" };
        println!("info string Static eval: {}cp (from {}'s perspective)", eval, side);
    }
}
