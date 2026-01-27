// src/uci.rs

use crate::engine::Engine;
use crate::tutor::Tutor;
use crate::search::SearchHandler;
use crate::tt::TranspositionTable;
use chess::{Board, ChessMove};
use std::str::FromStr;

const TT_SIZE_MB: usize = 512;

pub struct UciHandler {
    engine: Engine,
    tutor: Tutor,
    board: Board,
    tt: TranspositionTable,
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
        })
    }

    pub fn handle_command(&mut self, command: &str) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        match parts.get(0) {
            Some(&"uci") => self.handle_uci(),
            Some(&"isready") => self.handle_isready(),
            Some(&"ucinewgame") => self.handle_new_game(),
            Some(&"position") => self.handle_position(&parts[1..]),
            Some(&"go") => self.handle_go(&parts[1..]),
            Some(&"bench") => self.handle_bench(),
            Some(&"quit") => std::process::exit(0),
            _ => (), // Ignore unknown commands
        }
    }

    fn handle_uci(&self) {
        println!("id name KiyEngine V3");
        println!("id author Jules");
        println!("uciok");
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_new_game(&mut self) {
        self.tt.clear();
    }

    fn handle_position(&mut self, parts: &[&str]) {
        let mut current_board = match parts.get(0) {
            Some(&"startpos") => Board::default(),
            Some(&"fen") => {
                let fen_parts: Vec<&str> = parts.iter().skip(1).take_while(|&&p| p != "moves").cloned().collect();
                let fen_string = fen_parts.join(" ");
                Board::from_str(&fen_string).unwrap_or(Board::default())
            }
            _ => return,
        };

        if let Some(moves_idx) = parts.iter().position(|&p| p == "moves") {
            for move_str in &parts[moves_idx + 1..] {
                if let Ok(mv) = ChessMove::from_str(move_str) {
                    current_board = current_board.make_move_new(mv);
                }
            }
        }

        self.board = current_board;
    }

    fn handle_go(&mut self, parts: &[&str]) {
        let mut depth = 5; // Default depth
        if let Some(depth_idx) = parts.iter().position(|&p| p == "depth") {
            if let Some(d_str) = parts.get(depth_idx + 1) {
                if let Ok(d) = d_str.parse() {
                    depth = d;
                }
            }
        }

        let mut searcher = SearchHandler::new(&self.engine, &self.tutor, &mut self.tt, self.board);
        if let (Some(best_move), _) = searcher.search(depth) {
            println!("bestmove {}", best_move);
        }
    }

    fn handle_bench(&mut self) {
        let mut searcher = SearchHandler::new(&self.engine, &self.tutor, &mut self.tt, Board::default());
        let start_time = std::time::Instant::now();

        searcher.search(8); // A reasonable depth for a quick benchmark

        let duration = start_time.elapsed();
        if duration.as_secs_f64() > 0.0 {
            let nps = searcher.nodes_searched as f64 / duration.as_secs_f64();
            println!("Nodes per second: {}", nps as u64);
        } else {
            println!("Bench finished too quickly to calculate NPS.");
        }
    }
}
