// src/uci.rs
//
// UCI (Universal Chess Interface) protocol handler for KiyEngine V3.
// Enhanced for multi-core scaling with Lazy SMP.

use crate::engine::Engine;
use crate::search::{SearchHandler, TimeControl};
use crate::tt::TranspositionTable;
use crate::tutor::Tutor;
use chess::{Board, ChessMove};
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const TT_SIZE_MB: usize = 512;

pub struct UciHandler {
    engine: Arc<Engine>,
    tutor: Arc<Tutor>,
    board: Board,
    tt: Arc<TranspositionTable>,
    move_history: Vec<usize>,
    debug_mode: bool,
    num_threads: usize,
}

impl UciHandler {
    pub fn new() -> anyhow::Result<Self> {
        let engine = Arc::new(Engine::new()?);
        let tutor = Arc::new(Tutor::new());
        Ok(Self {
            engine,
            tutor,
            board: Board::default(),
            tt: Arc::new(TranspositionTable::new(TT_SIZE_MB)),
            move_history: Vec::new(),
            debug_mode: true,
            num_threads: 1,
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
            _ => (),
        }
    }

    fn handle_uci(&self) {
        println!("id name KiyEngine V3 Mamba-MoE");
        println!("id author Khoi");
        println!();
        println!("option name Hash type spin default 512 min 1 max 4096");
        println!("option name Threads type spin default 1 min 1 max 64");
        println!("option name Debug type check default true");
        println!("uciok");
    }

    fn handle_debug(&mut self, parts: &[&str]) {
        if let Some(&"on") = parts.first() {
            self.debug_mode = true;
        } else if let Some(&"off") = parts.first() {
            self.debug_mode = false;
        }
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_setoption(&mut self, parts: &[&str]) {
        let name_idx = parts.iter().position(|&p| p == "name");
        let value_idx = parts.iter().position(|&p| p == "value");

        if let (Some(ni), Some(vi)) = (name_idx, value_idx) {
            let name: String = parts[ni + 1..vi].join(" ");
            let value = parts.get(vi + 1).copied().unwrap_or("");

            match name.as_str() {
                "Hash" => {
                    if let Ok(size) = value.parse::<usize>() {
                        self.tt = Arc::new(TranspositionTable::new(size));
                    }
                }
                "Threads" => {
                    if let Ok(threads) = value.parse::<usize>() {
                        self.num_threads = threads;
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
    }

    fn handle_position(&mut self, parts: &[&str]) {
        self.move_history.clear();

        let mut current_board = match parts.first() {
            Some(&"startpos") => Board::default(),
            Some(&"fen") => {
                let fen_parts: Vec<&str> = parts
                    .iter()
                    .skip(1)
                    .take_while(|&&p| p != "moves")
                    .copied()
                    .collect();
                Board::from_str(&fen_parts.join(" ")).unwrap_or(Board::default())
            }
            _ => return,
        };

        if let Some(moves_idx) = parts.iter().position(|&p| p == "moves") {
            for move_str in &parts[moves_idx + 1..] {
                if let Ok(mv) = ChessMove::from_str(move_str) {
                    let token = crate::search::move_to_token(&mv, &current_board);
                    self.move_history.push(token);
                    current_board = current_board.make_move_new(mv);
                }
            }
        }
        self.board = current_board;
    }

    fn handle_go(&mut self, parts: &[&str]) {
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
                _ => i += 1,
            }
        }

        if tc.depth.is_none()
            && tc.wtime.is_none()
            && tc.btime.is_none()
            && tc.movetime.is_none()
            && !tc.infinite
        {
            tc.depth = Some(8);
        }

        self.tt.increment_age();

        let num_threads = self.num_threads;
        let board = self.board;
        let move_history = self.move_history.clone();
        let engine = Arc::clone(&self.engine);
        let tutor = Arc::clone(&self.tutor);
        let tt = Arc::clone(&self.tt);
        let stop_flag = Arc::new(AtomicBool::new(false));

        thread::scope(|s| {
            for thread_id in 0..num_threads {
                let engine = Arc::clone(&engine);
                let tutor = Arc::clone(&tutor);
                let tt = Arc::clone(&tt);
                let tc = &tc;
                let move_history = move_history.clone();
                let stop_flag = Arc::clone(&stop_flag);

                s.spawn(move || {
                    let mut searcher =
                        SearchHandler::new(&engine, &tutor, &tt, board, stop_flag, thread_id);
                    for &token in &move_history {
                        searcher.game_state.push_token(token);
                    }

                    let (best_move, _) = searcher.search_with_time(tc);

                    if thread_id == 0 {
                        if let Some(mv) = best_move {
                            println!("bestmove {}", mv);
                        } else {
                            let legal_moves = chess::MoveGen::new_legal(&board);
                            if let Some(mv) = legal_moves.into_iter().next() {
                                println!("bestmove {}", mv);
                            }
                        }
                    }
                });
            }
        });
    }

    fn handle_bench(&mut self) {
        println!("info string Running benchmark (depth 8, 1 thread)...");
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];

        let mut total_nodes = 0u64;
        let start_time = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));

        for fen in &positions {
            let board = Board::from_str(fen).unwrap_or(Board::default());
            let mut searcher = SearchHandler::new(
                &self.engine,
                &self.tutor,
                &self.tt,
                board,
                Arc::clone(&stop_flag),
                0,
            );
            let tc = TimeControl {
                depth: Some(8),
                ..Default::default()
            };
            searcher.search_with_time(&tc);
            total_nodes += searcher.nodes_searched;
        }

        let duration = start_time.elapsed();
        let nps = (total_nodes as f64 / duration.as_secs_f64()) as u64;
        println!("info string Benchmark NPS: {}", nps);
    }

    fn handle_eval(&self) {
        let eval = self.tutor.evaluate(&self.board);
        println!("info string Static eval: {}cp", eval);
    }
}
