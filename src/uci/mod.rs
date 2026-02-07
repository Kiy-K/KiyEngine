// src/uci/mod.rs

use crate::book::OpeningBook;
use crate::engine::Engine;
use crate::search::lazy_smp::Searcher;
use crate::search::tt::AtomicTT;
use chess::{Board, ChessMove, MoveGen, Square};
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

pub mod mcts_handler;

pub use mcts_handler::MCTSUciHandler;

pub struct MoveCodec;

impl MoveCodec {
    pub fn move_to_token(mv: &ChessMove) -> u32 {
        let from = mv.get_source().to_int();
        let to = mv.get_dest().to_int();
        let base_token = (from as u32 * 64) + to as u32;

        if let Some(promo) = mv.get_promotion() {
            let piece_idx = match promo {
                chess::Piece::Knight => 0,
                chess::Piece::Bishop => 1,
                chess::Piece::Rook => 2,
                chess::Piece::Queen => 3,
                _ => 0,
            };
            4096 + (piece_idx * 64) + (to as u32 % 64)
        } else {
            base_token
        }
    }

    pub fn token_to_move(token: u32, board: &Board) -> Option<ChessMove> {
        if token < 4096 {
            let from_idx = token / 64;
            let to_idx = token % 64;
            let from = unsafe {
                // SAFETY: from_idx is modulo 64 from token/64, so it's always 0-63
                Square::new(from_idx as u8)
            };
            let to = unsafe {
                // SAFETY: to_idx is modulo 64 from token%64, so it's always 0-63
                Square::new(to_idx as u8)
            };

            let mv = ChessMove::new(from, to, None);
            if board.legal(mv) {
                return Some(mv);
            }
            let mv_q = ChessMove::new(from, to, Some(chess::Piece::Queen));
            if board.legal(mv_q) {
                return Some(mv_q);
            }
        } else if token < 4608 {
            let promo_part = token - 4096;
            let piece_idx = promo_part / 64;
            let to_idx = promo_part % 64;
            let to = unsafe {
                // SAFETY: to_idx is modulo 64 from token%64, so it's always 0-63
                Square::new(to_idx as u8)
            };

            let piece = match piece_idx {
                0 => chess::Piece::Knight,
                1 => chess::Piece::Bishop,
                2 => chess::Piece::Rook,
                3 => chess::Piece::Queen,
                _ => chess::Piece::Queen,
            };

            let move_gen = MoveGen::new_legal(board);
            for mv in move_gen {
                if mv.get_dest() == to && mv.get_promotion() == Some(piece) {
                    return Some(mv);
                }
            }
        }
        None
    }
}

pub struct UciHandler {
    searcher: Searcher,
    tt: Arc<AtomicTT>,
    _book: OpeningBook,
    board: Board,
    move_history: Vec<usize>,   // For NN Context (Tokens)
    position_history: Vec<u64>, // For Repetition (Zobrist Hashes)
    stop_flag: Arc<AtomicBool>,
    tx_move: mpsc::Sender<String>,
    analysis_mode: bool,
    multipv: usize,
    show_wdl: bool,
}

impl UciHandler {
    pub fn new() -> anyhow::Result<Self> {
        let engine = Arc::new(Engine::new()?);
        let tt = Arc::new(TranspositionTable::new(512));
        let searcher = Searcher::new(Arc::clone(&engine), Arc::clone(&tt));
        let book = OpeningBook::open("book.bin");
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                println!("{}", msg);
            }
        });

        Ok(Self {
            searcher,
            tt,
            _book: book,
            board: Board::default(),
            move_history: Vec::new(),
            position_history: Vec::new(),
            stop_flag: Arc::new(AtomicBool::new(false)),
            tx_move: tx,
            analysis_mode: false,
            multipv: 1,
            show_wdl: false,
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
                println!("id name KiyEngine V4.4.5 Titan");
                println!("id author Khoi");
                println!("option name Hash type spin default 64 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!("option name Move Overhead type spin default 30 min 0 max 5000");
                println!("option name Analysis Mode type check default false");
                println!("option name MultiPV type spin default 1 min 1 max 500");
                println!("option name UCI_ShowWDL type check default false");
                println!("uciok");
            }
            Some("isready") => {
                println!("readyok");
            }
            Some("ucinewgame") => {
                self.board = Board::default();
                self.move_history.clear();
                self.position_history.clear();
                self.tt.clear();
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
                if parts.len() >= 5
                    && parts[1] == "name"
                    && parts[2] == "Hash"
                    && parts[3] == "value"
                {
                    if let Ok(mb) = parts[4].parse() {
                        self.tt.resize(mb);
                    }
                } else if parts.len() >= 5
                    && parts[1] == "name"
                    && parts[2] == "Threads"
                    && parts[3] == "value"
                {
                    if let Ok(num) = parts[4].parse() {
                        self.searcher.threads = num;
                    }
                } else if parts.len() >= 5
                    && parts[1] == "name"
                    && parts[2] == "Move Overhead"
                    && parts[3] == "value"
                {
                    if let Ok(ov) = parts[4].parse() {
                        self.searcher.move_overhead = ov;
                    }
                }
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
            "hash" => {
                if let Ok(_mb) = value.parse::<usize>() {
                    // Should resize TT here
                    self.tt.clear();
                }
            }
            "threads" => {
                if let Ok(num) = value.parse::<usize>() {
                    self.searcher.threads = num.max(1);
                }
            }
            "move overhead" => {
                if let Ok(ov) = value.parse::<u32>() {
                    self.searcher.move_overhead = ov;
                }
            }
            "analysis mode" => {
                self.analysis_mode = value == "true";
            }
            "multipv" => {
                if let Ok(mpv) = value.parse::<usize>() {
                    self.multipv = mpv;
                }
            }
            "uci_showwdl" => {
                self.show_wdl = value == "true";
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
