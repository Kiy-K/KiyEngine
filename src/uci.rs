// src/uci.rs

use crate::engine::Engine;
use crate::search::{Searcher, TranspositionTable};
use chess::{Board, ChessMove, MoveGen, Square};
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};

/// MoveCodec handles conversion between UCI strings, ChessMoves, and Model Tokens.
pub struct MoveCodec;

impl MoveCodec {
    /// Encodes a move to a token ID for the model.
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

    /// Decodes a token ID from the model to a ChessMove.
    pub fn token_to_move(token: u32, board: &Board) -> Option<ChessMove> {
        if token < 4096 {
            let from_idx = token / 64;
            let to_idx = token % 64;
            let from = unsafe { Square::new(from_idx as u8) };
            let to = unsafe { Square::new(to_idx as u8) };
            
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
            let to = unsafe { Square::new(to_idx as u8) };
            
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
    tt: Arc<TranspositionTable>,
    board: Board,
    move_history: Vec<usize>,
    stop_flag: Arc<AtomicBool>,
    tx_move: mpsc::Sender<String>,
}

impl UciHandler {
    pub fn new() -> anyhow::Result<Self> {
        let engine = Arc::new(Engine::new()?);
        let tt = Arc::new(TranspositionTable::new());
        let searcher = Searcher::new(Arc::clone(&engine), Arc::clone(&tt));
        let (tx, rx) = mpsc::channel();
        
        // Output thread for bestmove messages
        std::thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                println!("{}", msg);
            }
        });
        
        Ok(Self {
            searcher,
            tt,
            board: Board::default(),
            move_history: Vec::new(),
            stop_flag: Arc::new(AtomicBool::new(false)),
            tx_move: tx,
        })
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(cmd) = line else { break };
            let cmd = cmd.trim();
            if cmd.is_empty() { continue; }
            self.handle_command(cmd);
        }
    }

    fn handle_command(&mut self, command: &str) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        match parts.get(0).copied() {
            Some("uci") => {
                println!("id name KiyEngine V4 Omega");
                println!("id author Khoi");
                println!("uciok");
            }
            Some("isready") => {
                println!("readyok");
            }
            Some("ucinewgame") => {
                self.board = Board::default();
                self.move_history.clear();
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
            Some("quit") => {
                std::process::exit(0);
            }
            _ => {}
        }
    }

    fn handle_position(&mut self, parts: &[&str]) {
        self.move_history.clear();
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

        if parts.get(i) == Some(&"moves") {
            i += 1;
            for move_str in &parts[i..] {
                if let Ok(mv) = ChessMove::from_str(move_str) {
                    let token = MoveCodec::move_to_token(&mv);
                    self.move_history.push(token as usize);
                    self.board = self.board.make_move_new(mv);
                }
            }
        }
    }

    fn handle_go(&mut self, parts: &[&str]) {
        self.stop_flag.store(false, Ordering::SeqCst);
        
        let mut depth = 8; // Default depth for the scout
        if let Some(d_idx) = parts.iter().position(|&p| p == "depth") {
            if let Some(d_str) = parts.get(d_idx + 1) {
                if let Ok(d) = d_str.parse() {
                    depth = d;
                }
            }
        }

        self.searcher.search_async(
            self.board,
            self.move_history.clone(),
            depth,
            Arc::clone(&self.stop_flag),
            self.tx_move.clone(),
        );
    }
}
