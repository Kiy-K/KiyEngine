// src/uci/mod.rs

use crate::book::OpeningBook;
use crate::engine::{Engine, KVCache};
use crate::search::{Searcher, TranspositionTable};
use crate::search::syzygy::SyzygyTB;
use chess::{Board, ChessMove, MoveGen, Square};
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

pub struct MoveCodec;

impl MoveCodec {
    // ---------------------------------------------------------------
    // 72-plane policy encoding (4608 = 72 planes × 64 from-squares)
    //
    //   Planes  0-55: Queen-like moves (8 directions × 7 distances)
    //           direction_id * 7 + (distance - 1)
    //           Directions: N=0 NE=1 E=2 SE=3 S=4 SW=5 W=6 NW=7
    //           Queen promotions land here (pawn advance 1 sq to last rank).
    //
    //   Planes 56-63: Knight moves (8 L-shapes)
    //           NNE=56 ENE=57 ESE=58 SSE=59 SSW=60 WSW=61 WNW=62 NNW=63
    //
    //   Planes 64-71: Under-promotions
    //           64-66: Knight promo (left-diag, straight, right-diag)
    //           67-69: Bishop promo (left-diag, straight, right-diag)
    //           70-71: Rook   promo (left-diag, right-diag) [straight dropped]
    //
    //   Formula: index = plane * 64 + from_square
    // ---------------------------------------------------------------

    /// Encode a chess move into a 72-plane policy index (0..4607).
    /// This is used for BOTH input-token embedding AND policy-head lookup.
    pub fn move_to_token(mv: &ChessMove) -> u32 {
        let from = mv.get_source().to_int() as i32;
        let to = mv.get_dest().to_int() as i32;

        let df = (to % 8) - (from % 8); // delta file
        let dr = (to / 8) - (from / 8); // delta rank

        let plane: i32 = if let Some(promo) = mv.get_promotion() {
            if promo == chess::Piece::Queen {
                // Queen promotion → queen-like plane (distance 1 in some direction)
                let dir = Self::queen_direction(df, dr);
                dir * 7 // distance is always 1, so (1-1)=0
            } else {
                // Under-promotion: determine direction index
                //   left-diag=0 (df<0), straight=1 (df==0), right-diag=2 (df>0)
                let dir_idx = if df < 0 {
                    0
                } else if df == 0 {
                    1
                } else {
                    2
                };
                match promo {
                    chess::Piece::Knight => 64 + dir_idx,
                    chess::Piece::Bishop => 67 + dir_idx,
                    chess::Piece::Rook => {
                        // Only left(70) and right(71); straight maps to 70 as fallback
                        if df < 0 {
                            70
                        } else {
                            71
                        }
                    }
                    _ => 64 + dir_idx,
                }
            }
        } else if df.abs() * dr.abs() == 2 {
            // Knight move (one delta is ±1, other is ±2 → product is 2)
            56 + Self::knight_index(df, dr)
        } else {
            // Queen-like (sliding) move
            let dir = Self::queen_direction(df, dr);
            let dist = df.abs().max(dr.abs());
            dir * 7 + (dist - 1)
        };

        (plane * 64 + from) as u32
    }

    /// Decode a token back to a chess move by matching against legal moves.
    pub fn token_to_move(token: u32, board: &Board) -> Option<ChessMove> {
        if token >= 4608 {
            return None;
        }
        let move_gen = MoveGen::new_legal(board);
        for mv in move_gen {
            if Self::move_to_token(&mv) == token {
                return Some(mv);
            }
        }
        None
    }

    /// Convenience alias used in move-ordering code for clarity.
    #[inline]
    pub fn get_policy_index(mv: &ChessMove) -> usize {
        Self::move_to_token(mv) as usize
    }

    // --- private helpers ---------------------------------------------------

    /// Map (delta_file, delta_rank) to one of 8 queen directions (0..7).
    ///   N=0  NE=1  E=2  SE=3  S=4  SW=5  W=6  NW=7
    #[inline]
    fn queen_direction(df: i32, dr: i32) -> i32 {
        match (df.signum(), dr.signum()) {
            (0, 1) => 0,   // N
            (1, 1) => 1,   // NE
            (1, 0) => 2,   // E
            (1, -1) => 3,  // SE
            (0, -1) => 4,  // S
            (-1, -1) => 5, // SW
            (-1, 0) => 6,  // W
            (-1, 1) => 7,  // NW
            _ => 0,        // fallback (should never happen for legal moves)
        }
    }

    /// Map a knight (df,dr) to index 0..7.
    ///   NNE(+1,+2)=0  ENE(+2,+1)=1  ESE(+2,-1)=2  SSE(+1,-2)=3
    ///   SSW(-1,-2)=4  WSW(-2,-1)=5  WNW(-2,+1)=6  NNW(-1,+2)=7
    #[inline]
    fn knight_index(df: i32, dr: i32) -> i32 {
        match (df, dr) {
            (1, 2) => 0,
            (2, 1) => 1,
            (2, -1) => 2,
            (1, -2) => 3,
            (-1, -2) => 4,
            (-2, -1) => 5,
            (-2, 1) => 6,
            (-1, 2) => 7,
            _ => 0,
        }
    }
}

pub struct UciHandler {
    searcher: Searcher,
    tt: Arc<TranspositionTable>,
    book: OpeningBook,
    board: Board,
    move_history: Vec<usize>,   // For NN Context (Tokens)
    position_history: Vec<u64>, // For Repetition (Zobrist Hashes)
    stop_flag: Arc<AtomicBool>,
    tx_move: mpsc::Sender<String>,
    analysis_mode: bool,
    multipv: usize,
    show_wdl: bool,
    debug_mode: bool,
    ponder_mode: bool,
    /// KV cache for incremental NN inference (persists across moves)
    kv_cache: Arc<parking_lot::Mutex<KVCache>>,
}

impl UciHandler {
    pub fn new() -> anyhow::Result<Self> {
        let engine = Arc::new(Engine::new()?);
        let tt = Arc::new(TranspositionTable::new(512));
        let mut searcher = Searcher::new(Arc::clone(&engine), Arc::clone(&tt));

        // Try loading NNUE evaluation network
        let nnue_path = crate::nnue::DEFAULT_NNUE_PATH;
        if std::path::Path::new(nnue_path).exists() {
            match crate::nnue::NnueNetwork::load(nnue_path) {
                Ok(net) => {
                    println!("info string NNUE loaded: {}", nnue_path);
                    searcher.nnue = Some(std::sync::Arc::new(net));
                }
                Err(e) => {
                    eprintln!("info string NNUE load failed: {}", e);
                }
            }
        }

        let book = OpeningBook::load(&["src/book/book1.bin", "src/book/book2.bin"]);
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                println!("{}", msg);
            }
        });

        let num_layers = engine.num_layers();
        Ok(Self {
            searcher,
            tt,
            book,
            board: Board::default(),
            move_history: Vec::new(),
            position_history: Vec::new(),
            stop_flag: Arc::new(AtomicBool::new(false)),
            tx_move: tx,
            analysis_mode: false,
            multipv: 1,
            show_wdl: false,
            debug_mode: false,
            ponder_mode: false,
            kv_cache: Arc::new(parking_lot::Mutex::new(KVCache::new(num_layers))),
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
                println!("id name KiyEngine V6.0.0");
                println!("id author Khoi");
                println!("option name Hash type spin default 512 min 64 max 65536");
                println!("option name Threads type spin default 4 min 1 max 256");
                println!("option name Move Overhead type spin default 60 min 0 max 5000");
                println!("option name Analysis Mode type check default false");
                println!("option name MultiPV type spin default 1 min 1 max 500");
                println!("option name UCI_ShowWDL type check default false");
                println!("option name UCI_Ponder type check default false");
                println!("option name Clear Hash type button");
                println!("option name SyzygyPath type string default <empty>");
                println!("uciok");
            }
            Some("debug") => {
                if parts.get(1) == Some(&"on") {
                    self.debug_mode = true;
                } else if parts.get(1) == Some(&"off") {
                    self.debug_mode = false;
                }
            }
            Some("isready") => {
                println!("readyok");
            }
            Some("ucinewgame") => {
                self.board = Board::default();
                self.move_history.clear();
                self.position_history.clear();
                self.tt.clear();
                self.kv_cache.lock().clear();
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
            Some("ponderhit") => {
                // Switch from pondering to normal search timing
                self.ponder_mode = false;
                // The search is already running; just let time management take over
            }
            Some("setoption") => {
                self.handle_setoption(&parts[1..]);
            }
            Some("d") => {
                self.display_board();
            }
            Some("quit") => {
                std::process::exit(0);
            }
            _ => {}
        }
    }

    fn handle_setoption(&mut self, parts: &[&str]) {
        if parts.len() < 2 || parts[0] != "name" {
            return;
        }

        // Find the "value" keyword to split name and value
        let mut value_idx = 0;
        for (i, &part) in parts.iter().enumerate() {
            if part == "value" {
                value_idx = i;
                break;
            }
        }

        // Button-type options have no "value" keyword
        if value_idx == 0 {
            let name = parts[1..].join(" ").to_lowercase();
            match name.as_str() {
                "clear hash" => {
                    self.tt.clear();
                }
                _ => {}
            }
            return;
        }

        let name = parts[1..value_idx].join(" ").to_lowercase();
        let value = if value_idx + 1 < parts.len() {
            parts[(value_idx + 1)..].join(" ")
        } else {
            return;
        };

        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.tt.resize(mb);
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
            "uci_ponder" => {
                self.ponder_mode = value == "true";
            }
            "syzygypath" => {
                if !value.is_empty() && value != "<empty>" {
                    match SyzygyTB::new(&value) {
                        Ok(tb) => {
                            println!("info string Syzygy TB loaded: {} (max {} pieces)", value, tb.max_pieces());
                            self.searcher.syzygy = Some(std::sync::Arc::new(tb));
                        }
                        Err(e) => {
                            eprintln!("info string Syzygy TB load failed: {}", e);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn display_board(&self) {
        // Print board in a human-readable format
        let fen = format!("{}", self.board);
        println!("\n {}", fen);
        println!(" Side to move: {:?}", self.board.side_to_move());
        println!(" Hash: {:016x}", self.board.get_hash());
        println!(" Checkers: {:?}", self.board.checkers());
        println!();
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

        // Parse moves
        if parts.get(i) == Some(&"moves") {
            i += 1;
            // Record initial position hash
            self.position_history.push(self.board.get_hash());

            while i < parts.len() {
                let move_str = parts[i];
                if let Some(mv) = Self::parse_move(move_str, &self.board) {
                    let token = MoveCodec::move_to_token(&mv) as usize;
                    self.move_history.push(token);
                    self.board = self.board.make_move_new(mv);
                    self.position_history.push(self.board.get_hash());
                }
                i += 1;
            }
        }
    }

    fn parse_move(s: &str, board: &Board) -> Option<ChessMove> {
        if s.len() < 4 {
            return None;
        }
        let from = Square::from_str(&s[0..2]).ok()?;
        let to = Square::from_str(&s[2..4]).ok()?;
        let promo = if s.len() > 4 {
            match &s[4..5] {
                "q" => Some(chess::Piece::Queen),
                "r" => Some(chess::Piece::Rook),
                "b" => Some(chess::Piece::Bishop),
                "n" => Some(chess::Piece::Knight),
                _ => None,
            }
        } else {
            None
        };

        let mv = ChessMove::new(from, to, promo);
        if board.legal(mv) {
            return Some(mv);
        }
        // Try auto-promotion to queen
        if promo.is_none() {
            let mv_q = ChessMove::new(from, to, Some(chess::Piece::Queen));
            if board.legal(mv_q) {
                return Some(mv_q);
            }
        }
        None
    }

    fn handle_go(&mut self, parts: &[&str]) {
        let mut wtime: u32 = 0;
        let mut btime: u32 = 0;
        let mut winc: u32 = 0;
        let mut binc: u32 = 0;
        let mut movestogo: u32 = 0;
        let mut depth: u8 = 0;
        let mut movetime: u32 = 0;
        let mut _nodes: u64 = 0;
        let mut _mate: u32 = 0;
        let mut infinite = false;
        let mut _ponder = false;
        let mut _searchmoves: Vec<ChessMove> = Vec::new();

        let mut i = 0;
        while i < parts.len() {
            match parts[i] {
                "wtime" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        wtime = v;
                    }
                    i += 2;
                }
                "btime" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        btime = v;
                    }
                    i += 2;
                }
                "winc" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        winc = v;
                    }
                    i += 2;
                }
                "binc" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        binc = v;
                    }
                    i += 2;
                }
                "movestogo" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        movestogo = v;
                    }
                    i += 2;
                }
                "depth" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        depth = v;
                    }
                    i += 2;
                }
                "nodes" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        _nodes = v;
                    }
                    i += 2;
                }
                "mate" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        _mate = v;
                    }
                    i += 2;
                }
                "movetime" => {
                    if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                        movetime = v;
                    }
                    i += 2;
                }
                "infinite" => {
                    infinite = true;
                    i += 1;
                }
                "ponder" => {
                    _ponder = true;
                    i += 1;
                }
                "searchmoves" => {
                    // Collect all subsequent move strings until next keyword or end
                    i += 1;
                    while i < parts.len() {
                        if let Some(mv) = Self::parse_move(parts[i], &self.board) {
                            _searchmoves.push(mv);
                            i += 1;
                        } else {
                            break;
                        }
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }

        // Handle movetime: convert to wtime/btime with movestogo=1
        if movetime > 0 {
            wtime = movetime;
            btime = movetime;
            winc = 0;
            binc = 0;
            movestogo = 1;
        }

        // Handle infinite: set very large time so search runs until "stop"
        if infinite {
            wtime = u32::MAX / 2;
            btime = u32::MAX / 2;
        }

        self.stop_flag.store(false, Ordering::SeqCst);

        // --- Book probe: instant move if in book ---
        if !self.analysis_mode {
            if let Some(book_move) = self.book.probe(&self.board, self.move_history.len()) {
                eprintln!("Book move: {}", book_move);
                let _ = self.tx_move.send(format!("bestmove {}", book_move));
                return;
            }
        }

        self.searcher.search_async(
            self.board.clone(),
            self.move_history.clone(),
            depth,
            wtime,
            btime,
            winc,
            binc,
            movestogo,
            Arc::clone(&self.stop_flag),
            self.tx_move.clone(),
            Some(Arc::clone(&self.kv_cache)),
            self.position_history.clone(),
        );
    }
}
