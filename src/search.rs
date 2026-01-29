// src/search.rs
//
// Asymmetric Scout Search: Combines Mamba-MoE brain with lightweight Alpha-Beta scout.
// Enhanced for Stockfish match with Hard-Truth filtering and time management.

use std::time::{Duration, Instant};

use crate::engine::{Engine, GameState};
use crate::tutor::Tutor;
use crate::tt::{TTEntry, TTFlag, TranspositionTable};
use chess::{Board, ChessMove, MoveGen, Piece, EMPTY};

// Constants for search evaluation in centipawns
const MATE_SCORE: i32 = 30_000;
const INFINITY: i32 = 32_000;

// Time management constants
const OVERHEAD_MS: u64 = 50;        // FFI/UCI communication overhead buffer
const PANIC_TIME_MULT: f64 = 2.5;   // Time multiplier when blunder detected
const PANIC_THRESHOLD: i32 = 200;   // Eval drop triggering panic mode (centipawns)

// Scout configuration
const SCOUT_DEPTH_MIN: u8 = 4;      // Minimum scout verification depth
const SCOUT_DEPTH_MAX: u8 = 24;      // Maximum scout verification depth (user requested ~30, governed by node limit)
const BLUNDER_THRESHOLD: i32 = 150; // Centipawns drop threshold for veto
const TOP_N_CANDIDATES: usize = 5;  // Number of neural candidates to verify

/// Result of scout verification for a single move
#[derive(Debug, Clone)]
pub struct ScoutResult {
    pub mv: ChessMove,
    pub neural_prob: f32,
    pub scout_eval: i32,
    pub is_vetoed: bool,
    pub veto_reason: Option<String>,
}

/// Maps a `ChessMove` to a token index for the model's embedding layer.
/// Token = piece_index * 64 + destination_square
pub fn move_to_token(mv: &ChessMove, board: &Board) -> usize {
    let piece = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
    let piece_idx = mv.get_promotion().map_or_else(
        || piece.to_index(),
        |promo_piece| promo_piece.to_index(),
    );
    let to_square_idx = mv.get_dest().to_int() as usize;
    piece_idx * 64 + to_square_idx
}

/// Maps a token index back to possible moves (used for policy extraction).
#[allow(dead_code)]
pub fn token_to_moves(token: usize, board: &Board) -> Vec<ChessMove> {
    let piece_idx = token / 64;
    let dest_sq = token % 64;

    let legal_moves = MoveGen::new_legal(board);
    legal_moves
        .filter(|mv| {
            let mv_piece = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
            let mv_dest = mv.get_dest().to_int() as usize;

            // Check if promotion matches
            if let Some(promo) = mv.get_promotion() {
                promo.to_index() == piece_idx && mv_dest == dest_sq
            } else {
                mv_piece.to_index() == piece_idx && mv_dest == dest_sq
            }
        })
        .collect()
}

/// The Asymmetric Scout: Hard-Truth tactical verifier using Alpha-Beta + Quiescence.
pub struct Scout<'a> {
    tutor: &'a Tutor,
    tt: &'a mut TranspositionTable,
    scout_depth: u8,
    blunder_threshold: i32,
    nodes_searched: u64,
    killers: [[Option<ChessMove>; 2]; 64],
    ply: usize,
}

impl<'a> Scout<'a> {
    pub fn new(tutor: &'a Tutor, tt: &'a mut TranspositionTable, scout_depth: u8, blunder_threshold: i32) -> Self {
        Self {
            tutor,
            tt,
            scout_depth,
            blunder_threshold,
            nodes_searched: 0,
            killers: [[None; 2]; 64],
            ply: 0,
        }
    }

    /// Hard-Truth verification: paranoid filter for neural move candidates.
    /// Uses Iterative Deepening to reach maximum depth within limits.
    pub fn hard_truth_filter(
        &mut self,
        board: &Board,
        candidates: &[(ChessMove, f32)],
    ) -> Vec<ScoutResult> {
        let static_eval = self.tutor.evaluate(board);
        self.nodes_searched = 0;

        let mut results: Vec<ScoutResult> = Vec::with_capacity(candidates.len());
        
        // Resource limit per move verification to avoid stalling
        // For depth 30, we rely on pruning. Safety valve: 100k nodes per candidate.
        let node_limit = 100_000;

        for (mv, neural_prob) in candidates {
            let mut scout_eval = static_eval; // Default to static if search fails
            
            // Iterative Deepening for the Scout Search
            let new_board = board.make_move_new(*mv);
            let start_nodes = self.nodes_searched;
            
            // Reset search state for this candidate
            self.killers = [[None; 2]; 64];
            self.ply = 0;

            for d in 1..=self.scout_depth {
                // Alpha-Beta search
                let score = -self.alpha_beta(&new_board, d, -INFINITY, INFINITY);
                
                scout_eval = score;
                
                // Break if we've used too many nodes for this verification
                if self.nodes_searched - start_nodes > node_limit {
                    break;
                }
            }

            // Blunder detection
            let eval_drop = static_eval - scout_eval;
            let is_vetoed = eval_drop > self.blunder_threshold;

            let veto_reason = if is_vetoed {
                Some(format!("Tactical blunder: eval dropped {}cp", eval_drop))
            } else {
                None
            };

            results.push(ScoutResult {
                mv: *mv,
                neural_prob: *neural_prob,
                scout_eval,
                is_vetoed,
                veto_reason,
            });
        }

        results
    }

    /// Alpha-Beta search with NMP, LMR, and Razoring (Advanced)
    fn alpha_beta(&mut self, board: &Board, depth: u8, mut alpha: i32, mut beta: i32) -> i32 {
        self.nodes_searched += 1;
        let is_root = self.ply == 0; // Relative root for this search
        
        // 1. Mat/Stalemate Check
        let status = board.status();
        if status == chess::BoardStatus::Checkmate {
            return -MATE_SCORE + self.ply as i32;
        }
        if status == chess::BoardStatus::Stalemate {
            return 0;
        }

        // 2. Transposition Table Probe
        let hash = board.get_hash();
        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(hash) {
            tt_move = entry.best_move;
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::EXACT => return entry.score,
                    TTFlag::LOWERBOUND => alpha = alpha.max(entry.score),
                    TTFlag::UPPERBOUND => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
        }

        // 3. Quiescence Search at leaf
        if depth == 0 {
            return self.quiescence_search(board, alpha, beta);
        }

        let static_score = self.tutor.evaluate(board);
        let in_check = *board.checkers() != EMPTY;

        // 4. Razoring (Futility Pruning at low depth)
        // If we are way below alpha, just drop into qsearch
        if !in_check && depth <= 3 {
             let margin = 150 * (depth as i32);
             if static_score + margin < alpha {
                 let q_score = self.quiescence_search(board, alpha, beta);
                 if q_score < alpha {
                     return q_score;
                 }
             }
        }

        // 5. Null Move Pruning (NMP)
        // If we can pass and still beat beta, our position is too good. Cutoff.
        if depth >= 3 && !in_check && static_score >= beta && !is_root {
            // R = 2 usually, maybe 3 for high depths
            let r = 2 + (depth / 6); 
            if let Some(null_board) = board.null_move() {
                self.ply += 1;
                let null_score = -self.alpha_beta(&null_board, depth.saturating_sub(r), -beta, -beta + 1);
                self.ply -= 1;
                
                if null_score >= beta {
                    // Don't verify mate scores with NMP
                    if null_score < MATE_SCORE - 100 {
                         return beta;
                    }
                }
            }
        }

        // 6. Move Generation & Ordering
        let moves = MoveGen::new_legal(board);
        
        // Score moves for ordering
        let mut scored_moves: Vec<(ChessMove, i32)> = moves.map(|mv| {
            let mut score = 0;
            
            // TT Move
            if Some(mv) == tt_move {
                score = 1_000_000;
            } else if let Some(victim) = board.piece_on(mv.get_dest()) {
                 // MVV-LVA
                 let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                 score = 100_000 + crate::tutor::MG_VALUE[victim.to_index()] * 10 - crate::tutor::MG_VALUE[attacker.to_index()];
            } else {
                 // Killers
                 if self.ply < 64 {
                     if self.killers[self.ply][0] == Some(mv) { score = 90_000; }
                     else if self.killers[self.ply][1] == Some(mv) { score = 80_000; }
                 }
                 // History heuristic could go here
            }
            (mv, score)
        }).collect();
        
        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut tt_flag = TTFlag::UPPERBOUND;

        for (i, (mv, _)) in scored_moves.into_iter().enumerate() {
            let new_board = board.make_move_new(mv);
            self.ply += 1;
            
            // 7. Late Move Reduction (LMR)
            // Reduce depth for late moves that aren't captures/checks
            let is_capture = board.piece_on(mv.get_dest()).is_some();
            let is_promotion = mv.get_promotion().is_some();
            let gives_check = *new_board.checkers() != EMPTY;
            
            let mut score;
            
            if i >= 4 && depth >= 3 && !is_capture && !is_promotion && !in_check && !gives_check {
                let r = 1 + (depth / 6) + (i / 10) as u8;
                let d = depth.saturating_sub(r);
                
                // Search with reduced depth
                score = -self.alpha_beta(&new_board, d, -alpha - 1, -alpha);
                
                // Re-search if it improved alpha
                if score > alpha {
                    score = -self.alpha_beta(&new_board, depth - 1, -beta, -alpha);
                }
            } else {
                score = -self.alpha_beta(&new_board, depth - 1, -beta, -alpha);
            }

            self.ply -= 1;

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if best_score > alpha {
                alpha = best_score;
                tt_flag = TTFlag::EXACT;
            }

            if alpha >= beta {
                // Beta Cutoff
                if !is_capture && self.ply < 64 {
                    self.killers[self.ply][1] = self.killers[self.ply][0];
                    self.killers[self.ply][0] = Some(mv);
                }
                self.tt.store(TTEntry { key: hash, best_move: Some(mv), score: best_score, depth, flag: TTFlag::LOWERBOUND });
                return best_score;
            }
        }
        
        // No legal moves check handled at start (MoveGen would be empty logic logic needs adjustment? 
        // MoveGen::new_legal handles it, loop won't run. `best_score` remains -INFINITY.
        if best_score == -INFINITY {
             // If loop didn't run, check mate again (although done at top)
             if in_check { return -MATE_SCORE + self.ply as i32; }
             return 0; // Stalemate
        }

        self.tt.store(TTEntry { key: hash, best_move, score: best_score, depth, flag: tt_flag });
        best_score
    }

    /// Full Quiescence Search with TT support
    fn quiescence_search(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        self.nodes_searched += 1;

        // 1. Stand-pat
        let static_score = self.tutor.evaluate(board);
        if static_score >= beta {
            return beta;
        }
        if alpha < static_score {
            alpha = static_score;
        }

        // 2. Delta Pruning
        let big_delta = 900;
        if static_score + big_delta < alpha {
             return alpha;
        }

        // 3. Search Captures
        let moves = MoveGen::new_legal(board)
            .filter(|m| board.piece_on(m.get_dest()).is_some() || m.get_promotion().is_some());

        // Simple MVV-LVA Ordering
        let mut scored_moves: Vec<(ChessMove, i32)> = moves.map(|mv| {
            let victim = board.piece_on(mv.get_dest());
             let score = if let Some(v) = victim {
                 let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                 crate::tutor::MG_VALUE[v.to_index()] * 10 - crate::tutor::MG_VALUE[attacker.to_index()]
            } else { 500 };
            (mv, score)
        }).collect();
        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, see_score) in scored_moves {
            if see_score < 0 { continue; } // SEE Pruning

            let new_board = board.make_move_new(mv);
            let score = -self.quiescence_search(&new_board, -beta, -alpha);
             
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }

    pub fn get_nodes_searched(&self) -> u64 {
        self.nodes_searched
    }
}

/// Time management configuration for a single search
#[derive(Debug, Clone)]
pub struct TimeControl {
    pub wtime: Option<u64>,    // White time in ms
    pub btime: Option<u64>,    // Black time in ms
    pub winc: Option<u64>,     // White increment in ms
    pub binc: Option<u64>,     // Black increment in ms
    pub movestogo: Option<u32>, // Moves to go until time control
    pub movetime: Option<u64>, // Exact time for this move in ms
    pub infinite: bool,        // Infinite search (until "stop")
    pub depth: Option<u8>,     // Depth limit
}

impl Default for TimeControl {
    fn default() -> Self {
        Self {
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movestogo: None,
            movetime: None,
            infinite: false,
            depth: None,
        }
    }
}

impl TimeControl {
    /// Calculate allocated time for this move
    pub fn calculate_time(&self, is_white: bool) -> Duration {
        if self.infinite {
            return Duration::from_secs(3600); // 1 hour for infinite
        }

        if let Some(mt) = self.movetime {
            return Duration::from_millis(mt.saturating_sub(OVERHEAD_MS));
        }

        let our_time = if is_white { self.wtime } else { self.btime };
        let our_inc = if is_white { self.winc } else { self.binc };

        if let Some(time) = our_time {
            let inc = our_inc.unwrap_or(0);
            let moves_left = self.movestogo.unwrap_or(30) as u64;

            // Time formula: time/moves_left + increment - overhead
            let base_time = time / moves_left;
            let allocated = base_time + inc - OVERHEAD_MS;

            // Never use more than 50% of remaining time
            let max_time = time / 2;
            Duration::from_millis(allocated.min(max_time).max(100))
        } else {
            Duration::from_secs(5) // Default 5 seconds if no time control
        }
    }

    /// Calculate panic time (extended time when blunder detected)
    pub fn panic_time(&self, normal_time: Duration) -> Duration {
        Duration::from_millis((normal_time.as_millis() as f64 * PANIC_TIME_MULT) as u64)
    }
}

/// The main search handler combining Mamba brain with Scout verification.
pub struct SearchHandler<'a> {
    engine: &'a Engine,
    tutor: &'a Tutor,
    tt: &'a mut TranspositionTable,
    pub game_state: GameState,
    pub nodes_searched: u64,
    pub tt_hits: u64,
    killers: [[Option<ChessMove>; 2]; 64],
    ply: usize,
    start_time: Instant,
    time_limit: Duration,
    in_panic_mode: bool,
    last_eval: Option<i32>,
}

impl<'a> SearchHandler<'a> {
    pub fn new(engine: &'a Engine, tutor: &'a Tutor, tt: &'a mut TranspositionTable, board: Board) -> Self {
        let game_state = GameState::new(board);
        Self {
            engine,
            tutor,
            tt,
            game_state,
            nodes_searched: 0,
            tt_hits: 0,
            killers: [[None; 2]; 64],
            ply: 0,
            start_time: Instant::now(),
            time_limit: Duration::from_secs(10),
            in_panic_mode: false,
            last_eval: None,
        }
    }

    /// Main search function with time management.
    pub fn search_with_time(&mut self, tc: &TimeControl) -> (Option<ChessMove>, i32) {
        let is_white = self.game_state.board.side_to_move() == chess::Color::White;
        self.time_limit = tc.calculate_time(is_white);
        self.start_time = Instant::now();

        let max_depth = tc.depth.unwrap_or(64);
        self.search(max_depth)
    }

    /// Main search function using Asymmetric Scout pattern with Hard-Truth filter.
    pub fn search(&mut self, max_depth: u8) -> (Option<ChessMove>, i32) {
        self.start_time = Instant::now();
        self.nodes_searched = 0;
        self.tt_hits = 0;
        self.killers = [[None; 2]; 64];
        self.ply = 0;
        self.in_panic_mode = false;

        // Always use neural + scout for first move decision
        let neural_result = self.neural_hard_truth_search();

        // For very shallow depths, return neural + scout result directly
        if max_depth <= 3 {
            return neural_result;
        }

        // For deeper search, use iterative deepening with hybrid approach
        let mut best_move = neural_result.0;
        let mut best_score = neural_result.1;

        for depth in 1..=max_depth {
            // Check time
            if self.should_stop() {
                break;
            }

            let score = self.negamax(depth, -INFINITY, INFINITY);

            // Panic mode detection: sudden eval drop
            if let Some(last) = self.last_eval {
                if last - score > PANIC_THRESHOLD {
                    self.in_panic_mode = true;
                    let panic_time = Duration::from_millis(
                        (self.time_limit.as_millis() as f64 * PANIC_TIME_MULT) as u64
                    );
                    self.time_limit = panic_time;
                    println!("info string PANIC MODE: eval dropped {}cp, extending time", last - score);
                }
            }
            self.last_eval = Some(score);
            best_score = score;

            let elapsed = self.start_time.elapsed();
            let nps = (self.nodes_searched * 1000) / (elapsed.as_millis() as u64 + 1);

            if let Some(entry) = self.tt.probe(self.game_state.board.get_hash()) {
                if let Some(mv) = entry.best_move {
                    best_move = Some(mv);
                    println!(
                        "info depth {} score cp {} nodes {} nps {} time {} pv {}",
                        depth, score, self.nodes_searched, nps, elapsed.as_millis(), mv,
                    );
                }
            }
        }

        (best_move, best_score)
    }

    fn should_stop(&self) -> bool {
        self.start_time.elapsed() >= self.time_limit
    }

    /// Neural + Scout search with Hard-Truth filtering and UCI debug output.
    fn neural_hard_truth_search(&mut self) -> (Option<ChessMove>, i32) {
        let board = self.game_state.board;

        // Get policy from neural network
        let tokens = self.game_state.get_input_sequence();
        let (policy_logits, _neural_value) = match self.engine.forward(&tokens) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("info string Neural forward failed: {}", e);
                return self.fallback_scout_search();
            }
        };

        // Extract top-N moves from policy
        let candidates = self.extract_top_moves(&board, &policy_logits, TOP_N_CANDIDATES);

        if candidates.is_empty() {
            println!("info string No neural candidates, falling back to scout");
            return self.fallback_scout_search();
        }

        // Adaptive scout depth based on position complexity
        let in_check = *board.checkers() != EMPTY;
        let scout_depth = if in_check { SCOUT_DEPTH_MAX } else { SCOUT_DEPTH_MIN };

        // Hard-Truth verification
        let mut scout = Scout::new(self.tutor, self.tt, scout_depth, BLUNDER_THRESHOLD);
        let results = scout.hard_truth_filter(&board, &candidates);

        // UCI debug output
        for result in &results {
            let status = if result.is_vetoed { "VETOED" } else { "Accepted" };
            println!(
                "info string Mamba: {} (p={:.3}) | Scout: {}cp | {}{}",
                result.mv,
                result.neural_prob,
                result.scout_eval,
                status,
                result.veto_reason.as_ref().map(|r| format!(" ({})", r)).unwrap_or_default()
            );
        }

        self.nodes_searched += scout.get_nodes_searched();

        // Select best non-vetoed move
        let accepted: Vec<&ScoutResult> = results.iter().filter(|r| !r.is_vetoed).collect();

        if let Some(best) = accepted.first() {
            println!(
                "info string SELECTED: {} with scout eval {}cp (neural prob {:.3})",
                best.mv, best.scout_eval, best.neural_prob
            );
            return (Some(best.mv), best.scout_eval);
        }

        // All moves vetoed - pick the least bad one
        println!("info string WARNING: All neural candidates vetoed, selecting least bad");
        let mut all_results = results;
        all_results.sort_by(|a, b| b.scout_eval.cmp(&a.scout_eval));

        if let Some(least_bad) = all_results.first() {
            (Some(least_bad.mv), least_bad.scout_eval)
        } else {
            self.fallback_scout_search()
        }
    }

    /// Extracts top-N moves from policy logits with softmax normalization.
    fn extract_top_moves(&self, board: &Board, policy_logits: &candle_core::Tensor, top_n: usize) -> Vec<(ChessMove, f32)> {
        let logits: Vec<f32> = match policy_logits.to_vec1() {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        // Get all legal moves
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();

        // Map moves to their policy scores
        let mut scored_moves: Vec<(ChessMove, f32)> = legal_moves
            .into_iter()
            .map(|mv| {
                let token = move_to_token(&mv, board);
                let score = if token < logits.len() { logits[token] } else { f32::NEG_INFINITY };
                (mv, score)
            })
            .collect();

        // Softmax normalization over legal moves
        let max_score = scored_moves.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scored_moves.iter().map(|(_, s)| (s - max_score).exp()).sum();

        for (_, score) in &mut scored_moves {
            *score = (*score - max_score).exp() / exp_sum;
        }

        // Sort by probability descending
        scored_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-N
        scored_moves.into_iter().take(top_n).collect()
    }

    /// Fallback to pure scout/tutor search when neural network fails.
    fn fallback_scout_search(&mut self) -> (Option<ChessMove>, i32) {
        println!("info string Using fallback scout search");
        let board = self.game_state.board;

        let mut scout = Scout::new(self.tutor, self.tt, SCOUT_DEPTH_MIN, BLUNDER_THRESHOLD);
        let moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();

        let mut best_move = None;
        let mut best_score = -INFINITY;

        for mv in moves {
            let new_board = board.make_move_new(mv);
            let score = -scout.alpha_beta(&new_board, SCOUT_DEPTH_MIN - 1, -INFINITY, INFINITY);
            self.nodes_searched += scout.nodes_searched;
            scout.nodes_searched = 0;

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
        }

        (best_move, best_score)
    }

    #[allow(dead_code)]
    fn value_to_cp(value: f32) -> i32 {
        (value * 600.0) as i32
    }

    fn score_move(&self, mv: &ChessMove, tt_move: Option<ChessMove>) -> i32 {
        if Some(*mv) == tt_move {
            return 1_000_000;
        }

        let board = &self.game_state.board;
        if let Some(victim) = board.piece_on(mv.get_dest()) {
            let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
            // MVV-LVA using Tutor's MG_VALUE
            return 100_000 + crate::tutor::MG_VALUE[victim.to_index()] * 10 - crate::tutor::MG_VALUE[attacker.to_index()];
        }

        if self.ply < 64 {
            if self.killers[self.ply][0] == Some(*mv) {
                return 90_000;
            }
            if self.killers[self.ply][1] == Some(*mv) {
                return 80_000;
            }
        }

        0
    }

    fn negamax(&mut self, depth: u8, mut alpha: i32, mut beta: i32) -> i32 {
        // Time check every 1024 nodes
        if self.nodes_searched & 1023 == 0 && self.should_stop() {
            return 0;
        }

        self.nodes_searched += 1;
        let is_root = self.ply == 0;
        let hash = self.game_state.board.get_hash();

        // TT Probe
        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(hash) {
            tt_move = entry.best_move;
            if entry.depth >= depth && !is_root {
                self.tt_hits += 1;
                match entry.flag {
                    TTFlag::EXACT => return entry.score,
                    TTFlag::LOWERBOUND => alpha = alpha.max(entry.score),
                    TTFlag::UPPERBOUND => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
        }

        if self.game_state.board.status() == chess::BoardStatus::Checkmate {
            return -MATE_SCORE + self.ply as i32;
        }
        if self.game_state.board.status() == chess::BoardStatus::Stalemate {
            return 0;
        }

        if depth == 0 {
            return self.quiescence_search(alpha, beta);
        }

        // Static Eval
        let static_score = self.tutor.evaluate(&self.game_state.board);
        let in_check = *self.game_state.board.checkers() != EMPTY;

        // Futility Pruning (more aggressive)
        if depth <= 2 && !in_check && !is_root {
            let margin = 100 + 75 * (depth as i32);
            if static_score + margin < alpha {
                return static_score;
            }
        }

        // Null Move Pruning
        if depth >= 3 && !in_check && !is_root && static_score >= beta {
            if let Some(null_board) = self.game_state.board.null_move() {
                let board_backup = self.game_state.board;
                self.game_state.board = null_board;
                self.ply += 1;
                let r = 3 + depth / 6; // Adaptive reduction
                let score = -self.negamax(depth.saturating_sub(r), -beta, -beta + 1);
                self.ply -= 1;
                self.game_state.board = board_backup;

                if score >= beta {
                    return beta;
                }
            }
        }

        // Move Loop
        let moves = MoveGen::new_legal(&self.game_state.board);
        let mut scored_moves: Vec<(ChessMove, i32)> = moves.map(|mv| {
            let score = self.score_move(&mv, tt_move);
            (mv, score)
        }).collect();

        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut tt_flag = TTFlag::UPPERBOUND;

        for (i, (mv, _)) in scored_moves.into_iter().enumerate() {
            let board_backup = self.game_state.board;
            self.game_state.board = self.game_state.board.make_move_new(mv);

            self.ply += 1;
            let score = if i == 0 {
                -self.negamax(depth - 1, -beta, -alpha)
            } else {
                // Late Move Reduction (more aggressive)
                let is_tactical = board_backup.piece_on(mv.get_dest()).is_some()
                    || mv.get_promotion().is_some()
                    || (*self.game_state.board.checkers() != EMPTY);
                let reduction = if depth >= 3 && i >= 3 && !is_tactical && !in_check {
                    1 + (i / 6) as u8
                } else {
                    0
                };
                let new_depth = depth.saturating_sub(1 + reduction);
                let mut s = -self.negamax(new_depth, -alpha - 1, -alpha);
                if s > alpha && reduction > 0 {
                    s = -self.negamax(depth - 1, -beta, -alpha);
                }
                s
            };
            self.ply -= 1;
            self.game_state.board = board_backup;

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if best_score > alpha {
                alpha = best_score;
                tt_flag = TTFlag::EXACT;
            }

            if alpha >= beta {
                // Update Killers for quiet moves
                if board_backup.piece_on(mv.get_dest()).is_none() && self.ply < 64 {
                    if self.killers[self.ply][0] != Some(mv) {
                        self.killers[self.ply][1] = self.killers[self.ply][0];
                        self.killers[self.ply][0] = Some(mv);
                    }
                }
                self.tt.store(TTEntry { key: hash, best_move: Some(mv), score: best_score, depth, flag: TTFlag::LOWERBOUND });
                return best_score;
            }
        }

        self.tt.store(TTEntry { key: hash, best_move, score: best_score, depth, flag: tt_flag });
        best_score
    }

    fn quiescence_search(&mut self, mut alpha: i32, beta: i32) -> i32 {
        self.nodes_searched += 1;

        let static_score = self.tutor.evaluate(&self.game_state.board);

        if static_score >= beta {
            return beta;
        }
        if alpha < static_score {
            alpha = static_score;
        }

        // Delta pruning
        let big_delta = 900;
        if static_score + big_delta < alpha {
            return alpha;
        }

        // Only search captures and promotions
        let moves = MoveGen::new_legal(&self.game_state.board)
            .filter(|m| self.game_state.board.piece_on(m.get_dest()).is_some() || m.get_promotion().is_some());

        let mut scored_moves: Vec<(ChessMove, i32)> = moves.map(|mv| {
            let victim = self.game_state.board.piece_on(mv.get_dest());
            let score = if let Some(v) = victim {
                let attacker = self.game_state.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                crate::tutor::MG_VALUE[v.to_index()] * 10 - crate::tutor::MG_VALUE[attacker.to_index()]
            } else {
                500 // Promotion
            };
            (mv, score)
        }).collect();
        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, see_score) in scored_moves {
            // Skip clearly losing captures
            if see_score < 0 {
                continue;
            }

            let board_backup = self.game_state.board;
            self.game_state.board = self.game_state.board.make_move_new(mv);

            let score = -self.quiescence_search(-beta, -alpha);

            self.game_state.board = board_backup;

            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }
}
