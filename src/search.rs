use crate::board::Board;
use crate::evaluate::evaluate;
use crate::movegen::MoveGen;
use crate::mv::{Move, MoveList};
use crate::defs::{Color};
use crate::tt::{TranspositionTable, HashFlag};

#[derive(Clone)]
pub struct Search {
    pub tt: std::sync::Arc<TranspositionTable>,
    pub nodes: u64,
    pub killer_moves: [[Option<Move>; 2]; 64],
    pub history: [[i32; 64]; 12],
}

impl Search {
    pub fn new() -> Self {
        Search {
            tt: std::sync::Arc::new(TranspositionTable::new(64)),
            nodes: 0,
            killer_moves: [[None; 2]; 64],
            history: [[0; 64]; 12],
        }
    }

    pub fn loop_search(&mut self, board: &Board, max_depth: u8) -> Option<Move> {
        let mut best_move = None;
        self.nodes = 0;
        let start_time = std::time::Instant::now();

        for depth in 1..=max_depth {
            if crate::STOP_SEARCH.load(std::sync::atomic::Ordering::Relaxed) { break; }
            let mut alpha = -30000;
            let beta = 30000;
            
            // Search at this depth
            let mut moves = MoveGen::generate_moves(board);
            
            let mut tt_move = None;
            if let Some((_, m)) = self.tt.probe(board.hash, depth, alpha, beta) {
                tt_move = m;
            }

            self.score_moves(board, &mut moves, tt_move, depth);
            Self::sort_moves(&mut moves);
            
            let mut current_best_moves = Vec::new();
            let mut best_score = -50000;

            for i in 0..moves.count {
                let m = moves.moves[i];
                let mut new_board = board.clone();
                
                if new_board.make_move(m) {
                    // LEGALE CHECK
                    let king_bb = new_board.pieces[board.side as usize][crate::defs::PieceType::King as usize];
                    if king_bb != 0 {
                        let king_sq = king_bb.trailing_zeros() as u8;
                        if new_board.is_square_attacked(king_sq, new_board.side) { continue; }
                    }

                    let score = -self.alpha_beta(&new_board, depth - 1, -beta, -alpha);
                    if crate::STOP_SEARCH.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    
                    if score > alpha {
                        alpha = score;
                        current_best_moves.clear();
                        current_best_moves.push(m);
                        best_score = score;
                    } else if score == alpha && !current_best_moves.is_empty() {
                        current_best_moves.push(m);
                    }
                }
            }
            if crate::STOP_SEARCH.load(std::sync::atomic::Ordering::Relaxed) { break; }
            
            if !current_best_moves.is_empty() {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let m = current_best_moves.choose(&mut rng).unwrap();
                best_move = Some(*m);
                
                let elapsed = start_time.elapsed().as_millis() as u64;
                let nps = if elapsed > 0 { (self.nodes * 1000) / elapsed } else { 0 };
                println!("info depth {} score cp {} nodes {} nps {} time {} pv {}", 
                    depth, best_score, self.nodes, nps, elapsed, m.to_string());
            }
        }

        best_move
    }

    fn alpha_beta(&mut self, board: &Board, depth: u8, mut alpha: i32, beta: i32) -> i32 {
        if self.nodes % 2048 == 0 {
             if crate::STOP_SEARCH.load(std::sync::atomic::Ordering::Relaxed) { return alpha; }
             // Periodic stats (simplified for now, ideally with a timer)
             if self.nodes % 131072 == 0 {
                  println!("info nodes {} nps {}", self.nodes, 1000000); // Placeholder for real metrics
             }
        }
        self.nodes += 1;
        
        let side = board.side;
        let king_bb = board.pieces[side as usize][crate::defs::PieceType::King as usize];
        let king_sq = if king_bb != 0 { king_bb.trailing_zeros() as u8 } else { 0 };
        let in_check = board.is_square_attacked(king_sq, side.opposite());

        // --- 1. TT Probe ---
        let mut tt_move = None;
        if let Some((score, m)) = self.tt.probe(board.hash, depth, alpha, beta) {
            if depth > 0 && score != 0 { return score; }
            tt_move = m;
        }

        if depth == 0 {
            return self.quiescence(board, alpha, beta);
        }

        // --- 2. Null Move Pruning (NMP) ---
        if !in_check && depth >= 3 {
             let mut null_board = board.clone();
             null_board.side = side.opposite();
             null_board.hash ^= crate::board::ZOBRIST.side;
             let reduction = 3;
             let next_depth = if depth > reduction + 1 { depth - reduction - 1 } else { 0 };
             let score = -self.alpha_beta(&null_board, next_depth, -beta, -beta + 1);
             if score >= beta { return beta; }
        }

        let mut moves = MoveGen::generate_moves(board);
        self.score_moves(board, &mut moves, tt_move, depth);
        Self::sort_moves(&mut moves);

        let mut legal_moves_found = 0;
        let mut best_move = None;
        let old_alpha = alpha;

        for i in 0..moves.count {
            let m = moves.moves[i];
            let mut new_board = board.clone();
            
            if new_board.make_move(m) {
                // Legality check
                let k_bb = new_board.pieces[side as usize][crate::defs::PieceType::King as usize];
                if k_bb != 0 {
                    let k_sq = k_bb.trailing_zeros() as u8;
                    if new_board.is_square_attacked(k_sq, new_board.side) { continue; }
                }

                legal_moves_found += 1;
                
                // --- 3. Stable Extensions ---
                // We only extend if we haven't reached a crazy depth already
                let mut extension = 0;
                if in_check && depth < 32 { 
                    extension = 1; 
                }

                let mut score;
                let is_capture = (board.occupancy[side.opposite() as usize] >> m.target()) & 1 == 1;

                if depth >= 3 && legal_moves_found > 4 && !is_capture && !in_check {
                    // LMR
                    score = -self.alpha_beta(&new_board, depth - 2, -alpha - 1, -alpha);
                    if score > alpha {
                        score = -self.alpha_beta(&new_board, depth - 1 + extension, -beta, -alpha);
                    }
                } else {
                    // Standard search
                    let next_depth = if depth > 0 { depth - 1 + extension } else { 0 };
                    // Hard safety: if extension is used, ensure we don't loop forever at depth 1
                    let safe_depth = if next_depth >= depth && depth > 0 { depth - 1 } else { next_depth };
                    score = -self.alpha_beta(&new_board, safe_depth, -beta, -alpha);
                }
                
                if score >= beta {
                    // Update Heuristics for non-captures
                    if !is_capture && depth < 64 {
                        self.killer_moves[(depth as usize).min(63)][1] = self.killer_moves[(depth as usize).min(63)][0];
                        self.killer_moves[(depth as usize).min(63)][0] = Some(m);
                        
                        let pt_idx = (0..6).find(|&pt| (board.pieces[side as usize][pt] >> m.source()) & 1 == 1).unwrap_or(0);
                        self.history[(side as usize)*6 + pt_idx][m.target() as usize] += (depth as i32) * (depth as i32);
                    }

                    self.tt.store(board.hash, depth, beta, HashFlag::Beta, Some(m));
                    return beta;
                }
                if score > alpha {
                    alpha = score;
                    best_move = Some(m);
                }
            }
        }

        if legal_moves_found == 0 {
            if in_check { return -29000 + (depth as i32); }
            return 0; // Stalemate
        }

        let flag = if alpha <= old_alpha { HashFlag::Alpha } else { HashFlag::Exact };
        self.tt.store(board.hash, depth, alpha, flag, best_move);

        alpha
    }

    fn quiescence(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        let stand_pat = Self::evaluate_relative(board);
        if stand_pat >= beta { return beta; }
        if stand_pat > alpha { alpha = stand_pat; }

        let mut moves = MoveGen::generate_moves(board);
        self.score_moves(board, &mut moves, None, 0);
        Self::sort_moves(&mut moves);
        
        let side = board.side;
        for i in 0..moves.count {
            let m = moves.moves[i];
            let is_capture = (board.occupancy[side.opposite() as usize] >> m.target()) & 1 == 1;
            if !is_capture { continue; }

            let mut new_board = board.clone();
            if new_board.make_move(m) {
                // Legality check in Q-search
                let k_bb = new_board.pieces[side as usize][crate::defs::PieceType::King as usize];
                if k_bb != 0 {
                    let k_sq = k_bb.trailing_zeros() as u8;
                    if new_board.is_square_attacked(k_sq, new_board.side) {
                        continue;
                    }
                }
                let score = -self.quiescence(&new_board, -beta, -alpha);
                if score >= beta { return beta; }
                if score > alpha { alpha = score; }
            }
        }
        alpha
    }

    fn evaluate_relative(board: &Board) -> i32 {
        let eval = evaluate(board);
        if board.side == Color::White { eval } else { -eval }
    }

    fn score_moves(&self, board: &Board, moves: &mut MoveList, tt_move: Option<Move>, depth: u8) {
        let side_offset = (board.side as usize) * 6;
        for i in 0..moves.count {
            let m = &mut moves.moves[i];
            
            // 1. TT Move Priority
            if let Some(tm) = tt_move {
                if m.source() == tm.source() && m.target() == tm.target() {
                    m.score = 2_000_000;
                    continue;
                }
            }

            // 2. Captures (MVV-LVA)
            let target = m.target();
            let mut victim_type = None;
            for pt in 0..crate::defs::PIECE_TYPE_COUNT {
                if (board.pieces[board.side.opposite() as usize][pt] >> target) & 1 == 1 {
                    victim_type = Some(pt);
                    break;
                }
            }

            if let Some(v) = victim_type {
                let mut attacker = 0;
                for pt in 0..crate::defs::PIECE_TYPE_COUNT {
                    if (board.pieces[board.side as usize][pt] >> m.source()) & 1 == 1 {
                        attacker = pt;
                        break;
                    }
                }
                m.score = 1_000_000 + (v as i32 * 100) - (attacker as i32);
            } else {
                // 3. Killer Moves
                if depth < 64 {
                    if let Some(k1) = self.killer_moves[depth as usize][0] {
                        if m.source() == k1.source() && m.target() == k1.target() {
                            m.score = 900_000;
                        }
                    } else if let Some(k2) = self.killer_moves[depth as usize][1] {
                        if m.source() == k2.source() && m.target() == k2.target() {
                            m.score = 800_000;
                        }
                    }
                }

                // 4. History Heuristic
                if m.score == 0 {
                    // Identify piece type for history lookup
                    let mut pt_idx = 0;
                    for pt in 0..crate::defs::PIECE_TYPE_COUNT {
                        if (board.pieces[board.side as usize][pt] >> m.source()) & 1 == 1 {
                            pt_idx = pt;
                            break;
                        }
                    }
                    m.score = self.history[side_offset + pt_idx][m.target() as usize];
                }
            }
        }
    }

    fn sort_moves(moves: &mut MoveList) {
        moves.sort();
    }
}
