// src/search.rs
//
// Asymmetric Scout Search: Combines Mamba-MoE brain with lightweight Alpha-Beta scout.
// Refactored for stateful inference, Lazy SMP, and hybrid evaluation.

use crate::engine::{Engine, EngineState, GameState};
use crate::tt::{TTEntry, TTFlag, TranspositionTable};
use crate::tutor::Tutor;
use candle_core::Tensor;
use chess::{Board, ChessMove, MoveGen, Piece, EMPTY};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MATE_SCORE: i32 = 30_000;
const INFINITY: i32 = 32_000;

const OVERHEAD_MS: u64 = 50;

pub fn move_to_token(mv: &ChessMove, board: &Board) -> usize {
    let piece = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
    let piece_idx = mv
        .get_promotion()
        .map_or_else(|| piece.to_index(), |promo_piece| promo_piece.to_index());
    let to_square_idx = mv.get_dest().to_int() as usize;
    piece_idx * 64 + to_square_idx
}

pub struct SearchHandler<'a> {
    pub engine: &'a Engine,
    pub tutor: &'a Tutor,
    pub tt: &'a TranspositionTable,
    pub game_state: GameState,
    pub nodes_searched: u64,
    pub start_time: Instant,
    pub time_limit: Duration,
    pub stop_flag: Arc<AtomicBool>,
    pub thread_id: usize,
    killers: [[Option<ChessMove>; 2]; 64],
    ply: usize,
}

impl<'a> SearchHandler<'a> {
    pub fn new(
        engine: &'a Engine,
        tutor: &'a Tutor,
        tt: &'a TranspositionTable,
        board: Board,
        stop_flag: Arc<AtomicBool>,
        thread_id: usize,
    ) -> Self {
        Self {
            engine,
            tutor,
            tt,
            game_state: GameState::new(board),
            nodes_searched: 0,
            start_time: Instant::now(),
            time_limit: Duration::from_secs(5),
            stop_flag,
            thread_id,
            killers: [[None; 2]; 64],
            ply: 0,
        }
    }

    pub fn search_with_time(&mut self, tc: &TimeControl) -> (Option<ChessMove>, i32) {
        let is_white = self.game_state.board.side_to_move() == chess::Color::White;
        self.time_limit = tc.calculate_time(is_white);
        self.start_time = Instant::now();
        self.search(tc.depth.unwrap_or(64))
    }

    pub fn search(&mut self, max_depth: u8) -> (Option<ChessMove>, i32) {
        self.nodes_searched = 0;
        self.killers = [[None; 2]; 64];
        self.ply = 0;

        let tokens = self.game_state.get_input_sequence();
        let (root_engine_state, root_policy, root_value) =
            self.engine.get_initial_state(&tokens).unwrap();

        let mut best_move = None;
        let mut best_score = -INFINITY;

        for depth in 1..=max_depth {
            if self.should_stop() {
                break;
            }

            let score = self.negamax(
                depth,
                -INFINITY,
                INFINITY,
                root_engine_state.clone(),
                root_value,
                Some(&root_policy),
            );

            if self.should_stop() && depth > 1 {
                break;
            }

            best_score = score;
            if let Some(entry) = self.tt.probe(self.game_state.board.get_hash()) {
                if let Some(mv) = entry.best_move {
                    best_move = Some(mv);
                }
            }

            if self.thread_id == 0 {
                let elapsed = self.start_time.elapsed().as_millis() as u64 + 1;
                let nps = (self.nodes_searched * 1000) / elapsed;
                println!(
                    "info depth {} score cp {} nodes {} nps {} time {} pv {}",
                    depth,
                    best_score,
                    self.nodes_searched,
                    nps,
                    elapsed,
                    best_move.map(|m| m.to_string()).unwrap_or_default()
                );
            }
        }

        if self.thread_id == 0 {
            self.stop_flag.store(true, Ordering::SeqCst);
        }

        (best_move, best_score)
    }

    fn should_stop(&self) -> bool {
        self.stop_flag.load(Ordering::Relaxed) || self.start_time.elapsed() >= self.time_limit
    }

    fn evaluate_hybrid(&self, board: &Board, ai_value: f32) -> i32 {
        let tutor_score = self.tutor.evaluate(board);
        let ai_score = (ai_value * 600.0) as i32;
        (0.7 * ai_score as f32 + 0.3 * tutor_score as f32) as i32
    }

    fn negamax(
        &mut self,
        depth: u8,
        mut alpha: i32,
        mut beta: i32,
        engine_state: EngineState,
        ai_value: f32,
        policy: Option<&Tensor>,
    ) -> i32 {
        // Periodic time check
        if self.nodes_searched & 63 == 0 && self.should_stop() {
            return 0;
        }

        self.nodes_searched += 1;
        let hash = self.game_state.board.get_hash();

        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(hash) {
            tt_move = entry.best_move;
            if entry.depth >= depth && self.ply > 0 {
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

        let status = self.game_state.board.status();
        if status == chess::BoardStatus::Checkmate {
            return -MATE_SCORE + self.ply as i32;
        }
        if status == chess::BoardStatus::Stalemate {
            return 0;
        }

        if depth == 0 {
            return self.quiescence_search(alpha, beta);
        }

        let static_score = if depth >= 2 {
            self.evaluate_hybrid(&self.game_state.board, ai_value)
        } else {
            self.tutor.evaluate(&self.game_state.board)
        };

        let in_check = *self.game_state.board.checkers() != EMPTY;

        if !in_check && self.ply > 0 {
            if depth <= 2 && static_score + 150 + (50 * depth as i32) < alpha {
                return static_score;
            }

            if depth >= 3 && static_score >= beta {
                if let Some(null_board) = self.game_state.board.null_move() {
                    let board_backup = self.game_state.board;
                    self.game_state.board = null_board;
                    self.ply += 1;
                    let r = 3 + depth / 6;
                    let score = -self.negamax(
                        depth.saturating_sub(r),
                        -beta,
                        -beta + 1,
                        engine_state.clone(),
                        ai_value,
                        None,
                    );
                    self.ply -= 1;
                    self.game_state.board = board_backup;
                    if score >= beta {
                        return beta;
                    }
                }
            }
        }

        let moves = MoveGen::new_legal(&self.game_state.board);
        let policy_vec = policy.and_then(|p| p.to_vec1::<f32>().ok());

        let mut scored_moves: Vec<(ChessMove, i32)> = moves
            .map(|mv| {
                let mut score = 0;
                if Some(mv) == tt_move {
                    score = 2_000_000;
                } else if let Some(victim) = self.game_state.board.piece_on(mv.get_dest()) {
                    let attacker = self
                        .game_state
                        .board
                        .piece_on(mv.get_source())
                        .unwrap_or(Piece::Pawn);
                    score = 1_000_000 + crate::tutor::MG_VALUE[victim.to_index()] * 10
                        - crate::tutor::MG_VALUE[attacker.to_index()];
                } else {
                    if self.ply < 64 {
                        if self.killers[self.ply][0] == Some(mv) {
                            score = 900_000;
                        } else if self.killers[self.ply][1] == Some(mv) {
                            score = 800_000;
                        }
                    }
                    if let Some(ref p) = policy_vec {
                        let token = move_to_token(&mv, &self.game_state.board);
                        if token < p.len() {
                            score += (p[token] * 100_000.0) as i32;
                        }
                    }
                }
                (mv, score)
            })
            .collect();

        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut tt_flag = TTFlag::UPPERBOUND;

        for (i, (mv, _)) in scored_moves.into_iter().enumerate() {
            let board_backup = self.game_state.board;
            self.game_state.board = self.game_state.board.make_move_new(mv);
            self.ply += 1;

            let mut next_engine_state = engine_state.clone();
            let mut next_ai_value = ai_value;
            let mut next_policy = None;

            if depth >= 2 {
                let token = move_to_token(&mv, &board_backup);
                if let Ok((p, v)) = self
                    .engine
                    .forward_incremental(token, &mut next_engine_state)
                {
                    next_ai_value = v;
                    next_policy = Some(p);
                }
            }

            let score = if i == 0 {
                -self.negamax(
                    depth - 1,
                    -beta,
                    -alpha,
                    next_engine_state,
                    next_ai_value,
                    next_policy.as_ref(),
                )
            } else {
                let reduction = if depth >= 3
                    && i >= 3
                    && !in_check
                    && board_backup.piece_on(mv.get_dest()).is_none()
                {
                    1 + (i / 8) as u8
                } else {
                    0
                };

                let mut s = -self.negamax(
                    depth.saturating_sub(1 + reduction),
                    -alpha - 1,
                    -alpha,
                    next_engine_state.clone(),
                    next_ai_value,
                    next_policy.as_ref(),
                );
                if s > alpha && reduction > 0 {
                    s = -self.negamax(
                        depth - 1,
                        -beta,
                        -alpha,
                        next_engine_state,
                        next_ai_value,
                        next_policy.as_ref(),
                    );
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
                if board_backup.piece_on(mv.get_dest()).is_none() && self.ply < 64 {
                    self.killers[self.ply][1] = self.killers[self.ply][0];
                    self.killers[self.ply][0] = Some(mv);
                }
                self.tt.store(TTEntry {
                    key: hash,
                    best_move: Some(mv),
                    score: best_score,
                    depth,
                    flag: TTFlag::LOWERBOUND,
                });
                return best_score;
            }
        }

        self.tt.store(TTEntry {
            key: hash,
            best_move,
            score: best_score,
            depth,
            flag: tt_flag,
        });
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

        let moves = MoveGen::new_legal(&self.game_state.board).filter(|m| {
            self.game_state.board.piece_on(m.get_dest()).is_some() || m.get_promotion().is_some()
        });

        let mut scored_moves: Vec<(ChessMove, i32)> = moves
            .map(|mv| {
                let victim = self.game_state.board.piece_on(mv.get_dest());
                let score = if let Some(v) = victim {
                    let attacker = self
                        .game_state
                        .board
                        .piece_on(mv.get_source())
                        .unwrap_or(Piece::Pawn);
                    crate::tutor::MG_VALUE[v.to_index()] * 10
                        - crate::tutor::MG_VALUE[attacker.to_index()]
                } else {
                    500
                };
                (mv, score)
            })
            .collect();
        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in scored_moves {
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

pub struct TimeControl {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
    pub movetime: Option<u64>,
    pub infinite: bool,
    pub depth: Option<u8>,
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
    pub fn calculate_time(&self, is_white: bool) -> Duration {
        if self.infinite {
            return Duration::from_secs(3600);
        }
        if let Some(mt) = self.movetime {
            return Duration::from_millis(mt.saturating_sub(OVERHEAD_MS));
        }
        let our_time = if is_white { self.wtime } else { self.btime };
        let our_inc = if is_white { self.winc } else { self.binc };
        if let Some(time) = our_time {
            let inc = our_inc.unwrap_or(0);
            let moves_left = self.movestogo.unwrap_or(30) as u64;
            let allocated = time / moves_left + inc - OVERHEAD_MS;
            Duration::from_millis(allocated.min(time / 2).max(100))
        } else {
            Duration::from_secs(5)
        }
    }
}
