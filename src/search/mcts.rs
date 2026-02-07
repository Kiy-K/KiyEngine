// src/search/mcts.rs
//! AlphaZero-style Monte Carlo Tree Search (MCTS) implementation

use crate::engine::Engine;
use crate::uci::MoveCodec;
use chess::{Board, ChessMove, Color, MoveGen};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc::Sender, Arc};
use std::time::Instant;

// MCTS Constants
pub const DEFAULT_SIMULATIONS: usize = 16000;
pub const DEFAULT_BATCH_SIZE: usize = 8;
pub const MAX_BATCH_DELAY_MS: u64 = 5;
pub const DIRICHLET_ALPHA: f32 = 0.3;
pub const DIRICHLET_EPSILON: f32 = 0.25;
pub const CPUCT_BASE: f32 = 1.25;
pub const CPUCT_FACTOR: f32 = 19652.0;
pub const FPU_REDUCTION: f32 = 0.5;
pub const VIRTUAL_LOSS: f32 = 1.0;
pub const MAX_TREE_NODES: usize = 10_000_000;

/// Atomic F32 for lock-free updates
pub struct AtomicF32 {
    inner: AtomicU32,
}

impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        Self {
            inner: AtomicU32::new(value.to_bits()),
        }
    }

    pub fn load(&self, ordering: Ordering) -> f32 {
        f32::from_bits(self.inner.load(ordering))
    }

    pub fn fetch_add(&self, value: f32, ordering: Ordering) {
        loop {
            let current_bits = self.inner.load(ordering);
            let current = f32::from_bits(current_bits);
            let new_val = current + value;
            let new_bits = new_val.to_bits();

            if self
                .inner
                .compare_exchange_weak(current_bits, new_bits, ordering, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }
}

/// Cache-friendly edge storage using Vec (SoA pattern)
pub struct EdgeArray {
    moves: Vec<ChessMove>,
    edges: Vec<Edge>,
}

impl EdgeArray {
    pub fn new() -> Self {
        Self {
            moves: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            moves: Vec::with_capacity(capacity),
            edges: Vec::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, mv: ChessMove, edge: Edge) {
        self.moves.push(mv);
        self.edges.push(edge);
    }

    pub fn get(&self, mv: ChessMove) -> Option<&Edge> {
        self.moves
            .iter()
            .position(|&m| m == mv)
            .map(|idx| &self.edges[idx])
    }

    pub fn get_mut(&mut self, mv: ChessMove) -> Option<&mut Edge> {
        if let Some(idx) = self.moves.iter().position(|&m| m == mv) {
            Some(&mut self.edges[idx])
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (ChessMove, &Edge)> {
        self.moves.iter().copied().zip(self.edges.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ChessMove, &mut Edge)> {
        let moves = self.moves.clone();
        moves.into_iter().zip(self.edges.iter_mut())
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &ChessMove> {
        self.moves.iter()
    }
}

/// Edge in MCTS tree (represents an action)
pub struct Edge {
    pub prior: f32,
    pub visits: AtomicU32,
    pub total_value: AtomicF32,
    pub virtual_visits: AtomicU32,
}

impl Edge {
    pub fn new(prior: f32) -> Self {
        Self {
            prior,
            visits: AtomicU32::new(0),
            total_value: AtomicF32::new(0.0),
            virtual_visits: AtomicU32::new(0),
        }
    }

    pub fn get_visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    pub fn get_q(&self) -> f32 {
        let visits = self.get_visits();
        if visits == 0 {
            return f32::NAN;
        }
        let total = self.total_value.load(Ordering::Relaxed);
        total / visits as f32
    }

    pub fn effective_visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed) + self.virtual_visits.load(Ordering::Relaxed)
    }

    pub fn add_virtual_loss(&self) {
        self.virtual_visits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update(&self, value: f32) {
        self.total_value.fetch_add(value, Ordering::Relaxed);
        self.visits.fetch_add(1, Ordering::Relaxed);
        let virtual_v = self.virtual_visits.load(Ordering::Relaxed);
        if virtual_v > 0 {
            self.virtual_visits.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Node in MCTS tree (represents a position)
pub struct Node {
    pub board: Board,
    pub side_to_move: Color,
    pub is_terminal: bool,
    pub terminal_value: Option<f32>,
    pub edges: RwLock<EdgeArray>,
    pub is_expanded: AtomicBool,
    pub total_visits: AtomicU32,
}

impl Node {
    pub fn new(board: Board) -> Self {
        let side_to_move = board.side_to_move();

        // Check terminal
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();
        let (is_terminal, terminal_value) = if legal_moves.is_empty() {
            let checkers = board.checkers();
            if checkers.popcnt() > 0 {
                (true, Some(-1.0)) // Checkmate
            } else {
                (true, Some(0.0)) // Stalemate
            }
        } else if board.combined().popcnt() == 2 {
            (true, Some(0.0)) // Insufficient material
        } else {
            (false, None)
        };

        Self {
            board,
            side_to_move,
            is_terminal,
            terminal_value,
            edges: RwLock::new(EdgeArray::new()),
            is_expanded: AtomicBool::new(false),
            total_visits: AtomicU32::new(0),
        }
    }

    pub fn expand(&self, policy: &[(ChessMove, f32)]) {
        if self.is_terminal || self.is_expanded.load(Ordering::Relaxed) {
            return;
        }

        let mut edges = self.edges.write();
        for (mv, prior) in policy {
            edges.insert(*mv, Edge::new(*prior));
        }

        self.is_expanded.store(true, Ordering::Relaxed);
    }

    pub fn get_total_visits(&self) -> u32 {
        let real_visits = self.total_visits.load(Ordering::Relaxed);
        if real_visits == 0 {
            return 1;
        }
        real_visits
    }

    pub fn get_parent_q(&self) -> f32 {
        let visits = self.total_visits.load(Ordering::Relaxed);
        if visits == 0 {
            return 0.0;
        }

        let edges = self.edges.read();
        let mut total_w = 0.0;
        let mut total_n = 0;

        for (_, edge) in edges.iter() {
            let e_visits = edge.get_visits();
            if e_visits > 0 {
                total_w += edge.total_value.load(Ordering::Relaxed);
                total_n += e_visits;
            }
        }

        if total_n == 0 {
            0.0
        } else {
            total_w / total_n as f32
        }
    }
}

/// MCTS Configuration
#[derive(Debug, Clone)]
pub struct MCTSConfig {
    pub num_simulations: usize,
    pub batch_size: usize,
    pub max_batch_delay_ms: u64,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub fpu_reduction: f32,
    pub threads: usize,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_simulations: DEFAULT_SIMULATIONS,
            batch_size: DEFAULT_BATCH_SIZE,
            max_batch_delay_ms: MAX_BATCH_DELAY_MS,
            dirichlet_alpha: DIRICHLET_ALPHA,
            dirichlet_epsilon: DIRICHLET_EPSILON,
            cpuct_base: CPUCT_BASE,
            cpuct_factor: CPUCT_FACTOR,
            fpu_reduction: FPU_REDUCTION,
            threads: 4,
        }
    }
}

/// Selection path result
pub struct SelectionPath {
    pub nodes: Vec<Arc<Node>>,
    pub moves: Vec<ChessMove>,
}

/// Main MCTS Searcher
pub struct MCTSSearcher {
    pub engine: Arc<Engine>,
    config: MCTSConfig,
    stop_flag: Arc<AtomicBool>,
}

impl MCTSSearcher {
    pub fn new(engine: Arc<Engine>, config: MCTSConfig) -> Self {
        Self {
            engine,
            config,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.config.threads = threads.max(1);
    }

    pub fn set_config(&mut self, config: MCTSConfig) {
        self.config = config;
    }

    pub fn search_parallel(
        &self,
        board: &Board,
        move_history: &[usize],
        time_ms: u64,
    ) -> (ChessMove, f32) {
        use rayon::prelude::*;

        eprintln!(
            "[MCTS parallel] Starting with {} simulations, {}ms",
            self.config.num_simulations, time_ms
        );
        let root = Arc::new(Node::new(board.clone()));

        self.expand_root(&root, move_history);
        self.add_dirichlet_noise(&root);

        let start_time = Instant::now();
        let simulations_completed = Arc::new(AtomicU32::new(0));

        // Run parallel simulations in batches
        let batch_size = (self.config.num_simulations / self.config.threads.max(1)).max(10);

        (0..self.config.num_simulations / batch_size)
            .into_par_iter()
            .for_each(|_| {
                for _ in 0..batch_size {
                    if start_time.elapsed().as_millis() as u64 >= time_ms {
                        break;
                    }

                    self.run_simulation(&root, move_history);
                    simulations_completed.fetch_add(1, Ordering::Relaxed);
                }
            });

        let sims = simulations_completed.load(Ordering::Relaxed);
        eprintln!("[MCTS parallel] Completed {} simulations", sims);
        self.select_best_move(&root)
    }

    pub fn search(&self, board: &Board, move_history: &[usize], time_ms: u64) -> (ChessMove, f32) {
        eprintln!(
            "[MCTS search] Starting with {} simulations, {}ms",
            self.config.num_simulations, time_ms
        );
        let root = Arc::new(Node::new(board.clone()));

        eprintln!("[MCTS search] Expanding root...");
        self.expand_root(&root, move_history);
        eprintln!("[MCTS search] Root expanded, adding noise...");
        self.add_dirichlet_noise(&root);

        let start_time = Instant::now();
        let mut simulations = 0;

        eprintln!("[MCTS search] Starting simulation loop");
        while simulations < self.config.num_simulations {
            if start_time.elapsed().as_millis() as u64 >= time_ms {
                eprintln!(
                    "[MCTS search] Time limit reached after {} sims",
                    simulations
                );
                break;
            }

            if self.stop_flag.load(Ordering::Relaxed) {
                eprintln!("[MCTS search] Stop flag set");
                break;
            }

            self.run_simulation(&root, move_history);
            simulations += 1;

            if simulations % 100 == 0 {
                eprintln!("[MCTS search] Completed {} simulations", simulations);
            }
        }

        eprintln!(
            "[MCTS search] Selecting best move after {} simulations",
            simulations
        );
        let result = self.select_best_move(&root);
        eprintln!("[MCTS search] Best move: {}", result.0);
        result
    }

    fn run_simulation(&self, root: &Arc<Node>, move_history: &[usize]) {
        let path = self.select_path(root);

        if path.nodes.is_empty() {
            return;
        }

        let leaf = path.nodes.last().unwrap().clone();

        let value = if leaf.is_terminal {
            leaf.terminal_value.unwrap_or(0.0)
        } else if !leaf.is_expanded.load(Ordering::Relaxed) {
            let mut leaf_history = move_history.to_vec();
            for mv in &path.moves {
                leaf_history.push(MoveCodec::move_to_token(mv) as usize);
            }

            match self.evaluate_node(&leaf, &leaf_history) {
                Some((policy, v)) => {
                    leaf.expand(&policy);
                    v
                }
                None => 0.0,
            }
        } else {
            leaf.get_parent_q()
        };

        self.backpropagate(&path, value);
    }

    fn select_path(&self, root: &Arc<Node>) -> SelectionPath {
        let mut path = SelectionPath {
            nodes: vec![root.clone()],
            moves: Vec::new(),
        };

        let mut current = root.clone();

        loop {
            if current.is_terminal {
                return path;
            }

            if !current.is_expanded.load(Ordering::Relaxed) {
                return path;
            }

            let edges = current.edges.read();

            if edges.is_empty() {
                return path;
            }

            let total_visits = current.get_total_visits();
            let cpuct = self.compute_cpuct(total_visits);
            let parent_q = current.get_parent_q();

            let mut best_move = None;
            let mut best_score = f32::NEG_INFINITY;

            for (mv, edge) in edges.iter() {
                let score = self.puct_score(edge, total_visits, cpuct, parent_q);

                if score > best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
            }

            drop(edges);

            if let Some(mv) = best_move {
                {
                    let edges = current.edges.read();
                    if let Some(edge) = edges.get(mv) {
                        edge.add_virtual_loss();
                    }
                }

                let new_board = current.board.make_move_new(mv);
                let child = Arc::new(Node::new(new_board));

                path.moves.push(mv);
                path.nodes.push(child.clone());
                current = child;
            } else {
                break;
            }
        }

        path
    }

    fn puct_score(&self, edge: &Edge, parent_visits: u32, cpuct: f32, parent_q: f32) -> f32 {
        let q = if self.config.fpu_reduction > 0.0 {
            let visits = edge.get_visits();
            if visits == 0 {
                parent_q - self.config.fpu_reduction * 0.5f32.sqrt()
            } else {
                edge.get_q()
            }
        } else {
            let visits = edge.get_visits();
            if visits == 0 {
                0.0
            } else {
                edge.get_q()
            }
        };

        let effective_visits = edge.effective_visits() as f32;
        let u = cpuct * edge.prior * (parent_visits as f32).sqrt() / (1.0 + effective_visits);

        q + u
    }

    fn compute_cpuct(&self, parent_visits: u32) -> f32 {
        let log_visits = ((1 + parent_visits) as f32).ln();
        self.config.cpuct_base
            + log_visits * (self.config.cpuct_base / self.config.cpuct_factor.sqrt())
    }

    fn evaluate_node(
        &self,
        node: &Node,
        move_history: &[usize],
    ) -> Option<(Vec<(ChessMove, f32)>, f32)> {
        let (policy_logits, value) = self.engine.forward(move_history).ok()?;

        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&node.board).collect();
        if legal_moves.is_empty() {
            return Some((Vec::new(), value));
        }

        let logits = policy_logits.to_vec1::<f32>().ok()?;

        let mut move_logits: Vec<(ChessMove, f32)> = legal_moves
            .into_iter()
            .filter_map(|mv| {
                let token = MoveCodec::move_to_token(&mv) as usize;
                if token < logits.len() {
                    Some((mv, logits[token]))
                } else {
                    None
                }
            })
            .collect();

        if move_logits.is_empty() {
            return Some((Vec::new(), value));
        }

        let max_logit = move_logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut exp_sum = 0.0f32;
        for (_, logit) in &mut move_logits {
            let exp_val = (*logit - max_logit).exp();
            *logit = exp_val;
            exp_sum += exp_val;
        }

        if exp_sum > 0.0 {
            for (_, prob) in &mut move_logits {
                *prob /= exp_sum;
            }
        }

        let policy = move_logits;

        let value_for_parent = if node.side_to_move == Color::White {
            value
        } else {
            -value
        };

        Some((policy, value_for_parent))
    }

    fn expand_root(&self, root: &Arc<Node>, move_history: &[usize]) {
        if let Some((policy, _)) = self.evaluate_node(root, move_history) {
            root.expand(&policy);
        }
    }

    fn add_dirichlet_noise(&self, root: &Arc<Node>) {
        let mut edges = root.edges.write();
        if edges.is_empty() {
            return;
        }

        let n = edges.len();
        let noise = sample_dirichlet(self.config.dirichlet_alpha, n);

        let epsilon = self.config.dirichlet_epsilon;
        let mut i = 0;
        for (_, edge) in edges.iter_mut() {
            edge.prior = (1.0 - epsilon) * edge.prior + epsilon * noise[i];
            i += 1;
        }
    }

    fn backpropagate(&self, path: &SelectionPath, mut value: f32) {
        // Traverse backwards from leaf to root
        for (i, node) in path.nodes.iter().enumerate().rev() {
            node.total_visits.fetch_add(1, Ordering::Relaxed);

            if i > 0 {
                let parent = &path.nodes[i - 1];
                let mv = path.moves[i - 1];

                // Get side info before accessing edges
                let side = node.side_to_move;
                let parent_side = parent.side_to_move;

                // Turn-based value inversion: flip value when sides differ
                // This implements minimax within MCTS - what's good for one player is bad for the other
                if side != parent_side {
                    value = -value;
                }

                let edges = parent.edges.read();
                if let Some(edge) = edges.get(mv) {
                    edge.update(value);
                }
            }
        }
    }

    fn select_best_move(&self, root: &Arc<Node>) -> (ChessMove, f32) {
        let edges = root.edges.read();

        if edges.is_empty() {
            let legal = MoveGen::new_legal(&root.board);
            if let Some(mv) = legal.into_iter().next() {
                return (mv, 0.0);
            }
            panic!("No legal moves in position");
        }

        let mut best_move = None;
        let mut max_visits = 0u32;
        let mut best_q = 0.0f32;

        for (mv, edge) in edges.iter() {
            let visits = edge.get_visits();
            if visits > max_visits {
                max_visits = visits;
                best_move = Some(mv);
                best_q = edge.get_q();
            }
        }

        (
            best_move.unwrap_or_else(|| edges.keys().copied().next().unwrap()),
            best_q,
        )
    }

    pub fn search_async(
        &self,
        board: Board,
        move_history: Vec<usize>,
        time_ms: u64,
        stop_flag: Arc<AtomicBool>,
        tx: Sender<String>,
    ) {
        eprintln!("[MCTS] Starting search_async");
        let root = Arc::new(Node::new(board));
        eprintln!("[MCTS] Expanding root...");
        self.expand_root(&root, &move_history);
        eprintln!("[MCTS] Adding Dirichlet noise...");
        self.add_dirichlet_noise(&root);

        let start_time = Instant::now();
        let mut simulations = 0;
        let mut last_info_time = Instant::now();

        eprintln!(
            "[MCTS] Starting simulation loop (target: {} sims, time: {}ms)",
            self.config.num_simulations, time_ms
        );
        while simulations < self.config.num_simulations {
            if stop_flag.load(Ordering::Relaxed) {
                eprintln!("[MCTS] Stop flag set, breaking");
                break;
            }

            if start_time.elapsed().as_millis() as u64 >= time_ms {
                eprintln!("[MCTS] Time limit reached");
                break;
            }

            self.run_simulation(&root, &move_history);
            simulations += 1;

            if last_info_time.elapsed().as_millis() >= 100 {
                let (best_mv, score) = self.select_best_move(&root);
                let nps = (simulations as f64 / start_time.elapsed().as_secs_f64()) as u64;
                let msg = format!(
                    "info depth {} score cp {} nodes {} nps {} pv {}",
                    simulations / 20,
                    (score * 100.0) as i32,
                    simulations,
                    nps,
                    best_mv
                );
                if tx.send(msg.clone()).is_err() {
                    eprintln!("[MCTS] Failed to send info: {}", msg);
                }
                last_info_time = Instant::now();
            }
        }

        eprintln!(
            "[MCTS] Completed {} simulations, selecting best move",
            simulations
        );
        let (best_move, _) = self.select_best_move(&root);
        let bestmove_msg = format!("bestmove {}", best_move);
        eprintln!("[MCTS] Sending: {}", bestmove_msg);
        if tx.send(bestmove_msg.clone()).is_err() {
            eprintln!("[MCTS] Failed to send bestmove, printing to stdout");
            println!("{}", bestmove_msg);
        }
        eprintln!("[MCTS] Search complete");
    }
}

/// Sample from Dirichlet distribution using simple gamma approximation
fn sample_dirichlet(alpha: f32, n: usize) -> Vec<f32> {
    let mut samples = Vec::with_capacity(n);
    let mut sum = 0.0f64;

    for _ in 0..n {
        // Simple exponential approximation for gamma distribution
        let u = fast_random();
        let exp_sample = -(u.ln()) * (alpha as f64);
        let gamma_sample = exp_sample.max(0.001);

        samples.push(gamma_sample as f32);
        sum += gamma_sample;
    }

    if sum > 0.0 {
        for s in &mut samples {
            *s = (*s as f64 / sum) as f32;
        }
    } else {
        let uniform = 1.0 / n as f32;
        for s in &mut samples {
            *s = uniform;
        }
    }

    samples
}

/// Simple fast random number generator (xorshift)
fn fast_random() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SEED: AtomicU64 = AtomicU64::new(0);

    let mut seed = SEED.load(Ordering::Relaxed);
    if seed == 0 {
        seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
    }

    // xorshift64*
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    SEED.store(seed, Ordering::Relaxed);

    // Return value in [0.0001, 0.9999]
    let val = (seed >> 11) as f64 / ((1u64 << 53) as f64);
    val.max(0.0001).min(0.9999)
}
