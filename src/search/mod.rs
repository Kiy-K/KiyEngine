// src/search/mod.rs
//! Search module - Lazy SMP-based alpha-beta and MCTS search engines

pub mod lazy_smp;
pub mod mcts;
pub mod mcts_arena;
pub mod mcts_batch;
pub mod mcts_memory;
pub mod time;
pub mod tt;

// Re-export commonly used types
pub use lazy_smp::Searcher;
pub use mcts::{Edge, MCTSConfig, MCTSSearcher, Node};
pub use mcts_arena::{MCTSHandler, MCTSTree, NodeIdx};
pub use mcts_batch::MCTSBatchedInference;
pub use mcts_memory::{MCTSTreeManager, NodeRef};
pub use time::TimeManager;
pub use tt::{AtomicTT, TTFlag};
