// src/main.rs
//! KiyEngine V5 - AlphaZero-style MCTS Chess Engine

use kiy_engine_v5_alpha::uci::mcts_handler::MCTSUciHandler;

fn main() -> anyhow::Result<()> {
    let mut handler = MCTSUciHandler::new()?;
    handler.run();
    Ok(())
}
