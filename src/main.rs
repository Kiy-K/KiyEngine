// src/main.rs
//! KiyEngine V5 - Alpha-Beta Chess Engine with BitNet Dual-Head Evaluation

use kiy_engine_v5_alpha::uci::UciHandler;

fn main() -> anyhow::Result<()> {
    let mut handler = UciHandler::new()?;
    handler.run();
    Ok(())
}
