// src/main.rs
//! KiyEngine - Alpha-Beta Chess Engine with BitNet Dual-Head Evaluation

use kiyengine::uci::UciHandler;

fn main() -> anyhow::Result<()> {
    let mut handler = UciHandler::new()?;
    handler.run();
    Ok(())
}
