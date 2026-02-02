// src/main.rs

use kiy_engine_v4_omega::uci::UciHandler;

fn main() -> anyhow::Result<()> {
    let mut handler = UciHandler::new()?;
    handler.run();
    Ok(())
}
