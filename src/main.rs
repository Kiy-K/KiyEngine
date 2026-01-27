// src/main.rs

use kiy_engine_v3::uci::UciHandler;
use std::io::{self, BufRead};

fn main() -> anyhow::Result<()> {
    let mut uci_handler = UciHandler::new()?;
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let command = line.unwrap_or_else(|_| String::new());
        if !command.trim().is_empty() {
            uci_handler.handle_command(&command);
        }
    }
    Ok(())
}
