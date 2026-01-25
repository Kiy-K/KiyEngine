use std::io::{self, Write, BufRead};
use std::sync::atomic::Ordering;
use kiy_engine::{board::Board, mv::Move, search::Search, STOP_SEARCH, SEARCHING};

fn main() {
    let stdin = io::stdin();
    let mut board = Board::new();
    let mut searcher = Search::new();
    let mut skill_level: u8 = 5;

    std::panic::set_hook(Box::new(|info| {
        let msg = format!("PANIC: {}", info);
        let _ = std::fs::write("engine_crash.log", msg);
    }));

    for line_result in stdin.lock().lines() {
        let line = line_result.unwrap_or_default();
        let cmd = line.trim();

        if cmd == "uci" {
            println!("id name KiyEngine 1.0");
            println!("id author Khoi");
            println!("option name Skill Level type spin default 10 min 0 max 10");
            println!("option name Hash type spin default 64 min 1 max 1024");
            println!("option name Threads type spin default 1 min 1 max 1");
            println!("uciok");
        } else if cmd == "isready" {
            println!("readyok");
        } else if cmd.starts_with("setoption name Skill Level") {
            if let Some(val_str) = cmd.split("value ").nth(1) {
                if let Ok(val) = val_str.parse::<u8>() { skill_level = val.clamp(0, 10); }
            }
        } else if cmd.starts_with("setoption name Hash") {
            if let Some(val_str) = cmd.split("value ").nth(1) {
                if let Ok(val) = val_str.parse::<usize>() {
                    searcher.tt.resize(val);
                }
            }
        } else if cmd.starts_with("setoption name Threads") {
            // Standard acknowledgement, functionality limited to 1 thread for now.
        } else if cmd == "ucinewgame" {
            board = Board::new();
            searcher = Search::new();
        } else if cmd.starts_with("position startpos") {
            board = Board::new();
            if let Some(moves_part) = cmd.split("moves ").nth(1) {
                for move_str in moves_part.split_whitespace() {
                     let moves = kiy_engine::movegen::MoveGen::generate_moves(&board);
                     let mut found = false;
                     for i in 0..moves.count {
                          let m = moves.moves[i];
                          if m.to_string() == move_str || (move_str.len() == 5 && move_str.starts_with(&m.to_string())) {
                              board.make_move(m);
                              found = true;
                              break;
                          }
                     }
                     if !found && move_str.len() >= 4 {
                         let s = (move_str.as_bytes()[0] - b'a') + (move_str.as_bytes()[1] - b'1') * 8;
                         let d = (move_str.as_bytes()[2] - b'a') + (move_str.as_bytes()[3] - b'1') * 8;
                         board.make_move(Move::new(s, d, None));
                     }
                }
            }
        } else if cmd.starts_with("go") {
            let mut depth = (skill_level + 2).min(15) as u8;
            if skill_level == 10 { depth = 15; }
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            for i in 0..parts.len() {
                if parts[i] == "depth" && i + 1 < parts.len() {
                    if let Ok(d) = parts[i+1].parse::<u8>() { depth = d; }
                }
            }

            let mut choice_move = None;
            let mut rng = rand::thread_rng();
            use rand::Rng;
            if rng.gen_bool((10 - skill_level) as f64 * 0.05) {
                let moves = kiy_engine::movegen::MoveGen::generate_moves(&board);
                if moves.count > 0 { choice_move = Some(moves.moves[rng.gen_range(0..moves.count)]); }
            }

            if choice_move.is_none() {
                 let board_clone = board.clone();
                 let mut searcher_clone = searcher.clone();
                 STOP_SEARCH.store(false, Ordering::SeqCst);
                 SEARCHING.store(true, Ordering::SeqCst);
                 
                 let handle = std::thread::Builder::new()
                     .name("SearchThread".into())
                     .stack_size(32 * 1024 * 1024)
                     .spawn(move || {
                         searcher_clone.loop_search(&board_clone, depth)
                     }).unwrap();
                
                match handle.join() {
                    Ok(res) => { choice_move = res; },
                    Err(_) => { choice_move = None; }
                }
                SEARCHING.store(false, Ordering::SeqCst);
            }

            if let Some(best_move) = choice_move {
                 println!("bestmove {}", best_move.to_string());
            } else {
                 println!("bestmove (none)");
            }
        } else if cmd == "stop" {
            STOP_SEARCH.store(true, Ordering::SeqCst);
        } else if cmd == "quit" {
            break;
        } else if cmd == "bench" {
            let mut s = searcher.clone();
            let _ = s.loop_search(&Board::new(), 15);
            println!("Nodes: {}", s.nodes);
        }
        io::stdout().flush().ok();
    }
}
