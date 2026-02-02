use chess::{Board, ChessMove};
use kiy_engine_v4_omega::engine::Engine;
use kiy_engine_v4_omega::search::{Searcher, TranspositionTable};
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let engine = Arc::new(Engine::new()?);
    let tt = Arc::new(TranspositionTable::new());
    let searcher = Searcher::new(Arc::clone(&engine), Arc::clone(&tt));

    let test_positions = vec![
        (
            "STARTPOS",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "KIWIPETE",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
        ("ENDGAME_PAWN", "8/8/8/8/8/3k4/7p/3K4 w - - 0 1"),
        (
            "TACTICAL_FORK",
            "r3k2r/ppp2ppp/2n5/3p4/2B1P3/2N5/PPP2PPP/R3K2R w KQkq - 0 1",
        ),
        ("MATE_IN_1", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"),
        ("UNDERPROMO", "k7/P7/8/8/8/8/8/K7 w - - 0 1"),
    ];

    println!("=== KiyEngine V4 Omega - Extensive Standard Test ===");

    for (name, fen) in test_positions {
        println!("\nTesting Position: {}", name);
        println!("FEN: {}", fen);

        let board = Board::from_str(fen).map_err(|e| anyhow::anyhow!("FEN error: {:?}", e))?;
        let stop_flag = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();

        let start = Instant::now();
        // Lower depth for bench to stay responsive
        searcher.search_async(board, vec![], 3, Arc::clone(&stop_flag), tx);

        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(msg) => {
                let duration = start.elapsed();
                println!("Result: {} (took {:?})", msg, duration);

                let move_str = msg.split_whitespace().last().unwrap();
                let mv = ChessMove::from_str(move_str)
                    .map_err(|e| anyhow::anyhow!("Move error: {:?}", e))?;
                if board.legal(mv) {
                    println!("Status: ✅ LEGAL MOVE");
                } else {
                    println!("Status: ❌ ILLEGAL MOVE PRODUCED!");
                    return Err(anyhow::anyhow!(
                        "Engine produced illegal move {} in pos {}",
                        move_str,
                        name
                    ));
                }

                if name == "MATE_IN_1" && move_str == "e1e8" {
                    println!("Quality: ⭐ FOUND BEST MOVE (Mate)");
                }
            }
            Err(e) => {
                println!("Status: ❌ FAILED TO RESPOND ({:?})", e);
                return Err(anyhow::anyhow!("Engine timeout or crash in pos {}", name));
            }
        }
    }

    println!("\n=== Final Summary: ALL STANDARD TESTS PASSED ===");
    Ok(())
}
