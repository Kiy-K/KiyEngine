use chess::Board;
use kiy_engine_v5_alpha::engine::Engine;
use kiy_engine_v5_alpha::search::lazy_smp::Searcher;
use kiy_engine_v5_alpha::search::tt::AtomicTT;
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};
use std::time::Instant;

fn main() {
    println!("Initializing KiyEngine Benchmark...");

    // Setup
    let engine = Arc::new(Engine::new().expect("Failed to load engine"));
    let tt = Arc::new(AtomicTT::new(64)); // 64MB hash
    let searcher = Searcher::new(engine, tt);

    // Use multi-threading if available
    let _threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // searcher.threads = threads; // Struct field is public?
    // Wait, I need to check if I made `threads` public in `lazy_smp.rs`.
    // Yes, `pub threads: usize`.

    // Ideally we bench with 1 thread for stable NPS comparison, or max for throughput.
    // Let's do 1 thread for "Single Core NPS" which is standard metric.
    // Actually, let's test both? No, assume 1 thread for standard bench.

    // Position: Kiwipete (Tricky)
    // r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
    // Position: Startpos

    let _iterations = 3;
    let depth = 12; // Adjusted for speed

    println!("Benchmarking to Depth {} on Startpos...", depth);

    let board = Board::default();
    let history = vec![];
    let (tx, rx) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));

    let start = Instant::now();

    searcher.search_async(
        board,
        history,
        vec![], // Empty Zobrist history
        depth as u8,
        0,
        0,
        0,
        0,
        0,
        stop,
        tx,
    );

    // Consume output and wait
    let mut nodes = 0;
    while let Ok(msg) = rx.recv() {
        // Parse info string for nodes
        if msg.starts_with("info") {
            // "info depth {} score cp {} nodes {} ..."
            let parts: Vec<&str> = msg.split_whitespace().collect();
            for i in 0..parts.len() {
                if parts[i] == "nodes" {
                    if let Ok(n) = parts[i + 1].parse::<u64>() {
                        nodes = n;
                    }
                }
            }
        }
        if msg.starts_with("bestmove") {
            break;
        }
    }

    let duration = start.elapsed();
    let nps = (nodes as f64 / duration.as_secs_f64()) as u64;

    println!("--------------------------------------------------");
    println!("Depth: {}", depth);
    println!("Nodes: {}", nodes);
    println!("Time:  {:.3}s", duration.as_secs_f64());
    println!("NPS:   {}", nps);
    println!("--------------------------------------------------");
}
