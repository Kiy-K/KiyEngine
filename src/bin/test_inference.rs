use kiy_engine_v5_alpha::engine::Engine;

fn main() -> anyhow::Result<()> {
    println!("Loading engine...");
    let engine = Engine::new()?;
    println!("Engine loaded.");

    let test_cases = vec![vec![0], vec![100, 200, 300], vec![10, 20, 30, 40, 50]];

    for tokens in test_cases {
        println!("\n--- Testing sequence: {:?} ---", tokens);
        let (logits, eval) = engine.forward(&tokens)?;
        let best_move_id = logits.argmax(0)?.to_scalar::<u32>()?;
        println!("Best Move ID: {}", best_move_id);
        println!("Evaluation Score: {:.4}", eval);
    }

    Ok(())
}
