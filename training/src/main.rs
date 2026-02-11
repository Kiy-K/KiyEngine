//! KiyEngine NNUE Trainer
//!
//! Architecture: (768 → 512)×2 → SCReLU → 1 (with material output buckets)
//! Framework: Bullet (https://github.com/jw1912/bullet)
//! Data: Lichess broadcast games converted to bulletformat
//!
//! Usage:
//!   cargo run --release -- --data <path_to_data> [--test <path_to_test>] [--resume <checkpoint>]

use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse CLI args
    let mut data_path = String::from("data/lichess_broadcast.bullet");
    let mut test_path: Option<String> = None;
    let mut net_id = String::from("kiyengine-nnue");
    let mut epochs = 400;
    let mut lr = 0.001;
    let mut batch_size = 16384;
    let mut superbatch_size = 6_104; // ~100M positions per superbatch

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { data_path = args[i + 1].clone(); i += 2; }
            "--test" => { test_path = Some(args[i + 1].clone()); i += 2; }
            "--id" => { net_id = args[i + 1].clone(); i += 2; }
            "--epochs" => { epochs = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i + 1].parse().unwrap(); i += 2; }
            "--batch-size" => { batch_size = args[i + 1].parse().unwrap(); i += 2; }
            "--superbatch-size" => { superbatch_size = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("=== KiyEngine NNUE Trainer ===");
    println!("Architecture: (768 → 512)×2 → SCReLU → 1");
    println!("Data:         {}", data_path);
    println!("Net ID:       {}", net_id);
    println!("Epochs:       {}", epochs);
    println!("LR:           {}", lr);
    println!("Batch size:   {}", batch_size);
    println!();

    // Data loader — reads .bullet files (bulletformat: 32 bytes per position)
    let data_loader = DirectSequentialDataLoader::new(&[&data_path]);

    // Training schedule
    use bullet_lib::value::schedule::{TrainingSchedule, TrainingSteps, WdlScheduler, LrScheduler};
    let schedule = TrainingSchedule {
        net_id,
        eval_scale: 400.0, // centipawn scaling factor
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: superbatch_size,
            start_superbatch: 1,
            end_superbatch: epochs,
        },
        wdl_scheduler: WdlScheduler::Linear { start: 0.2, end: 0.5 },
        lr_scheduler: LrScheduler::Step {
            start: lr,
            gamma: 0.3,
            step: epochs / 4, // drop LR 4 times
        },
        save_rate: 20,
    };

    // Local settings
    use bullet_lib::value::schedule::LocalSettings;
    let settings = LocalSettings {
        threads: 4,
        test_set: test_path.as_ref().map(|p| {
            use bullet_lib::value::schedule::TestDataset;
            TestDataset {
                path: p.clone(),
                freq: 20,
            }
        }),
        output_directory: "checkpoints",
        batch_queue_size: 512,
    };

    // Build the network: (768 → 512)×2 → SCReLU → 1
    //
    // Chess768: standard HalfKP-like 768 inputs (64 squares × 6 piece types × 2 colors)
    // Dual perspective: separate accumulators for STM and NSTM
    // SCReLU activation: squared clipped ReLU for better gradient flow
    // Output buckets: MaterialCount8 for material-aware evaluation
    //
    // Quantization: i16 × 255 for all layers (matches Stockfish NNUE format)

    use bullet_lib::nn::optimiser::AdamWParams;
    use bullet_lib::value::save::SavedFormat;
    use bullet_lib::game::outputs::MaterialCount8;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(Chess768)
        .output_buckets(MaterialCount8)
        .optimiser(AdamW {
            params: AdamWParams {
                decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                min_weight: -1.98,
                max_weight: 1.98,
            },
        })
        .save_format(&[
            // Feature transformer: 768 → 512 (perspective)
            SavedFormat::id("l0w").quantise::<i16>(255),
            SavedFormat::id("l0b").quantise::<i16>(255),
            // Output layer: 1024 → 1 (after SCReLU doubles the hidden dim)
            SavedFormat::id("l1w").quantise::<i16>(64),
            SavedFormat::id("l1b").quantise::<i16>(64 * 255),
        ])
        .loss_fn(|output, targets| {
            // Sigmoid MSE loss
            let diff = output - targets;
            diff.abs_pow(2.0)
        })
        .build(|builder, stm, nstm, buckets| {
            // Layer 0: Feature transformer (768 → 512) applied to both perspectives
            let l0 = builder.new_affine("l0", 512);
            let stm_hidden = l0.forward(stm).screlu();
            let nstm_hidden = l0.forward(nstm).screlu();

            // Concatenate both perspectives: 512 + 512 = 1024
            let combined = stm_hidden.concat(nstm_hidden);

            // Layer 1: Output (1024 → 1 per bucket)
            let l1 = builder.new_affine("l1", 1);
            l1.forward(combined).select(buckets)
        });

    // Run training
    trainer.run(&schedule, &settings, &data_loader);

    println!("\n=== Training complete! ===");
    println!("Checkpoints saved to: checkpoints/");
    println!("To use in KiyEngine, copy the quantized .bin file to the engine directory.");
}
