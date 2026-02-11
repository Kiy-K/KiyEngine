//! KiyEngine NNUE Trainer
//!
//! Architecture: (768 → 512)×2 → SCReLU → 1 (with material output buckets)
//! Framework: Bullet (https://github.com/jw1912/bullet)
//! Data: Lichess broadcast games converted to bulletformat
//!
//! Usage:
//!   cargo run --release -- --data <path_to_data> [--test <path_to_test>]

use bullet_lib::{
    game::{inputs::Chess768, outputs::MaterialCount},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

const HIDDEN_SIZE: usize = 512;
const NUM_OUTPUT_BUCKETS: usize = 8;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse CLI args
    let mut data_path = String::from("data/train.bullet");
    let mut test_path: Option<String> = None;
    let mut net_id = String::from("kiyengine-nnue");
    let mut epochs: usize = 400;
    let mut start_lr: f32 = 0.001;
    let mut batch_size: usize = 16384;
    let mut superbatch_size: usize = 6_104;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { data_path = args[i + 1].clone(); i += 2; }
            "--test" => { test_path = Some(args[i + 1].clone()); i += 2; }
            "--id" => { net_id = args[i + 1].clone(); i += 2; }
            "--epochs" => { epochs = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { start_lr = args[i + 1].parse().unwrap(); i += 2; }
            "--batch-size" => { batch_size = args[i + 1].parse().unwrap(); i += 2; }
            "--superbatch-size" => { superbatch_size = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("=== KiyEngine NNUE Trainer ===");
    println!("Architecture: (768 → {})×2 → SCReLU → {} output buckets", HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);
    println!("Data:         {}", data_path);
    println!("Net ID:       {}", net_id);
    println!("Epochs:       {}", epochs);
    println!("LR:           {}", start_lr);
    println!("Batch size:   {}", batch_size);
    println!();

    // Build the network: (768 → 512)×2 → SCReLU → 8 output buckets
    // Output bucket selected by MaterialCount (total pieces → bucket index)
    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>);

    // CPU backend needs explicit thread count; GPU backend handles parallelism
    #[cfg(not(feature = "cuda"))]
    { builder = builder.use_threads(4); }

    let mut trainer = builder
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            // Transpose l1w for fast CPU inference (row-major per bucket)
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    // Training schedule
    let final_lr = start_lr * 0.3f32.powi(5);
    let schedule = TrainingSchedule {
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: superbatch_size,
            start_superbatch: 1,
            end_superbatch: epochs,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: start_lr, final_lr, final_superbatch: epochs },
        save_rate: 20,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: test_path.as_deref().map(|p| {
            bullet_lib::trainer::settings::TestDataset { path: p, freq: 20 }
        }),
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = DirectSequentialDataLoader::new(&[&data_path]);

    trainer.run(&schedule, &settings, &data_loader);

    println!("\n=== Training complete! ===");
    println!("Checkpoints saved to: checkpoints/");
    println!("To use in KiyEngine, copy the quantized .bin file to the engine directory as kiyengine.nnue");
}
