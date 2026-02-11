//! KiyEngine NNUE Trainer — Stockfish-style Architecture
//!
//! Architecture: (HalfKAv2_hm → 512)×2 → SCReLU → 8 output buckets
//!
//! Key improvements over basic Chess768:
//! - King-bucketed inputs (ChessBucketsMirrored): 10 king buckets with horizontal mirroring
//!   so the network learns different piece values depending on king position.
//! - Input factoriser: shared 768→HIDDEN weights added to each bucket's weights,
//!   enabling generalisation across king positions while still specialising.
//! - MaterialCount output buckets: 8 buckets for different material phases.
//!
//! Framework: Bullet (https://github.com/jw1912/bullet)
//!
//! Usage:
//!   cargo run --release -- --data <path_to_data> [--test <path_to_test>]

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

use acyclib::graph::{GraphNodeId, GraphNodeIdTy};

const HIDDEN_SIZE: usize = 512;
const NUM_OUTPUT_BUCKETS: usize = 8;

/// King bucket layout: maps 32 half-board squares (horizontally mirrored) to 10 buckets.
/// Mirroring: if king is on files e-h (file >= 4), the entire board is horizontally mirrored,
/// so we only need the left half (files a-d, 32 squares). Squares are indexed:
///   rank 1: 0-3, rank 2: 4-7, ..., rank 8: 28-31
///
/// Layout rationale (following Stockfish conventions):
///   - Rank 1 corners are distinct (castling rights matter)
///   - Central ranks are grouped (king in centre behaves similarly)
///   - Upper ranks grouped (king rarely there in middlegame)
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

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

    println!("=== KiyEngine NNUE Trainer (Stockfish-style Architecture) ===");
    println!("Architecture: (HalfKAv2_hm[{}kb] → {})×2 → SCReLU → {} output buckets",
             NUM_INPUT_BUCKETS, HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);
    println!("Data:         {}", data_path);
    println!("Net ID:       {}", net_id);
    println!("Epochs:       {}", epochs);
    println!("LR:           {}", start_lr);
    println!("Batch size:   {}", batch_size);
    println!();

    // Build the network with king-bucketed inputs + factoriser
    let builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>);

    let mut trainer = builder
        .save_format(&[
            // Merge factoriser weights into l0w before quantisation:
            // This produces a single (NUM_INPUT_BUCKETS * 768, HIDDEN_SIZE) weight matrix
            // where each bucket's 768 rows include the shared factoriser contribution.
            SavedFormat::id("l0w")
                .add_transform(|graph, _id, weights| {
                    let f_idx = graph.weight_idx("l0f").unwrap();
                    let f_id = GraphNodeId::new(f_idx, GraphNodeIdTy::Values);
                    let factoriser_base: Vec<f32> = graph.get(f_id).unwrap().borrow().get_dense_vals().unwrap();
                    let factoriser: Vec<f32> = factoriser_base.repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            // Transpose l1w for fast CPU inference (row-major per bucket)
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Input factoriser: shared 768→HIDDEN weights added to every bucket.
            // This helps the network generalise across king positions (Bullet convention).
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // Feature transformer: (NUM_INPUT_BUCKETS * 768) → HIDDEN per perspective
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // Output layer: 2*HIDDEN → NUM_OUTPUT_BUCKETS
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            // Inference: dual perspective with SCReLU activation
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    // Stricter weight clipping for factorised inputs (prevents overflow in i16)
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

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
    println!("Convert with: python3 scripts/convert_to_kynn.py checkpoints/<best>/quantised.bin");
}
