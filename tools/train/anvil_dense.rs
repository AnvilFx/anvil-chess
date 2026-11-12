// Copy into a bullet checkout (github.com/jw1912/bullet, tested at da426a4) as examples/anvil_dense.rs.
// Anvil's dense net: (768x4 -> 1024)x2 -> pairwise CReLU -> (1024 -> 16 -> 32 -> 1)x8, SCReLU
// between the dense layers, 4 horizontally mirrored king buckets (factorised), output buckets
// by piece count. Saves plain f32 weights; tools/gen_net.py quantises them.
//
//   cargo r -r --features cuda --example anvil_dense -- <data file> [superbatches] [net id] [batches] [resume superbatch]
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

const HL_SIZE: usize = 1024;
const L1_SIZE: usize = 16;
const L2_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 8;

// King square (files a-d, the e-h half is mirrored onto it) -> input bucket, rank 1 first.
// Must match nets/<net>.json "king_buckets" so the engine indexes the same way.
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];
const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dataset_path = args.get(1).map(String::as_str).unwrap_or("data/kb4-train.data");
    let superbatches: usize = args.get(2).map_or(60, |s| s.parse().expect("superbatches"));
    let net_id = args.get(3).cloned().unwrap_or_else(|| "anvil-dense".to_string());
    // Optional, for smoke tests: batches per superbatch.
    let batches: usize = args.get(4).map_or(6104, |s| s.parse().expect("batches per superbatch"));
    // Optional: resume from checkpoints/<net id>-<n> at superbatch n + 1.
    let resume: Option<usize> = args.get(5).map(|s| s.parse().expect("resume superbatch"));

    // Kept equal to the plain config so the two nets differ only in the dense layers.
    let wdl_proportion = 0.3;
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        // Factoriser folded into every bucket; dense weights transposed to [out][in], so each
        // output bucket's rows are contiguous.
        .save_format(&[
            SavedFormat::id("l0w").transform(|store, weights| {
                let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
            }),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w").transpose(),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Shared weights every bucket trains, so rare buckets start from a sensible net.
            let l0f = builder.new_weights("l0f", Shape::new(HL_SIZE, 768), InitSettings::Zeroed);
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HL_SIZE);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + l0f.repeat(NUM_INPUT_BUCKETS);

            let l1 = builder.new_affine("l1", HL_SIZE, NUM_OUTPUT_BUCKETS * L1_SIZE);
            let l2 = builder.new_affine("l2", L1_SIZE, NUM_OUTPUT_BUCKETS * L2_SIZE);
            let l3 = builder.new_affine("l3", L2_SIZE, NUM_OUTPUT_BUCKETS);

            // Pairwise CReLU: each half of a side's accumulator multiplies the other half.
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, HL_SIZE / 2) * ft(stm_inputs, HL_SIZE / 2, HL_SIZE);
            let ntm_hidden = ft(ntm_inputs, 0, HL_SIZE / 2) * ft(ntm_inputs, HL_SIZE / 2, HL_SIZE);

            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    // Bucket weight plus factoriser must stay within the clip range.
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    if let Some(n) = resume {
        trainer.load_from_checkpoint(&format!("checkpoints/{net_id}-{n}"));
    }

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: batches,
            start_superbatch: resume.map_or(1, |n| n + 1),
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let dataloader = DirectSequentialDataLoader::new(&[dataset_path]);

    trainer.run(&schedule, &settings, &dataloader);
}
