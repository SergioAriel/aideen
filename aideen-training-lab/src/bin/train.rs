//! CLI binary for training AIDEEN models.
//!
//! Usage:
//!   cargo run -p aideen-training --bin train --release -- [options]

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::checkpoint;
use aideen_training::data::DataLoader;
use aideen_training::schedule::CosineSchedule;
use aideen_training::trainer::Trainer;

use std::env;
use std::process;

struct CliArgs {
    data: String,
    val_data: Option<String>,
    tokenizer: String,
    checkpoint: String,
    d_r: usize,
    lr: f32,
    steps: usize,
    eval_every: usize,
    save_every: usize,
    ctx_len: usize,
    seed: u64,
    resume: bool,
}

impl CliArgs {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();

        let mut cli = CliArgs {
            data: "data/corpus/train.tokens.bin".to_string(),
            val_data: None,
            tokenizer: "aideen-backbone/tokenizer.json".to_string(),
            checkpoint: "checkpoints/latest".to_string(),
            d_r: 256,
            lr: 1e-3,
            steps: 100000,
            eval_every: 1000,
            save_every: 5000,
            ctx_len: 256,
            seed: 42,
            resume: false,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--data" => {
                    i += 1;
                    cli.data = args.get(i).cloned().unwrap_or_else(|| {
                        eprintln!("Error: --data requires a value");
                        process::exit(1);
                    });
                }
                "--val-data" => {
                    i += 1;
                    cli.val_data = Some(args.get(i).cloned().unwrap_or_else(|| {
                        eprintln!("Error: --val-data requires a value");
                        process::exit(1);
                    }));
                }
                "--tokenizer" => {
                    i += 1;
                    cli.tokenizer = args.get(i).cloned().unwrap_or_else(|| {
                        eprintln!("Error: --tokenizer requires a value");
                        process::exit(1);
                    });
                }
                "--checkpoint" => {
                    i += 1;
                    cli.checkpoint = args.get(i).cloned().unwrap_or_else(|| {
                        eprintln!("Error: --checkpoint requires a value");
                        process::exit(1);
                    });
                }
                "--d-r" => {
                    i += 1;
                    cli.d_r = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --d-r requires a valid integer");
                            process::exit(1);
                        });
                }
                "--lr" => {
                    i += 1;
                    cli.lr = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --lr requires a valid float");
                            process::exit(1);
                        });
                }
                "--steps" => {
                    i += 1;
                    cli.steps = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --steps requires a valid integer");
                            process::exit(1);
                        });
                }
                "--eval-every" => {
                    i += 1;
                    cli.eval_every = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --eval-every requires a valid integer");
                            process::exit(1);
                        });
                }
                "--save-every" => {
                    i += 1;
                    cli.save_every = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --save-every requires a valid integer");
                            process::exit(1);
                        });
                }
                "--ctx-len" => {
                    i += 1;
                    cli.ctx_len = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --ctx-len requires a valid integer");
                            process::exit(1);
                        });
                }
                "--seed" => {
                    i += 1;
                    cli.seed = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Error: --seed requires a valid integer");
                            process::exit(1);
                        });
                }
                "--resume" => {
                    cli.resume = true;
                }
                other => {
                    eprintln!("Unknown argument: {}", other);
                    eprintln!("Usage: train [--data PATH] [--val-data PATH] [--tokenizer PATH]");
                    eprintln!("             [--checkpoint PATH] [--d-r N] [--lr FLOAT] [--steps N]");
                    eprintln!("             [--eval-every N] [--save-every N] [--ctx-len N]");
                    eprintln!("             [--seed N] [--resume]");
                    process::exit(1);
                }
            }
            i += 1;
        }

        cli
    }
}

fn main() {
    let args = CliArgs::parse();

    println!("=== AIDEEN Training CLI ===");
    println!("  data:       {}", args.data);
    println!("  val_data:   {}", args.val_data.as_deref().unwrap_or("(split from data)"));
    println!("  tokenizer:  {}", args.tokenizer);
    println!("  checkpoint: {}", args.checkpoint);
    println!("  d_r:        {}", args.d_r);
    println!("  lr:         {}", args.lr);
    println!("  steps:      {}", args.steps);
    println!("  eval_every: {}", args.eval_every);
    println!("  save_every: {}", args.save_every);
    println!("  ctx_len:    {}", args.ctx_len);
    println!("  seed:       {}", args.seed);
    println!("  resume:     {}", args.resume);
    println!();

    // 1. Create architecture config
    let mut config = ArchitectureConfig::default();
    config.d_r = args.d_r;
    config.vocab_size = 64000;
    config.ctx_len = args.ctx_len;

    // 2. Load tokenizer from HuggingFace tokenizer.json
    println!("[1/5] Loading tokenizer from {} ...", args.tokenizer);
    let tokenizer = Tokenizer::from_file(&args.tokenizer, config.clone()).unwrap_or_else(|e| {
        eprintln!("Failed to load tokenizer: {}", e);
        process::exit(1);
    });

    // 3. Create or resume Trainer
    println!("[2/5] Initializing trainer ...");
    let mut trainer = if args.resume {
        println!("       Resuming from checkpoint: {}", args.checkpoint);
        let mut t = Trainer::from_tokenizer_seeded(tokenizer, args.lr, args.seed);
        checkpoint::load_checkpoint(&mut t, &args.checkpoint).unwrap_or_else(|e| {
            eprintln!("Failed to load checkpoint: {}", e);
            process::exit(1);
        });
        t
    } else {
        Trainer::from_tokenizer_seeded(tokenizer, args.lr, args.seed)
    };

    // 4. Load training data (and validation data if separate file provided)
    println!("[3/5] Loading data from {} ...", args.data);
    let (mut train_loader, mut val_loader) = if let Some(ref val_path) = args.val_data {
        // Separate train/val files — no redundant split needed
        let train_loader = DataLoader::from_file(&args.data, args.ctx_len, args.seed)
            .unwrap_or_else(|e| {
                eprintln!("Failed to load training data: {}", e);
                process::exit(1);
            });
        println!("       Loading validation data from {} ...", val_path);
        let val_loader = DataLoader::from_file(val_path, args.ctx_len, args.seed + 1)
            .unwrap_or_else(|e| {
                eprintln!("Failed to load validation data: {}", e);
                process::exit(1);
            });
        println!(
            "       Train tokens: {}, Val tokens: {}",
            train_loader.len(),
            val_loader.len()
        );
        (train_loader, val_loader)
    } else {
        // Legacy: single file, split 90/10
        let full_loader = DataLoader::from_file(&args.data, args.ctx_len, args.seed)
            .unwrap_or_else(|e| {
                eprintln!("Failed to load training data: {}", e);
                process::exit(1);
            });
        let (train_tokens, val_tokens) = DataLoader::split(&full_loader.tokens, 0.9);
        println!(
            "       Train tokens: {}, Val tokens: {}",
            train_tokens.len(),
            val_tokens.len()
        );
        drop(full_loader); // free ~14GB before allocating train/val
        let train_loader = DataLoader::new(train_tokens, args.ctx_len, args.seed);
        let val_loader = DataLoader::new(val_tokens, args.ctx_len, args.seed + 1);
        (train_loader, val_loader)
    };

    // 5. Set up cosine learning rate schedule
    let warmup_steps = args.steps / 20; // 5% warmup
    let min_lr = args.lr * 0.1;
    let schedule = CosineSchedule::new(warmup_steps, args.steps, args.lr, min_lr);

    // 6. Training loop
    println!("[4/5] Starting training for {} steps ...", args.steps);
    println!();

    for step in 0..args.steps {
        let lr = schedule.lr_at(step);
        trainer.training_config.lr = lr;

        // Sample a batch
        let (input, target) = train_loader.next_batch();

        // Train on the batch
        let loss = trainer.train_sequence(&input, &target, true, trainer.config.deq_epsilon);

        // Print loss every 100 steps
        if step % 100 == 0 {
            println!("[step {:>6}/{}] loss = {:.6}, lr = {:.6}", step, args.steps, loss, lr);
        }

        // Evaluate on validation set every eval_every steps
        if step > 0 && step % args.eval_every == 0 {
            let (val_input, _val_target) = val_loader.next_batch();
            let val_loss = trainer.eval_loss(&val_input);
            println!(
                "[step {:>6}/{}] === EVAL === val_loss = {:.6}",
                step, args.steps, val_loss
            );
        }

        // Save checkpoint every save_every steps
        if step > 0 && step % args.save_every == 0 {
            let ckpt_path = format!("{}/step_{}", args.checkpoint, step);
            println!("[step {:>6}/{}] Saving checkpoint to {} ...", step, args.steps, ckpt_path);
            trainer.sync_inference_weights();
            if let Err(e) = checkpoint::save_checkpoint(&trainer, &ckpt_path) {
                eprintln!("Warning: failed to save checkpoint: {}", e);
            }
        }
    }

    // 7. Save final checkpoint
    println!();
    println!("[5/5] Training complete. Saving final checkpoint ...");
    trainer.sync_inference_weights();
    let final_path = format!("{}/final", args.checkpoint);
    if let Err(e) = checkpoint::save_checkpoint(&trainer, &final_path) {
        eprintln!("Warning: failed to save final checkpoint: {}", e);
    }
    println!("Done. Final checkpoint saved to {}", final_path);
}
