/// Integration tests for the Trainer struct.
///
/// Uses small dimensions (d_r=32, h_slots=2, vocab=100, max_deq_iters=3)
/// for fast execution.

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn small_config() -> ArchitectureConfig {
    ArchitectureConfig {
        d_m: 32,
        d_r: 32,
        d_c: 16,
        d_e: 16,
        d_sim: 32,
        h_slots: 2,
        vocab_size: 100,
        ctx_len: 16,
        max_deq_iters: 3,
        deq_epsilon: 1e-4,
        cg_iters: 4,
        train_deq: true,
        deq_grad_scale: 0.01,
        renorm_every_steps: 50,
        num_samples: 32,
        weight_decay: 0.01,
    }
}

fn make_trainer(seed: u64, lr: f32) -> Trainer {
    let cfg = small_config();
    let vocab_size = cfg.vocab_size;
    let d_r = cfg.d_r;

    let mut tok = Tokenizer::new_empty(vocab_size, cfg.clone());
    // Initialize embeddings with small random values
    {
        let mut rng = StdRng::seed_from_u64(seed);
        tok.embeddings = DMatrix::from_fn(vocab_size, d_r, |_, _| {
            (rng.gen::<f32>() - 0.5) * 0.02
        });
    }

    let mut trainer = Trainer::from_tokenizer_seeded(tok, lr, seed);
    trainer.config = cfg.clone();
    trainer.training_config.lr = lr;
    trainer.gpu_deq = None;
    trainer.gpu_lm = None;
    trainer.gpu_emb = None;
    trainer
}

fn random_tokens(seed: u64, n: usize, vocab_size: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0..vocab_size as u32)).collect()
}

#[test]
fn trainer_eval_loss_returns_finite() {
    let trainer = make_trainer(42, 1e-3);
    let tokens = random_tokens(99, 10, trainer.config.vocab_size);
    let loss = trainer.eval_loss(&tokens);
    assert!(loss.is_finite(), "eval_loss should return finite, got {}", loss);
    assert!(loss > 0.0, "eval_loss should be positive, got {}", loss);
}

#[test]
fn trainer_train_sequence_returns_finite() {
    let mut trainer = make_trainer(42, 1e-3);
    let tokens = random_tokens(99, 6, trainer.config.vocab_size);
    let tokens_in = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];
    let loss = trainer.train_sequence(tokens_in, targets, true, 1e-4);
    assert!(loss.is_finite(), "train_sequence should return finite, got {}", loss);
    assert!(loss > 0.0, "train_sequence should be positive, got {}", loss);
}

#[test]
fn trainer_loss_decreases_over_steps() {
    let mut trainer = make_trainer(42, 3e-3);
    // Use a short, fixed batch for overfitting
    let tokens: Vec<u32> = vec![1, 5, 10, 15, 20, 25, 30];
    let tokens_in = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];

    // First pass: record initial loss
    let initial_loss = trainer.train_sequence(tokens_in, targets, true, 1e-4);
    assert!(initial_loss.is_finite(), "initial loss not finite: {}", initial_loss);

    // Train for 30+ more steps on the same batch
    let mut final_loss = initial_loss;
    for _ in 0..35 {
        final_loss = trainer.train_sequence(tokens_in, targets, true, 1e-4);
    }

    assert!(
        final_loss.is_finite(),
        "final loss not finite: {}",
        final_loss
    );
    assert!(
        final_loss < initial_loss * 0.9,
        "Loss should decrease: initial={:.4}, final={:.4} (threshold={:.4})",
        initial_loss,
        final_loss,
        initial_loss * 0.9
    );
}
