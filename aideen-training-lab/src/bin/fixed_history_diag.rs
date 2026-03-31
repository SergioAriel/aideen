use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;

fn make_trainer(text: &str, seed: u64) -> (Trainer, Vec<u32>) {
    let mut cfg = ArchitectureConfig::default();
    cfg.d_r = 64;
    cfg.ctx_len = 6;
    cfg.max_deq_iters = 6;
    cfg.adj_iters = 3;
    let tok = Tokenizer::from_text(text, cfg.clone());
    let tokens = tok.encode(text);
    let mut trainer = Trainer::from_tokenizer_seeded(tok, 5e-3, seed);
    trainer.config = cfg;
    trainer.training_config.lr = 5e-3;
    trainer.training_config.deq_lr_mult = 0.01;
    (trainer, tokens)
}

fn train_quick(trainer: &mut Trainer, tokens: &[u32], epochs: usize, eps: f32) {
    let inputs = &tokens[..tokens.len().saturating_sub(1)];
    let targets = &tokens[1..];
    for epoch in 0..epochs {
        let loss = trainer.train_sequence(inputs, targets, true, eps);
        if epoch == 0 || epoch + 1 == epochs {
            println!("    epoch {:>2}/{epochs} loss={:.4}", epoch + 1, loss);
        }
    }
    #[cfg(feature = "wgpu")]
    trainer.sync_inference_weights();
}

fn eval_with_flag(trainer: &Trainer, tokens: &[u32], fixed: bool) -> f32 {
    if fixed {
        std::env::set_var("AIDEEN_DEQ_FIXED_HISTORY_REFERENCE", "1");
    } else {
        std::env::remove_var("AIDEEN_DEQ_FIXED_HISTORY_REFERENCE");
    }
    let out = trainer.eval_loss(tokens);
    std::env::remove_var("AIDEEN_DEQ_FIXED_HISTORY_REFERENCE");
    out
}

fn run_case(name: &str, text: &str, seed: u64, epochs: usize) {
    println!();
    println!("== {} ==", name);
    let (mut trainer, tokens) = make_trainer(text, seed);
    if tokens.len() < 3 {
        println!("  skip: dataset corto");
        return;
    }

    train_quick(&mut trainer, &tokens, epochs, 1e-4);

    let plain = eval_with_flag(&trainer, &tokens, false);
    let fixed = eval_with_flag(&trainer, &tokens, true);

    let saved_ctx = trainer.config.ctx_len;
    trainer.config.ctx_len = 1;
    let plain_ctx1 = eval_with_flag(&trainer, &tokens, false);
    let fixed_ctx1 = eval_with_flag(&trainer, &tokens, true);
    trainer.config.ctx_len = saved_ctx;

    println!("  default_ctx plain={:.6} fixed={:.6} delta={:.6}", plain, fixed, fixed - plain);
    println!(
        "  ctx_len=1   plain={:.6} fixed={:.6} delta={:.6}",
        plain_ctx1,
        fixed_ctx1,
        fixed_ctx1 - plain_ctx1
    );
}

fn main() {
    let natural = "la memoria temporal estabiliza el contexto y la atencion entre slots. \
la memoria temporal estabiliza el contexto y la atencion entre slots. ";
    let synthetic = "xayxbyxayxbyxayxbyxayxbyxayxbyxayxbyxayxbyxayxby";

    println!("Fixed-history diagnostic");
    println!("Comparando plain vs fixed-history sobre el mismo trainer entrenado.");
    run_case("natural_short", natural, 42, 2);
    run_case("synthetic_ambiguous_ctx1", synthetic, 1337, 4);
}
