//! Comparative CPU vs GPU (M1 Metal) training benchmark for AIDEEN.
//!
//! Trains identical sequences to measure real throughput in (tokens/s)
//! using full implicit differentiation with Conjugate Gradient.
//!
//! Uso:
//!   cargo run --release --features wgpu -p aideen-training --bin benchmark

use std::time::Instant;

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;

const PROMPT: &str = "\
el equilibrio....... \
el equilibrio....... \
el equilibrio....... \
la red.............. \
la red.............. \
la red.............. \
aideen inteligencia. \
aideen inteligencia. \
aideen inteligencia. \
";

fn run_training_benchmark(use_gpu: bool, title: &str) {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║ BENCHMARK: {:<40} ║", title);
    println!("╚══════════════════════════════════════════════════════╝");

    let config = ArchitectureConfig::default();
    let tok = Tokenizer::from_text(PROMPT, config);
    let tokens = tok.encode(PROMPT);
    if tokens.len() < 2 {
        println!("  Dataset too short for autoregressive training.");
        return;
    }
    let mut train_tokens = &tokens[..tokens.len() - 1];
    let mut targets = &tokens[1..];

    // Professional mode: FULL by default.
    // Quick only if explicitly requested: AIDEEN_BENCH_QUICK=1
    let quick_bench = std::env::var("AIDEEN_BENCH_QUICK").ok().as_deref() == Some("1");
    if quick_bench {
        let quick_len = train_tokens.len().min(16);
        train_tokens = &train_tokens[..quick_len];
        targets = &targets[..quick_len];
        println!(
            "  QUICK mode (diagnostic): {} tokens (AIDEEN_BENCH_QUICK=1)",
            quick_len
        );
    } else {
        println!("  FULL mode (default): complete dataset");
    }

    let mut trainer = Trainer::from_tokenizer(tok, 0.05); // LR elevado para overfit rápido
    trainer.config.train_deq = true;
    trainer.training_config.epochs = if quick_bench { 8 } else { 200 }; // 200 épocas
    trainer.config.max_deq_iters = if quick_bench { 4 } else { 18 };
    trainer.config.adj_iters = if quick_bench { 4 } else { 15 };
    trainer.config.deq_epsilon = if quick_bench { 1e-4 } else { 1e-4 };
    trainer.config.deq_grad_scale = if quick_bench { 0.001 } else { 0.01 };
    trainer.config.renorm_every_steps = if quick_bench { 8 } else { 16 };
    trainer.config.ctx_len = if quick_bench { 12 } else { 24 };

    println!("  -- Configuration Used --");
    println!("  vocab_size: {}", trainer.config.vocab_size);
    println!("  tokens: {}", train_tokens.len());
    println!("  D_R: {}", trainer.config.d_r);
    println!("  ctx_len: {}", trainer.config.ctx_len);
    println!("  epochs: {}", trainer.training_config.epochs);
    println!("  lr: {}", trainer.training_config.lr);
    println!("  train_deq: {}", trainer.config.train_deq);
    println!("  -----------------------------");

    // Configuring the Backend
    if use_gpu {
        #[cfg(feature = "wgpu")]
        {
            use aideen_backbone::gpu_backend::WgpuBlockBackend;
            if let Some(gpu) = WgpuBlockBackend::new_blocking() {
                trainer.reasoning.set_backend(gpu);
                println!("  Hardware Backend: Metal GPU ✅");
            } else {
                println!("  Hardware Backend: Fallback to CPU ❌");
                return;
            }
        }
        #[cfg(not(feature = "wgpu"))]
        {
            println!("  Hardware Backend: Error - wgpu feature disabled ❌");
            return;
        }
    } else {
        println!("  Hardware Backend: Rust CPU (Secuencial) 💻");
    }

    println!("  Training context: {} tokens", train_tokens.len());

    // Timed training
    let epochs = trainer.training_config.epochs;
    let tokens_per_epoch = train_tokens.len();

    // Warmup iterativo (Sequence Fusing)
    let warmup_epochs = if quick_bench { 0 } else { 1 };
    if warmup_epochs > 0 {
        println!("  [Warmup of {} epoch to settle shaders...]", warmup_epochs);
        for w in 0..warmup_epochs {
            let tw = Instant::now();
            trainer.train_sequence(train_tokens, targets, false, 1e-4);
            println!(
                "    warmup {} done in {}ms",
                w + 1,
                tw.elapsed().as_millis()
            );
        }
    } else {
        println!("  [Warmup disabled in QUICK mode]");
    }

    let t0 = Instant::now();
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        let t_epoch = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_steps = 0;

        // Dataloader Autoregresivo (Sliding Window)
        let ctx_len = trainer.config.ctx_len;
        let step = ctx_len / 2; // 50% solapamiento para mejor aprendizaje de transiciones

        for i in (0..train_tokens.len().saturating_sub(ctx_len)).step_by(step) {
            let window_tokens = &train_tokens[i..i + ctx_len];
            let window_targets = &targets[i..i + ctx_len];

            epoch_loss += trainer.train_sequence(window_tokens, window_targets, false, 1e-4);
            num_steps += 1;
        }

        // Si la secuencia es muy corta y no entró en el loop, o para el resto final:
        if num_steps == 0 && train_tokens.len() >= 2 {
            epoch_loss = trainer.train_sequence(train_tokens, targets, false, 1e-4);
            num_steps = 1;
        }

        final_loss = epoch_loss / num_steps.max(1) as f32;
        let epoch_ms = t_epoch.elapsed().as_millis();
        let tok_s = (tokens_per_epoch as f64) / (epoch_ms as f64 / 1000.0);

        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!(
                "  Epoch {:>2}/{} | Loss: {:.4} | Time: {}ms | Speed: {:.1} tok/s",
                epoch, epochs, final_loss, epoch_ms, tok_s
            );
        }
    }

    let total_secs = t0.elapsed().as_secs_f64();
    let total_tokens_processed = epochs * tokens_per_epoch;
    let avg_tok_s = total_tokens_processed as f64 / total_secs;

    println!("\n  ── 📊 RESULTS {} ──", title);
    println!("  Total time:      {:.2}s", total_secs);
    println!("  Total tokens:    {}", total_tokens_processed);
    println!("  Avg. speed:      {:.2} tokens/s E2E", avg_tok_s);
    println!("  Final loss:      {:.4}", final_loss);

    #[cfg(feature = "wgpu")]
    trainer.sync_inference_weights();

    println!("\n  ── 🧠 LEARNING VALIDATION ──");
    for prompt in &["el equilibrio.", "la red......", "aideen int"] {
        // Sample with temperature: 0.8, top_p: 0.9, top_k: 40, repetition_penalty: 1.1
        let generated = trainer.generate(prompt, 60, 0.8, 0.9, 40, 1.1);
        println!("  \"{}\" → \"{}\"", prompt, generated.trim());
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║      AIDEEN MVP: CPU vs GPU Training Benchmark       ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("This benchmark trains repetition sequences and evaluates");
    println!("the speed of Implicit Gradient Descent at D_R=512.");

    // run_training_benchmark(false, "CPU Training"); // Disabled, takes >10 mins

    #[cfg(feature = "wgpu")]
    run_training_benchmark(true, "Metal GPU Training");
}
