//! Benchmark comparativo de entrenamiento CPU vs GPU (M1 Metal) en AIDEEN.
//!
//! Entrena secuencias idénticas para medir el throughput real en (tokens/s)
//! usando diferenciación implícita completa con Conjugate Gradient.
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
        println!("  Dataset demasiado corto para entrenamiento autoregresivo.");
        return;
    }
    let mut train_tokens = &tokens[..tokens.len() - 1];
    let mut targets = &tokens[1..];

    // Modo profesional: FULL por defecto.
    // Quick solo si se pide explícitamente: AIDEEN_BENCH_QUICK=1
    let quick_bench = std::env::var("AIDEEN_BENCH_QUICK").ok().as_deref() == Some("1");
    if quick_bench {
        let quick_len = train_tokens.len().min(16);
        train_tokens = &train_tokens[..quick_len];
        targets = &targets[..quick_len];
        println!(
            "  Modo QUICK (diagnóstico): {} tokens (AIDEEN_BENCH_QUICK=1)",
            quick_len
        );
    } else {
        println!("  Modo FULL (default): dataset completo");
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

    println!("  -- Configuración Utilizada --");
    println!("  vocab_size: {}", trainer.config.vocab_size);
    println!("  tokens: {}", train_tokens.len());
    println!("  D_R: {}", trainer.config.d_r);
    println!("  ctx_len: {}", trainer.config.ctx_len);
    println!("  epochs: {}", trainer.training_config.epochs);
    println!("  lr: {}", trainer.training_config.lr);
    println!("  train_deq: {}", trainer.config.train_deq);
    println!("  -----------------------------");

    // Configurando el Backend
    if use_gpu {
        #[cfg(feature = "wgpu")]
        {
            use aideen_backbone::gpu_backend::WgpuBlockBackend;
            if let Some(gpu) = WgpuBlockBackend::new_blocking() {
                trainer.reasoning.set_backend(gpu);
                println!("  Hardware Backend: Metal GPU ✅");
            } else {
                println!("  Hardware Backend: Fallback a CPU ❌");
                return;
            }
        }
        #[cfg(not(feature = "wgpu"))]
        {
            println!("  Hardware Backend: Error - wgpu feature desactivada ❌");
            return;
        }
    } else {
        println!("  Hardware Backend: Rust CPU (Secuencial) 💻");
    }

    println!("  Training context: {} tokens", train_tokens.len());

    // Entrenamiento cronometrado
    let epochs = trainer.training_config.epochs;
    let tokens_per_epoch = train_tokens.len();

    // Warmup iterativo (Sequence Fusing)
    let warmup_epochs = if quick_bench { 0 } else { 1 };
    if warmup_epochs > 0 {
        println!(
            "  [Warmup de {} epoch para asentar shaders...]",
            warmup_epochs
        );
        for w in 0..warmup_epochs {
            let tw = Instant::now();
            trainer.train_sequence(train_tokens, targets, false, 1e-4);
            println!(
                "    warmup {} listo en {}ms",
                w + 1,
                tw.elapsed().as_millis()
            );
        }
    } else {
        println!("  [Warmup desactivado en modo QUICK]");
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

    println!("\n  ── 📊 RESULTADOS {} ──", title);
    println!("  Tiempo total:    {:.2}s", total_secs);
    println!("  Tokens Totales:  {}", total_tokens_processed);
    println!("  V. Promedio:     {:.2} tokens/s E2E", avg_tok_s);
    println!("  Loss final:      {:.4}", final_loss);

    #[cfg(feature = "wgpu")]
    trainer.sync_inference_weights();

    println!("\n  ── 🧠 VALIDACIÓN DE APRENDIZAJE ──");
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
    println!("Este benchmark entrena secuencias de repetición y evalúa");
    println!("la velocidad del Implicit Gradient Descent en D_R=512.");

    // run_training_benchmark(false, "Entrenamiento en CPU"); // Deshabilitado, consume >10 mins

    #[cfg(feature = "wgpu")]
    run_training_benchmark(true, "Entrenamiento en Metal GPU");
}
