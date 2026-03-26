//! Binary de entrenamiento de AIDEEN.
//!
//! # Notas de arquitectura para futuros LLMs que lean esto
//!
//! ## Cuantización (1.58b / BitNet)
//! FASE 1: Entrenar en float32. FASE 2: reentrenar desde cero con `ternary = true` (QAT/STE).
//! NO hacer post-training quantization — a 1.58b destruye la calidad. El STE ya está
//! implementado en los shaders GPU (embedding_train.wgsl, lm_train.wgsl, fused_deq_update.wgsl).
//!
//! ## Atención uniforme — riesgo crítico a monitorear
//! En pesos random, `attn_ent = log(8) = 2.079` siempre (entropía máxima, slots idénticos).
//! Para que el DEQ sea poderoso, los slots DEBEN especializarse durante el entrenamiento
//! (attn_ent debe bajar de 2.079). Si no ocurre, los 8 slots colapsan a lo mismo y la
//! capacidad de razonamiento paralelo se desperdicia completamente.
//! Monitorear `attn_ent` en GPU-DEBUG. Si no baja después de ~1000 pasos reales, investigar.
//!
//! ## Modos de uso:
//!   # Modo rápido — dataset.txt existente (pequeño, para tests)
//!   cargo run --release --features wgpu -p aideen-training --bin train
//!
//!   # Modo archivo grande — cualquier .txt (streaming, sin límite de RAM)
//!   cargo run --release --features wgpu -p aideen-training --bin train -- --file path/to/corpus.txt
//!
//!   # Modo checkpoint — continuar desde un checkpoint previo
//!   cargo run --release --features wgpu -p aideen-training --bin train -- --file corpus.txt --resume model

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;

use std::{env, fs};

fn print_banner(config: &ArchitectureConfig) {
    println!();
    println!("╔═══════════════════════════════════════════════╗");
    println!("║     AIDEEN — Semba Engine v3                 ║");
    println!(
        "║     Slot-Equilibrium Mamba (D_R={})         ║",
        config.d_r
    );
    println!("╚═══════════════════════════════════════════════╝");
    println!();
}

fn setup_gpu(_trainer: &mut Trainer) {
    #[cfg(feature = "wgpu")]
    {
        println!("  Backend: GPU (Metal) ✅ [Auto-managed]");
    }
    #[cfg(not(feature = "wgpu"))]
    println!("  Backend: CPU (compile with --features wgpu for GPU)");
}

fn main() {
    // ── Parse args ──────────────────────────────────────────────────────────
    let args: Vec<String> = env::args().collect();
    let mut large_file: Option<String> = None;
    let mut resume_path: Option<String> = None;
    let mut epochs: usize = 1;
    let mut log_every: usize = 3;
    let mut save_every: usize = 10;
    let mut freeze_deq = false;
    let mut freeze_emb = false;
    let mut freeze_lm = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--file" | "--train" => {
                i += 1;
                large_file = args.get(i).cloned();
            }
            "--resume" => {
                i += 1;
                resume_path = args.get(i).cloned();
            }
            "--epochs" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    epochs = v.parse().unwrap_or(80);
                }
            }
            "--log-every" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    log_every = v.parse().unwrap_or(3);
                }
            }
            "--save-every" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    save_every = v.parse().unwrap_or(10);
                }
            }
            "--freeze-deq" => freeze_deq = true,
            "--freeze-emb" => freeze_emb = true,
            "--freeze-lm" => freeze_lm = true,
            _ => {}
        }
        i += 1;
    }

    // Require explicit --file to avoid silent fallback to the tiny dataset.
    let Some(ref txt_path) = large_file else {
        eprintln!("ERROR: falta --file <corpus.txt>. El modo fallback fue deshabilitado.");
        std::process::exit(2);
    };
    run_large_file(
        txt_path,
        resume_path,
        epochs,
        log_every,
        save_every,
        freeze_deq,
        freeze_emb,
        freeze_lm,
    );
}

fn run_small_dataset(
    resume_path: Option<String>,
    epochs: usize,
    log_every: usize,
    freeze_deq: bool,
    freeze_emb: bool,
    freeze_lm: bool,
) {
    let corpus = fs::read_to_string("aideen-backbone/dataset.txt")
        .or_else(|_| fs::read_to_string("dataset.txt"))
        .unwrap_or_else(|_| "AIDEEN engine running basic logic.".to_string());

    let config_default = ArchitectureConfig::default();

    let tok_path = find_tokenizer_path();
    let mut tok = if let Some(ref path) = tok_path {
        println!("  Tokenizer: BPE ({path}) ✅");
        Tokenizer::from_file(path, config_default.clone()).expect("Failed to load tokenizer.json")
    } else {
        println!("  Tokenizer: Char-level (fallback) ⚠️");
        Tokenizer::from_text(&corpus, config_default.clone())
    };

    let mut config = ArchitectureConfig::default();
    config.vocab_size = tok.vocab_size();
    config.max_deq_iters = 15;
    tok.config = config.clone();

    print_banner(&config);
    let tokens = tok.encode(&corpus);
    println!(
        "  Corpus: {} chars → {} tokens, vocab={}",
        corpus.len(),
        tokens.len(),
        tok.vocab_size()
    );

    let lr = std::env::var("AIDEEN_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0001);

    let mut trainer = if let Some(ref base) = resume_path {
        println!("  Resumiendo desde checkpoint: {base}");
        Trainer::load_checkpoint(base).expect("Error cargando checkpoint")
    } else {
        let mut t = Trainer::from_tokenizer(tok, lr);
        t.training_config.lr_min = lr / 10.0;
        t.training_config.warmup_epochs = 3;
        t.training_config.epochs = epochs;
        // FASE 1: Entrenar en float32 hasta que los resultados sean buenos.
        // FASE 2: Cuando el modelo float32 funcione bien, cambiar a `true` para
        //         reentrenar desde cero con QAT (Quantization-Aware Training, STE).
        //         NO hacer post-training quantization — 1.58b requiere QAT desde el inicio.
        t.training_config.ternary = false;
        t
    };

    trainer.frozen_deq = freeze_deq;
    trainer.frozen_emb = freeze_emb;
    trainer.frozen_lm = freeze_lm;
    if freeze_deq {
        println!("  [Ablation] DEQ Frozen ❄️");
    }
    if freeze_emb {
        println!("  [Ablation] Embeddings Frozen ❄️");
    }

    setup_gpu(&mut trainer);
    if trainer.training_config.ternary {
        println!("  Bit-Dieta (Ternary 1.58-bit) activo 🔥  [STE durante training]");
    }

    let t0 = std::time::Instant::now();
    trainer.train_on_tokens(&tokens, epochs, log_every);
    println!("\n  Tiempo total: {:.1}s", t0.elapsed().as_secs_f32());
    println!("  Spectral norms: {:?}", trainer.reasoning.spectral_norms());

    save_and_generate(&mut trainer, "model");
}

/// Entrena sobre un .txt grande usando streaming (sin límite de RAM).
/// Tokeniza el texto, lo escribe a un .bin temporal y llama train_on_file.
fn run_large_file(
    txt_path: &str,
    resume_path: Option<String>,
    epochs: usize,
    log_every: usize,
    save_every: usize,
    freeze_deq: bool,
    freeze_emb: bool,
    freeze_lm: bool,
) {
    println!("  Modo: archivo grande → {txt_path}");

    // ── Resolver ruta real del dataset ─────────────────────────────────────
    let (resolved_path, corpus) = match fs::read_to_string(txt_path) {
        Ok(c) => (txt_path.to_string(), Some(c)),
        Err(_) => {
            let filename = std::path::Path::new(txt_path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();
            let alt_paths = [
                format!("../aideen-backbone/{}", filename),
                format!("aideen-backbone/{}", filename),
            ];
            let mut found = None;
            for p in alt_paths {
                if let Ok(c) = fs::read_to_string(&p) {
                    found = Some((p, Some(c)));
                    break;
                }
            }
            found.expect("No se puede encontrar el dataset en ninguna ruta conocida.")
        }
    };
    let txt_path = resolved_path;

    // ── Construir tokenizer ────────────────────────────────────────────────
    let config_default = ArchitectureConfig::default();
    let tok_path = find_tokenizer_path();
    let mut tok = if let Some(ref path) = tok_path {
        println!("  Tokenizer: BPE ({path}) ✅");
        Tokenizer::from_file(path, config_default.clone()).expect("Failed to load tokenizer.json")
    } else {
        let corpus = corpus.as_ref().expect("Tokenizer char-level requiere corpus en memoria.");
        println!("  Tokenizer: Char-level — escaneando vocab...");
        Tokenizer::from_text(corpus, config_default.clone())
    };

    tok.config.vocab_size = tok.vocab_size();
    tok.config.ctx_len = std::env::var("AIDEEN_CTX_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);
    tok.config.train_deq = true;

    let vocab_size = tok.vocab_size();

    // ── Tokenizar y escribir .bin (o reutilizar cache) ─────────────────────
    let bin_path = format!("{txt_path}.tokens.bin");
    let txt_meta = fs::metadata(&txt_path).ok();
    let bin_meta = fs::metadata(&bin_path).ok();
    let use_cache = bin_meta
        .as_ref()
        .and_then(|b| b.modified().ok())
        .zip(txt_meta.as_ref().and_then(|t| t.modified().ok()))
        .map(|(b, t)| b >= t)
        .unwrap_or(false);

    if use_cache && tok_path.is_some() {
        let bin_bytes = bin_meta.unwrap().len() as usize;
        let tokens_len = bin_bytes / 4;
        println!(
            "  Cache OK: reutilizando {bin_path} ({} tokens, vocab={})",
            tokens_len, vocab_size
        );
    } else {
        let corpus = corpus.as_ref().expect("Corpus en memoria requerido para tokenizar.");
        println!("  Tokenizando {txt_path} → {bin_path} ...");
        {
            use std::io::Write;
            let tokens = tok.encode(corpus);
            let byte_data: &[u8] = bytemuck::cast_slice(&tokens);

            // Asegurar que el directorio padre existe
            if let Some(parent) = std::path::Path::new(&bin_path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }

            let mut f = fs::File::create(&bin_path).expect("No se puede crear .bin");
            f.write_all(byte_data).expect("Error escribiendo .bin");
            println!(
                "  {} chars → {} tokens, vocab={} → {:.2} MB",
                corpus.len(),
                tokens.len(),
                vocab_size,
                byte_data.len() as f64 / 1_048_576.0
            );
        }
    }

    let lr = std::env::var("AIDEEN_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0001);
    let checkpoint_base = "model_large";

    let mut trainer = if let Some(ref base) = resume_path {
        println!("  Resumiendo desde checkpoint: {base}");
        Trainer::load_checkpoint(base).expect("Error cargando checkpoint")
    } else {
        let mut t = Trainer::from_tokenizer(tok, lr);
        t.training_config.lr_min = lr / 10.0;
        t.training_config.warmup_epochs = 0;
        t.training_config.epochs = epochs;
        t.training_config.ternary = false;
        t
    };

    trainer.frozen_deq = freeze_deq;
    trainer.frozen_emb = freeze_emb;
    trainer.frozen_lm = freeze_lm;
    if freeze_deq {
        println!("  [Ablation] DEQ Frozen ❄️");
    }
    if freeze_emb {
        println!("  [Ablation] Embeddings Frozen ❄️");
    }

    setup_gpu(&mut trainer);
    if trainer.training_config.ternary {
        println!("  Bit-Dieta (Ternary 1.58-bit) activo 🔥  [STE durante training]");
    }
    println!();

    // EOS token:
    // En este pipeline no inyectamos un token EOS explícito en el stream de training.
    // Usar 2 para BPE fragmentaba por un token común (p.ej. '#') y anulaba la pérdida.
    // 0 desactiva el split por EOS en train_on_file.
    let eos_token: u32 = 0;

    let t0 = std::time::Instant::now();
    trainer
        .train_on_file(
            &bin_path,
            epochs,
            log_every,
            eos_token,
            save_every,
            checkpoint_base,
        )
        .expect("Error durante train_on_file");

    println!("\n  Tiempo total: {:.1}s", t0.elapsed().as_secs_f32());
    println!("  Spectral norms: {:?}", trainer.reasoning.spectral_norms());

    if save_every == 0 {
        println!("  [Skip] save/generate desactivado (save_every=0)");
        return;
    }
    save_and_generate(&mut trainer, checkpoint_base);
}

fn find_tokenizer_path() -> Option<String> {
    let search_paths = [
        "aideen-backbone/tokenizer.json",
        "tokenizer.json",
        "../aideen-backbone/tokenizer.json",
    ];
    for p in &search_paths {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
}

fn save_and_generate(trainer: &mut Trainer, base: &str) {
    println!();
    println!("── Checkpoint ──────────────────────────────────");
    match trainer.save_checkpoint(base) {
        Ok(_) => {
            let aidn_size = fs::metadata(format!("{base}.aidn"))
                .map(|m| m.len())
                .unwrap_or(0);
            println!(
                "  {base}.aidn ({:.2} MB) + {base}.opt",
                aidn_size as f64 / 1_048_576.0
            );
        }
        Err(e) => println!("  Error guardando checkpoint: {e}"),
    }

    println!();
    println!("── Generación ──────────────────────────────────");
    let prompts = [
        "la inteligencia artificial",
        "cada neurona",
        "aideen es una red",
        "el equilibrio profundo",
        "la red neuronal distribuida",
    ];
    for prompt in &prompts {
        let generated = trainer.generate(prompt, 40, 0.8, 0.9, 40, 1.1);
        println!("  \"{prompt}\" →");
        println!("    \"{generated}\"");
    }

    println!();
    println!("✅ AIDEEN training completo.");
}
