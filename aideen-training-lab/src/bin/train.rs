//! AIDEEN training binary.
//!
//! # Architecture notes for future LLMs reading this
//!
//! ## Quantization (1.58b / BitNet)
//! PHASE 1: Train in float32. PHASE 2: retrain from scratch with `ternary = true` (QAT/STE).
//! Do NOT do post-training quantization — at 1.58b it destroys quality. The STE is already
//! implemented in the GPU shaders (embedding_train.wgsl, lm_train.wgsl, fused_deq_update.wgsl).
//!
//! ## Uniform attention — critical risk to monitor
//! With random weights, `attn_ent = log(8) = 2.079` always (maximum entropy, identical slots).
//! For the DEQ to be powerful, the slots MUST specialize during training
//! (attn_ent must drop below 2.079). If it does not, the 8 slots collapse to the same thing and
//! the parallel reasoning capacity is completely wasted.
//! Monitor `attn_ent` in GPU-DEBUG. If it does not drop after ~1000 real steps, investigate.
//!
//! ## Usage modes:
//!   # Fast mode — existing dataset.txt (small, for tests)
//!   cargo run --release --features wgpu -p aideen-training --bin train
//!
//!   # Large file mode — any .txt (streaming, no RAM limit)
//!   cargo run --release --features wgpu -p aideen-training --bin train -- --file path/to/corpus.txt
//!
//!   # Checkpoint mode — continue from a previous checkpoint
//!   cargo run --release --features wgpu -p aideen-training --bin train -- --file corpus.txt --resume model

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;

use std::{env, fs};

const DOC_MARKER: &str = "<|endoftext|>";
const BPE_EOS_TOKEN: u32 = 50256;

fn env_u64(name: &str) -> Option<u64> {
    env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
}

fn setup_gpu(trainer: &mut Trainer) {
    #[cfg(feature = "wgpu")]
    {
        if trainer.gpu_deq.is_some() {
            println!("  Backend: GPU (Metal) ✅ [Auto-managed]");
        } else {
            println!("  Backend: CPU (GPU not available)");
        }
    }
    #[cfg(not(feature = "wgpu"))]
    println!("  Backend: CPU (compile with --features wgpu for GPU)");
}

fn encode_training_corpus(tok: &Tokenizer, corpus: &str) -> (Vec<u32>, u32) {
    if tok.hf_tokenizer.is_none() || !corpus.contains(DOC_MARKER) {
        return (tok.encode(corpus), 0);
    }

    let mut tokens = Vec::new();
    let mut docs_written = 0usize;
    for doc in corpus.split(DOC_MARKER) {
        let doc = doc.trim();
        if doc.is_empty() {
            continue;
        }
        let encoded = tok.encode(doc);
        if encoded.is_empty() {
            continue;
        }
        if !tokens.is_empty() {
            tokens.push(BPE_EOS_TOKEN);
        }
        tokens.extend_from_slice(&encoded);
        docs_written += 1;
    }

    if docs_written == 0 {
        return (tok.encode(corpus), 0);
    }

    println!(
        "  Tokenization with explicit boundaries: {} documents, eos_token={}",
        docs_written, BPE_EOS_TOKEN
    );
    (tokens, BPE_EOS_TOKEN)
}

fn tokenizer_cache_suffix(tok: &Tokenizer) -> &'static str {
    if tok.hf_tokenizer.is_some() {
        "bpe"
    } else {
        "char"
    }
}

fn main() {
    // ── Parse args ──────────────────────────────────────────────────────────
    let args: Vec<String> = env::args().collect();
    let mut large_file: Option<String> = None;
    let mut val_file: Option<String> = None;
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
            "--val-file" | "--validation" => {
                i += 1;
                val_file = args.get(i).cloned();
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
        eprintln!("ERROR: missing --file <corpus.txt>. The fallback mode was disabled.");
        std::process::exit(2);
    };
    run_large_file(
        txt_path,
        val_file,
        resume_path,
        epochs,
        log_every,
        save_every,
        freeze_deq,
        freeze_emb,
        freeze_lm,
    );
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|v| {
            let vl = v.trim().to_ascii_lowercase();
            vl == "1" || vl == "true" || vl == "yes"
        })
        .unwrap_or(false)
}

/// Trains on a large .txt using streaming (no RAM limit).
/// Tokenizes the text, writes it to a temporary .bin and calls train_on_file.
fn run_large_file(
    txt_path: &str,
    val_file: Option<String>,
    resume_path: Option<String>,
    epochs: usize,
    log_every: usize,
    save_every: usize,
    freeze_deq: bool,
    freeze_emb: bool,
    freeze_lm: bool,
) {
    println!("  Mode: large file → {txt_path}");

    // ── Resolve real dataset path ──────────────────────────────────────────
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
            found.expect("Cannot find the dataset in any known path.")
        }
    };
    let txt_path = resolved_path;

    // ── Build tokenizer ────────────────────────────────────────────────────
    let config_default = ArchitectureConfig::default();
    let force_char_tokenizer = env_flag("AIDEEN_FORCE_CHAR_TOKENIZER");
    let tok_path = if force_char_tokenizer {
        None
    } else {
        find_tokenizer_path()
    };
    let mut tok = if let Some(ref path) = tok_path {
        println!("  Tokenizer: BPE ({path}) ✅");
        Tokenizer::from_file(path, config_default.clone()).expect("Failed to load tokenizer.json")
    } else {
        let corpus = corpus
            .as_ref()
            .expect("Char-level tokenizer requires corpus in memory.");
        if force_char_tokenizer {
            println!("  Tokenizer: Char-level (forced) — scanning vocab...");
        } else {
            println!("  Tokenizer: Char-level — scanning vocab...");
        }
        Tokenizer::from_text(corpus, config_default.clone())
    };

    tok.config.vocab_size = tok.vocab_size();
    tok.config.ctx_len = std::env::var("AIDEEN_CTX_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        // Training throughput is strongly launch-bound at 256. 512 stays stable on M1 Pro
        // and gives the GPU more useful work per dispatch; callers can still override it.
        .unwrap_or(512);
    tok.config.num_samples = std::env::var("AIDEEN_NUM_SAMPLES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(tok.config.num_samples)
        .max(1);
    tok.config.train_deq = true;
    println!(
        "  LM samples: {}{}",
        tok.config.num_samples,
        if tok.config.num_samples >= tok.config.vocab_size {
            " (full vocab)"
        } else {
            ""
        }
    );

    let vocab_size = tok.vocab_size();
    let heldout_tokens = val_file.as_ref().map(|path| {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Cannot read --val-file {path}: {e}"));
        let (mut tokens, _) = encode_training_corpus(&tok, &text);
        if let Some(max_tokens) = env::var("AIDEEN_HELDOUT_MAX_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v >= 2)
        {
            tokens.truncate(max_tokens);
        }
        println!("  Held-out: {} tokens from {}", tokens.len(), path);
        tokens
    });

    // ── Tokenize and write .bin (or reuse cache) ───────────────────────────
    let cache_suffix = tokenizer_cache_suffix(&tok);
    let bin_path = format!("{txt_path}.{cache_suffix}.tokens.bin");
    let txt_meta = fs::metadata(&txt_path).ok();
    let bin_meta = fs::metadata(&bin_path).ok();
    let use_cache = bin_meta
        .as_ref()
        .and_then(|b| b.modified().ok())
        .zip(txt_meta.as_ref().and_then(|t| t.modified().ok()))
        .map(|(b, t)| b >= t)
        .unwrap_or(false);

    if use_cache {
        let bin_bytes = bin_meta.unwrap().len() as usize;
        let tokens_len = bin_bytes / 4;
        println!(
            "  Cache OK: reusing {bin_path} ({} tokens, vocab={})",
            tokens_len, vocab_size
        );
    } else {
        let corpus = corpus
            .as_ref()
            .expect("Corpus in memory required for tokenization.");
        println!("  Tokenizing {txt_path} → {bin_path} ...");
        {
            use std::io::Write;
            let (tokens, doc_eos_token) = encode_training_corpus(&tok, corpus);
            let byte_data: &[u8] = bytemuck::cast_slice(&tokens);

            // Ensure the parent directory exists
            if let Some(parent) = std::path::Path::new(&bin_path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }

            let mut f = fs::File::create(&bin_path).expect("Cannot create .bin");
            f.write_all(byte_data).expect("Error writing .bin");
            println!(
                "  {} chars → {} tokens, vocab={} → {:.2} MB",
                corpus.len(),
                tokens.len(),
                vocab_size,
                byte_data.len() as f64 / 1_048_576.0
            );
            if doc_eos_token != 0 {
                println!("  Document boundaries active in .bin (eos_token={doc_eos_token})");
            }
        }
    }

    let lr = std::env::var("AIDEEN_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0001);
    let checkpoint_base = env::var("AIDEEN_CHECKPOINT_BASE")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "model_large".to_string());

    let train_seed = env_u64("AIDEEN_TRAIN_SEED");
    let mut trainer = if let Some(ref base) = resume_path {
        println!("  Resuming from checkpoint: {base}");
        let mut t = Trainer::load_checkpoint(base).expect("Error loading checkpoint");
        // Resume should still allow an explicit LR override for continuation probes.
        // Without this, AIDEEN_LR only affects fresh runs and resume experiments silently
        // keep the checkpoint's stored schedule.
        t.training_config.lr = lr;
        t.training_config.lr_min = lr / 10.0;
        t.training_config.warmup_epochs = 0;
        t.training_config.epochs = epochs;
        t.plateau_lr_cap = t.plateau_lr_cap.min(lr);
        t
    } else {
        if let Some(seed) = train_seed {
            println!("  Init seed: {seed}");
        }
        let mut t = if let Some(seed) = train_seed {
            Trainer::from_tokenizer_seeded(tok, lr, seed)
        } else {
            Trainer::from_tokenizer(tok, lr)
        };
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
        println!("  Bit-Diet (Ternary 1.58-bit) active 🔥  [STE during training]");
    }
    println!();

    // If the corpus contains explicit document markers, we convert them to real
    // EOS tokens during tokenization. In that case training must reset state there.
    let eos_token: u32 = if tok_path.is_some() {
        if let Ok(corpus_text) = fs::read_to_string(&txt_path) {
            if corpus_text.contains(DOC_MARKER) {
                BPE_EOS_TOKEN
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    };

    let t0 = std::time::Instant::now();
    trainer
        .train_on_file(
            &bin_path,
            epochs,
            log_every,
            eos_token,
            save_every,
            &checkpoint_base,
        )
        .expect("Error during train_on_file");

    if let Some(tokens) = heldout_tokens.as_ref() {
        let val_t0 = std::time::Instant::now();
        let val_loss = trainer.eval_tokens_gpu(tokens);
        println!(
            "  [HELDOUT] val_loss={:.4} tokens={} time={:.1}s",
            val_loss,
            tokens.len(),
            val_t0.elapsed().as_secs_f32()
        );
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f32());
    println!("  Spectral norms: {:?}", trainer.reasoning.spectral_norms());

    if save_every == 0 {
        println!("  [Skip] save/generate disabled (save_every=0)");
        return;
    }
    save_and_generate(&mut trainer, &checkpoint_base);
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
        Err(e) => println!("  Error saving checkpoint: {e}"),
    }

    println!();
    println!("── Generation ──────────────────────────────────");
    let prompts = [
        "artificial intelligence",
        "each neuron",
        "aideen is a network",
        "deep equilibrium",
        "the distributed neural network",
    ];
    for prompt in &prompts {
        let generated = trainer.generate(prompt, 40, 0.8, 0.9, 40, 1.1);
        println!("  \"{prompt}\" →");
        println!("    \"{generated}\"");
    }

    println!();
    println!("✅ AIDEEN training complete.");
}
