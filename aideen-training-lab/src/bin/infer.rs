//! AIDEEN inference CLI.
//!
//! Usage:
//!   cargo run -p aideen-training --bin infer --release -- \
//!       --checkpoint checkpoints/latest/model.aidn \
//!       --tokenizer aideen-backbone/tokenizer.json \
//!       --max-tokens 64 --temperature 0.8 --d-r 256 \
//!       "Your prompt here"

use aideen_backbone::lm_head::LmHead;
use aideen_backbone::mamba_slot_reasoning::MambaSlotReasoning;
use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::model::AidenModel;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::ArchitectureConfig;
use std::io::Write;

// ---------------------------------------------------------------------------
// Argument parsing (no external crate needed)
// ---------------------------------------------------------------------------

struct Args {
    checkpoint: String,
    tokenizer_path: String,
    max_tokens: usize,
    temperature: f32,
    d_r: usize,
    prompt: String,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();

    let mut checkpoint = "checkpoints/latest/model.aidn".to_string();
    let mut tokenizer_path = "aideen-backbone/tokenizer.json".to_string();
    let mut max_tokens: usize = 64;
    let mut temperature: f32 = 0.8;
    let mut d_r: usize = 256;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--checkpoint" => {
                i += 1;
                checkpoint = raw
                    .get(i)
                    .ok_or("--checkpoint requires a value")?
                    .clone();
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = raw
                    .get(i)
                    .ok_or("--tokenizer requires a value")?
                    .clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = raw
                    .get(i)
                    .ok_or("--max-tokens requires a value")?
                    .parse()
                    .map_err(|e| format!("--max-tokens: {e}"))?;
            }
            "--temperature" => {
                i += 1;
                temperature = raw
                    .get(i)
                    .ok_or("--temperature requires a value")?
                    .parse()
                    .map_err(|e| format!("--temperature: {e}"))?;
            }
            "--d-r" => {
                i += 1;
                d_r = raw
                    .get(i)
                    .ok_or("--d-r requires a value")?
                    .parse()
                    .map_err(|e| format!("--d-r: {e}"))?;
            }
            other if other.starts_with('-') => {
                return Err(format!("Unknown flag: {other}"));
            }
            _ => {
                positional.push(raw[i].clone());
            }
        }
        i += 1;
    }

    let prompt = if positional.is_empty() {
        return Err("No prompt provided. Pass the prompt as the last argument.".to_string());
    } else {
        positional.join(" ")
    };

    Ok(Args {
        checkpoint,
        tokenizer_path,
        max_tokens,
        temperature,
        d_r,
        prompt,
    })
}

// ---------------------------------------------------------------------------
// Load model weights from an AidenModel checkpoint.
// Returns (reasoning_block, lm_head, embedding_weights).
// ---------------------------------------------------------------------------

fn load_model(
    path: &str,
    config: &mut ArchitectureConfig,
) -> Result<(MambaSlotReasoning, LmHead, Option<Vec<f32>>), String> {
    let model = AidenModel::load(path)?;

    // Adopt the checkpoint's architecture config but keep the caller's d_r if
    // the checkpoint does not override it.
    *config = model.config.clone();

    let mut reasoning = MambaSlotReasoning::new(config.clone());
    reasoning.import_weights(&model.weights)?;

    let mut head = LmHead::new(config.clone());
    head.import_weights(&model.weights)?;

    // Embedding matrix is stored row-major as "embed.w" when available.
    let embed_data = model.weights.get("embed.w").cloned();

    Ok((reasoning, head, embed_data))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!(
                "Usage: infer [--checkpoint PATH] [--tokenizer PATH] \
                 [--max-tokens N] [--temperature F] [--d-r N] \"prompt\""
            );
            std::process::exit(1);
        }
    };

    // ── 1. Build the architecture config ──────────────────────────────────
    let mut config = ArchitectureConfig::default();
    config.d_r = args.d_r;

    // ── 2. Load the tokenizer ─────────────────────────────────────────────
    let tokenizer_path = std::path::Path::new(&args.tokenizer_path);
    let mut tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_file(&args.tokenizer_path, config.clone())
            .unwrap_or_else(|e| {
                eprintln!("Warning: failed to load tokenizer ({e}), falling back to char-level");
                Tokenizer::from_text(&args.prompt, config.clone())
            })
    } else {
        eprintln!(
            "Warning: tokenizer file not found at {}, using char-level fallback",
            args.tokenizer_path
        );
        Tokenizer::from_text(&args.prompt, config.clone())
    };

    // ── 3. Load the model weights ─────────────────────────────────────────
    let checkpoint_path = std::path::Path::new(&args.checkpoint);
    let (reasoning, head, embed_data) = if checkpoint_path.exists() {
        match load_model(&args.checkpoint, &mut config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: failed to load checkpoint ({e}), using random init");
                (
                    MambaSlotReasoning::new(config.clone()),
                    LmHead::new(config.clone()),
                    None,
                )
            }
        }
    } else {
        eprintln!(
            "Warning: checkpoint not found at {}, using random init",
            args.checkpoint
        );
        (
            MambaSlotReasoning::new(config.clone()),
            LmHead::new(config.clone()),
            None,
        )
    };

    // Patch tokenizer embeddings from the checkpoint when available.
    if let Some(ref emb) = embed_data {
        let vocab = tokenizer.vocab_size();
        let d_r = config.d_r;
        if emb.len() == vocab * d_r {
            tokenizer.embeddings = nalgebra::DMatrix::from_row_slice(vocab, d_r, emb);
        }
    }

    // Update config.vocab_size to match the tokenizer.
    config.vocab_size = tokenizer.vocab_size();

    // ── 4. Encode the prompt ──────────────────────────────────────────────
    let prompt_tokens = tokenizer.encode(&args.prompt);
    if prompt_tokens.is_empty() {
        eprintln!("Error: prompt encoded to zero tokens");
        std::process::exit(1);
    }

    eprintln!(
        "[infer] d_r={} vocab={} prompt_tokens={} max_tokens={} temperature={}",
        config.d_r,
        config.vocab_size,
        prompt_tokens.len(),
        args.max_tokens,
        args.temperature,
    );

    // EOS token ID — use token 0 as a convention (many tokenizers use 0 or a
    // special ID). This will be refined once training produces real models.
    let eos_id: u32 = 0;

    // ── 5. Autoregressive generation loop ─────────────────────────────────
    // We keep a running context of token IDs for repetition penalty.
    let mut context: Vec<u32> = prompt_tokens.clone();

    // Print the prompt first.
    print!("{}", args.prompt);
    let _ = std::io::stdout().flush();

    // Temporal memory (M) — starts at zero.
    let mut m_state = aideen_core::state::HSlots::zeros(&config);

    // Start generation from the last prompt token.
    let mut last_token = *prompt_tokens.last().unwrap();

    for _step in 0..args.max_tokens {
        // (a) Embed the last token.
        let s = tokenizer.embed(last_token);

        // (b) Initialise H from the embedding.
        let h0 = reasoning.init(&s);

        // (c) Run Picard iterations (DEQ fixed-point solve).
        let mut h = h0;
        for _picard in 0..config.max_deq_iters {
            h = reasoning.step(&h, &s, None);
        }

        // (d) Temporal update: M_t = g(M_{t-1}, H*).
        m_state = reasoning.temporal_step(&m_state, &h);

        // (e) Project H* to logits via LmHead.
        let logits = head.forward(&h);

        // (f) Sample the next token.
        let next_token = LmHead::sample(
            &logits,
            args.temperature,
            0.95,  // top_p
            40,    // top_k
            1.1,   // repetition_penalty
            &context,
        );

        // (g) Stop on EOS.
        if next_token == eos_id {
            break;
        }

        // (h) Decode and print immediately (streaming).
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        let _ = std::io::stdout().flush();

        // (i) Advance.
        context.push(next_token);
        last_token = next_token;
    }

    println!();
}
