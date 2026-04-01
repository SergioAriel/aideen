//! Inference runner no interactivo para checkpoints de AIDEEN.
//!
//! Uso:
//!   cargo run --release --features wgpu -p aideen-training --bin infer -- \
//!     --model model_large --prompt "La memoria temporal permite"

use aideen_training::trainer::Trainer;
use std::{
    env, fs,
    io::{self, Read},
    path::Path,
    time::Instant,
};

const DEFAULT_MODEL: &str = "model_large";
const DEFAULT_MAX_TOKENS: usize = 120;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.9;
const DEFAULT_TOP_K: usize = 40;
const DEFAULT_REP_PENALTY: f32 = 1.1;
const DEFAULT_PROMPT: &str = "La memoria temporal permite";

fn read_prompt_from_stdin() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf.trim().to_string())
}

fn resolve_model_path(model: &str) -> (String, bool) {
    if model.ends_with(".aidn") {
        return (model.to_string(), true);
    }
    (model.to_string(), false)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model = DEFAULT_MODEL.to_string();
    let mut prompt: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut read_stdin = false;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut top_p = DEFAULT_TOP_P;
    let mut top_k = DEFAULT_TOP_K;
    let mut rep_penalty = DEFAULT_REP_PENALTY;
    let mut stream = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    model = v.clone();
                }
            }
            "--prompt" | "-p" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    prompt = Some(v.clone());
                }
            }
            "--prompt-file" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    prompt_file = Some(v.clone());
                }
            }
            "--stdin" => read_stdin = true,
            "--max-tokens" | "-n" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    max_tokens = v.parse().unwrap_or(DEFAULT_MAX_TOKENS);
                }
            }
            "--temperature" | "-t" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    temperature = v.parse().unwrap_or(DEFAULT_TEMPERATURE);
                }
            }
            "--top-p" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    top_p = v.parse().unwrap_or(DEFAULT_TOP_P);
                }
            }
            "--top-k" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    top_k = v.parse().unwrap_or(DEFAULT_TOP_K);
                }
            }
            "--rep-penalty" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    rep_penalty = v.parse().unwrap_or(DEFAULT_REP_PENALTY);
                }
            }
            "--stream" => stream = true,
            _ => {}
        }
        i += 1;
    }

    let prompt = if let Some(path) = prompt_file {
        fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("No se pudo leer --prompt-file {}: {}", path, e))
            .trim()
            .to_string()
    } else if read_stdin {
        read_prompt_from_stdin().expect("No se pudo leer prompt desde stdin")
    } else {
        prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string())
    };

    let (model_path, is_full_path) = resolve_model_path(&model);

    println!("AIDEEN inference");
    println!("  model       : {}", model_path);
    println!("  prompt chars: {}", prompt.len());
    println!(
        "  sampling    : max_tokens={} temp={} top_p={} top_k={} rep={}",
        max_tokens, temperature, top_p, top_k, rep_penalty
    );

    let load_start = Instant::now();
    let mut trainer = if is_full_path {
        Trainer::load_full(&model_path).unwrap_or_else(|e| {
            panic!("No se pudo cargar modelo {}: {}", model_path, e)
        })
    } else if Path::new(&format!("{model_path}.aidn")).exists() {
        Trainer::load_checkpoint(&model_path).unwrap_or_else(|e| {
            panic!("No se pudo cargar checkpoint {}: {}", model_path, e)
        })
    } else {
        Trainer::load_full(&model_path).unwrap_or_else(|e| {
            panic!("No se pudo cargar modelo {}: {}", model_path, e)
        })
    };
    let load_elapsed = load_start.elapsed().as_secs_f32();

    let prompt_tokens = trainer.tokenizer.encode(&prompt);
    println!("  prompt tokens: {}", prompt_tokens.len());
    println!("  load time    : {:.2}s", load_elapsed);
    println!();
    println!("--- PROMPT ---");
    println!("{}", prompt);
    println!("--- OUTPUT ---");

    let gen_start = Instant::now();
    let output = if stream {
        trainer.generate_stream(
            &prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            rep_penalty,
            |chunk| print!("{}", chunk),
        )
    } else {
        let out = trainer.generate(
            &prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            rep_penalty,
        );
        println!("{}", out);
        out
    };
    let gen_elapsed = gen_start.elapsed().as_secs_f32();
    if stream {
        println!();
    }

    let output_tokens = trainer.tokenizer.encode(&output);
    println!("--- METRICS ---");
    println!("  output chars : {}", output.len());
    println!("  output tokens: {}", output_tokens.len());
    println!("  gen time     : {:.2}s", gen_elapsed);
    println!(
        "  tok/s        : {:.1}",
        output_tokens.len() as f32 / gen_elapsed.max(1e-9)
    );
}
