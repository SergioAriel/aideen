//! Binary de chat interactivo con AIDEEN.
//!
//! Carga un checkpoint y permite conversar en modo texto.
//!
//! Uso:
//!   cargo run --release --features wgpu -p aideen-training --bin chat
//!   cargo run --release --features wgpu -p aideen-training --bin chat -- --model model_large
//!   cargo run --release --features wgpu -p aideen-training --bin chat -- --model model_large --max-tokens 80

use aideen_training::trainer::Trainer;
use std::{env, io::{self, BufRead, Write}};

const DEFAULT_MODEL: &str = "model_large";
const DEFAULT_MAX_TOKENS: usize = 120;
const DEFAULT_TEMPERATURE: f32 = 0.8;
const DEFAULT_TOP_P: f32 = 0.9;
const DEFAULT_TOP_K: usize = 40;
const DEFAULT_REP_PENALTY: f32 = 1.1;
const CTX_WINDOW: usize = 512; // caracteres máximos del historial de conversación

fn main() {
    // ── Parse args ──────────────────────────────────────────────────────────
    let args: Vec<String> = env::args().collect();
    let mut model_base  = DEFAULT_MODEL.to_string();
    let mut max_tokens  = DEFAULT_MAX_TOKENS;
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut top_p       = DEFAULT_TOP_P;
    let mut top_k       = DEFAULT_TOP_K;
    let mut rep_penalty = DEFAULT_REP_PENALTY;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if let Some(v) = args.get(i) { model_base = v.clone(); }
            }
            "--max-tokens" | "-n" => {
                i += 1;
                if let Some(v) = args.get(i) { max_tokens = v.parse().unwrap_or(DEFAULT_MAX_TOKENS); }
            }
            "--temperature" | "-t" => {
                i += 1;
                if let Some(v) = args.get(i) { temperature = v.parse().unwrap_or(DEFAULT_TEMPERATURE); }
            }
            "--top-p" => {
                i += 1;
                if let Some(v) = args.get(i) { top_p = v.parse().unwrap_or(DEFAULT_TOP_P); }
            }
            "--top-k" => {
                i += 1;
                if let Some(v) = args.get(i) { top_k = v.parse().unwrap_or(DEFAULT_TOP_K); }
            }
            "--rep-penalty" => {
                i += 1;
                if let Some(v) = args.get(i) { rep_penalty = v.parse().unwrap_or(DEFAULT_REP_PENALTY); }
            }
            _ => {}
        }
        i += 1;
    }

    // ── Banner ───────────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AIDEEN — Chat interactivo                                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Modelo : {model_base}");
    println!("  Tokens : {max_tokens}  temp={temperature}  top_p={top_p}  top_k={top_k}  rep={rep_penalty}");
    println!();
    println!("  Escribe tu mensaje y pulsa Enter. Escribe 'exit' o Ctrl+D para salir.");
    println!("  Escribe '/reset' para limpiar el contexto de conversación.");
    println!("  Escribe '/info' para ver el contexto activo.");
    println!();

    // ── Cargar checkpoint ────────────────────────────────────────────────────
    print!("  Cargando checkpoint '{model_base}'... ");
    io::stdout().flush().ok();

    let mut trainer = match Trainer::load_checkpoint(&model_base) {
        Ok(t) => {
            println!("✅");
            t
        }
        Err(e) => {
            eprintln!("❌\n  Error: {e}");
            eprintln!("  Asegúrate de haber entrenado primero con:");
            eprintln!("    cargo run --release --features wgpu -p aideen-training --bin train -- --file <dataset>");
            std::process::exit(1);
        }
    };

    println!();

    // ── Bucle de conversación ────────────────────────────────────────────────
    // Mantenemos un contexto de texto acumulado (ventana deslizante).
    let mut context = String::new();
    let stdin = io::stdin();

    loop {
        print!("You: ");
        io::stdout().flush().ok();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,          // Ctrl+D / EOF
            Ok(_) => {}
            Err(e) => { eprintln!("Error leyendo stdin: {e}"); break; }
        }

        let input = line.trim();
        if input.is_empty() { continue; }

        // Comandos especiales
        match input {
            "exit" | "quit" | "q" => break,
            "/reset" => {
                context.clear();
                println!("  [contexto reiniciado]\n");
                continue;
            }
            "/info" => {
                println!("  [contexto actual ({} chars)]:", context.len());
                if context.is_empty() {
                    println!("  (vacío)");
                } else {
                    // Mostrar últimas líneas del contexto
                    let preview: String = context.chars().rev().take(200).collect::<String>()
                        .chars().rev().collect();
                    println!("  ...{preview}");
                }
                println!();
                continue;
            }
            _ => {}
        }

        // Construir el prompt: historial + turno actual
        context.push_str("Human: ");
        context.push_str(input);
        context.push_str("\nAIDEEN:");

        // Ventana deslizante: si el contexto es muy largo, recortamos por el principio
        let prompt = if context.len() > CTX_WINDOW {
            &context[context.len() - CTX_WINDOW..]
        } else {
            &context
        };

        print!("AIDEEN: ");
        io::stdout().flush().ok();

        let response = trainer.generate_stream(
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            rep_penalty,
            |chunk| {
                print!("{chunk}");
                io::stdout().flush().ok();
            },
        );

        println!();
        println!();

        // Añadir respuesta al contexto
        context.push(' ');
        context.push_str(response.trim());
        context.push('\n');
    }

    println!();
    println!("  Hasta luego.");
}
