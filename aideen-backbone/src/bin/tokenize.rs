//! Streaming corpus tokenizer for AIDEEN.
//!
//! Memory-maps the corpus and tokenizes in batched parallel chunks,
//! flushing each batch to disk before starting the next.
//! Handles 10GB+ corpora in ~4 GB peak RAM.
//!
//! Usage:
//!   cargo run -p aideen-backbone --release --bin tokenize [-- OPTIONS]
//!
//! Options:
//!   --corpus PATH      Path to raw_corpus.txt  (default: data/corpus/raw_corpus.txt)
//!   --tokenizer PATH   Path to tokenizer.json  (default: aideen-backbone/tokenizer.json)
//!   --out-dir PATH     Output directory         (default: data/corpus)
//!   --split FLOAT      Train fraction 0..1      (default: 0.9)
//!   --chunk-mb N       Chunk size in MB         (default: 64)
//!   --batch N          Chunks per parallel batch (default: 4)

use memmap2::Mmap;
use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process;
use std::time::Instant;
use tokenizers::Tokenizer;

// ── CLI ──────────────────────────────────────────────────────────────

struct Args {
    corpus: PathBuf,
    tokenizer: PathBuf,
    out_dir: PathBuf,
    train_frac: f64,
    chunk_mb: usize,
    batch: usize,
}

impl Args {
    fn parse() -> Self {
        let argv: Vec<String> = env::args().collect();
        let mut a = Args {
            corpus: PathBuf::from("data/corpus/raw_corpus.txt"),
            tokenizer: PathBuf::from("aideen-backbone/tokenizer.json"),
            out_dir: PathBuf::from("data/corpus"),
            train_frac: 0.9,
            chunk_mb: 64,
            batch: 4,
        };
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--corpus" => { i += 1; a.corpus = PathBuf::from(&argv[i]); }
                "--tokenizer" => { i += 1; a.tokenizer = PathBuf::from(&argv[i]); }
                "--out-dir" => { i += 1; a.out_dir = PathBuf::from(&argv[i]); }
                "--split" => { i += 1; a.train_frac = argv[i].parse().unwrap(); }
                "--chunk-mb" => { i += 1; a.chunk_mb = argv[i].parse().unwrap(); }
                "--batch" => { i += 1; a.batch = argv[i].parse().unwrap(); }
                other => {
                    eprintln!("Unknown flag: {other}");
                    process::exit(1);
                }
            }
            i += 1;
        }
        a
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn fmt_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1_048_576 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1_073_741_824 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    }
}

/// Split the byte slice into chunks at newline boundaries.
fn chunk_at_newlines(data: &[u8], max_bytes: usize) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < data.len() {
        let end = (start + max_bytes).min(data.len());
        let boundary = if end == data.len() {
            end
        } else {
            match data[start..end].iter().rposition(|&b| b == b'\n') {
                Some(pos) => start + pos + 1,
                None => end,
            }
        };
        chunks.push(&data[start..boundary]);
        start = boundary;
    }
    chunks
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    println!("================================================================");
    println!("  AIDEEN Corpus Tokenizer (Rust — batched parallel)");
    println!("================================================================");
    println!();

    // 1. Load tokenizer
    if !args.tokenizer.exists() {
        eprintln!("ERROR: tokenizer not found at {}", args.tokenizer.display());
        process::exit(1);
    }
    println!("Loading tokenizer from {} ...", args.tokenizer.display());
    let tokenizer = Tokenizer::from_file(&args.tokenizer).unwrap_or_else(|e| {
        eprintln!("Failed to load tokenizer: {e}");
        process::exit(1);
    });
    println!("  Vocab size: {}", tokenizer.get_vocab_size(true));
    println!();

    // 2. Memory-map corpus
    if !args.corpus.exists() {
        eprintln!("ERROR: corpus not found at {}", args.corpus.display());
        eprintln!("Run  python scripts/fetch_corpus.py  first.");
        process::exit(1);
    }
    let file = File::open(&args.corpus).unwrap();
    let corpus_len = file.metadata().unwrap().len();
    println!("Memory-mapping {} ({}) ...", args.corpus.display(), fmt_size(corpus_len));

    // SAFETY: file is opened read-only and we don't modify the backing file.
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    let text = match std::str::from_utf8(&mmap) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: corpus is not valid UTF-8 at byte offset {}", e.valid_up_to());
            eprintln!("Hint: clean the corpus or use the Python fallback with errors='replace'.");
            process::exit(1);
        }
    };
    println!("  Corpus: {} chars, {}", text.len(), fmt_size(corpus_len));
    println!();

    // 3. Chunk the corpus
    let chunk_bytes = args.chunk_mb * 1_048_576;
    let chunks = chunk_at_newlines(text.as_bytes(), chunk_bytes);
    let n_chunks = chunks.len();
    let n_batches = (n_chunks + args.batch - 1) / args.batch;
    println!("Tokenizing {} chunks (~{} MB) in {} batches of {} with {} threads ...",
        n_chunks, args.chunk_mb, n_batches, args.batch, rayon::current_num_threads());
    println!();

    let t0 = Instant::now();

    // Phase 1: tokenize in batches, writing all tokens to a single temp file.
    // Each batch: par-tokenize N chunks → write to disk → drop results.
    fs::create_dir_all(&args.out_dir).unwrap();
    let tmp_path = args.out_dir.join("all.tokens.tmp");
    let mut tmp_writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&tmp_path).unwrap());
    let mut total_tokens: u64 = 0;

    for (batch_idx, batch_chunks) in chunks.chunks(args.batch).enumerate() {
        // Tokenize this batch in parallel
        let batch_results: Vec<Vec<u32>> = batch_chunks
            .par_iter()
            .map(|chunk_bytes| {
                let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
                let encoding = tokenizer.encode(chunk_str, false).unwrap_or_else(|e| {
                    eprintln!("Tokenization error: {e}");
                    process::exit(1);
                });
                encoding.get_ids().to_vec()
            })
            .collect();

        // Write this batch immediately, then drop
        let mut batch_tokens: u64 = 0;
        for ids in &batch_results {
            // Write all IDs as a contiguous block of LE u32
            let byte_slice = unsafe {
                std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4)
            };
            // Handle endianness: on little-endian (x86) this is a no-op memcpy
            #[cfg(target_endian = "little")]
            {
                tmp_writer.write_all(byte_slice).unwrap();
            }
            #[cfg(target_endian = "big")]
            {
                for &id in ids {
                    tmp_writer.write_all(&id.to_le_bytes()).unwrap();
                }
            }
            batch_tokens += ids.len() as u64;
        }
        total_tokens += batch_tokens;
        drop(batch_results); // free memory immediately

        let elapsed = t0.elapsed().as_secs_f64();
        let done_chunks = (batch_idx + 1) * args.batch;
        let progress = done_chunks.min(n_chunks) as f64 / n_chunks as f64;
        let eta = if progress > 0.0 { elapsed / progress - elapsed } else { 0.0 };
        println!("  batch {}/{}: +{} tokens (total: {}, {:.0}%, ETA {:.0}s)",
            batch_idx + 1, n_batches, batch_tokens, total_tokens,
            progress * 100.0, eta);
    }
    tmp_writer.flush().unwrap();
    drop(tmp_writer);

    let elapsed = t0.elapsed();
    let chars_per_tok = text.len() as f64 / total_tokens as f64;

    println!();
    println!("  Total tokens:  {total_tokens}");
    println!("  Chars / token: {chars_per_tok:.2}");
    println!("  Time:          {:.1}s ({:.1} MB/s)",
        elapsed.as_secs_f64(),
        corpus_len as f64 / 1_048_576.0 / elapsed.as_secs_f64());
    println!();

    // Phase 2: split the temp file into train / val by byte-range copy
    let train_path = args.out_dir.join("train.tokens.bin");
    let val_path = args.out_dir.join("val.tokens.bin");

    let split_token = (total_tokens as f64 * args.train_frac) as u64;
    let split_byte = split_token * 4;
    let total_byte = total_tokens * 4;

    println!("Splitting {:.0}% / {:.0}% ...",
        args.train_frac * 100.0, (1.0 - args.train_frac) * 100.0);
    println!("  Train: {} tokens ({})", split_token, fmt_size(split_byte));
    println!("  Val:   {} tokens ({})", total_tokens - split_token, fmt_size(total_byte - split_byte));
    println!();

    // Stream-copy from tmp → train + val
    let copy_buf_size = 64 * 1024 * 1024; // 64 MB
    let mut src = File::open(&tmp_path).unwrap();

    {
        use std::io::Read;
        let mut train_w = BufWriter::new(File::create(&train_path).unwrap());
        let mut remaining = split_byte;
        let mut buf = vec![0u8; copy_buf_size];
        while remaining > 0 {
            let to_read = (remaining as usize).min(buf.len());
            let n = std::io::Read::read(&mut src, &mut buf[..to_read]).unwrap();
            if n == 0 { break; }
            train_w.write_all(&buf[..n]).unwrap();
            remaining -= n as u64;
        }
        train_w.flush().unwrap();

        let mut val_w = BufWriter::new(File::create(&val_path).unwrap());
        loop {
            let n = std::io::Read::read(&mut src, &mut buf).unwrap();
            if n == 0 { break; }
            val_w.write_all(&buf[..n]).unwrap();
        }
        val_w.flush().unwrap();
    }

    fs::remove_file(&tmp_path).unwrap();

    let train_size = fs::metadata(&train_path).unwrap().len();
    let val_size = fs::metadata(&val_path).unwrap().len();

    println!("Done.");
    println!("  train : {} tokens  ({})", split_token, fmt_size(train_size));
    println!("  val   : {} tokens  ({})", total_tokens - split_token, fmt_size(val_size));
    println!("  total : {} tokens", total_tokens);
    println!("  time  : {:.1}s", t0.elapsed().as_secs_f64());
}
