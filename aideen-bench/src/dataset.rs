/// Dataset: TinyShakespeare with char-level tokenizer.
///
/// Downloads ~1MB from GitHub, builds 65-char vocab,
/// and returns train/val splits as Vec<u32>.

const URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const CACHE: &str = "aideen-bench/tinyshakespeare.txt";

pub struct Dataset {
    pub vocab: Vec<char>,
    pub train: Vec<u32>,
    pub val: Vec<u32>,
}

impl Dataset {
    pub fn load() -> Self {
        let text = load_text();
        build(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.vocab.iter().position(|&v| v == c).map(|i| i as u32))
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&t| self.vocab.get(t as usize).copied())
            .collect()
    }
}

fn load_text() -> String {
    // 1. Try to read from local cache
    if let Ok(text) = std::fs::read_to_string(CACHE) {
        if text.len() > 100_000 {
            println!("  Dataset: local cache ({} chars)", text.len());
            return text;
        }
    }

    // 2. Download with curl (available by default on macOS)
    println!("  Dataset: downloading TinyShakespeare...");
    let output = std::process::Command::new("curl")
        .args(["-sL", "--max-time", "30", URL])
        .output()
        .expect("curl not available");

    if !output.status.success() || output.stdout.len() < 100_000 {
        panic!("Error downloading TinyShakespeare. Check internet connection.");
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    println!("  Dataset: {} chars downloaded", text.len());

    // Save cache
    if let Some(parent) = std::path::Path::new(CACHE).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(CACHE, &text);

    text
}

fn build(text: String) -> Dataset {
    // Build sorted char-level vocab
    let mut vocab: Vec<char> = text
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    vocab.sort();

    let tokens: Vec<u32> = text
        .chars()
        .filter_map(|c| vocab.iter().position(|&v| v == c).map(|i| i as u32))
        .collect();

    // Split 90% train / 10% val
    let split = (tokens.len() as f32 * 0.9) as usize;
    let train = tokens[..split].to_vec();
    let val = tokens[split..].to_vec();

    println!(
        "  Vocab: {} chars | Train: {} tokens | Val: {} tokens",
        vocab.len(),
        train.len(),
        val.len()
    );

    Dataset { vocab, train, val }
}
