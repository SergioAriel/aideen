use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;

fn main() {
    let prompt = "el equilibrio....... \
el equilibrio....... \
el equilibrio....... \
la red.............. \
la red.............. \
la red.............. \
aideen inteligencia. \
aideen inteligencia. \
aideen inteligencia. \
";
    let config = ArchitectureConfig::default();
    let tok = Tokenizer::from_text(prompt, config);
    println!("Vocab size: {}", tok.vocab_size());
    let tokens = tok.encode(prompt);
    println!("Total tokens: {}", tokens.len());
    println!("Tokens: {:?}", tokens);
}
