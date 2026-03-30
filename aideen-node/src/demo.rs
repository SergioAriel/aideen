//! # Aideen AI MVP — End-to-End Demo
//!
//! Demonstrates three inference loop operation modes:
//!
//! ## Mode 1: Local-Only
//! Local DEQ converges without querying external experts.
//!
//! ## Mode 2: Expert-Hop (in-process)
//! The node queries an expert in the same process via
//! `InProcessChannel`. The Critic updates reputation after each response.
//!
//! ## Mode 3: Full Pipeline (DEQ → H* → MambaDecoder → text)
//! Runs the full pipeline:
//!   query → encoder → DEQ (MambaSlotReasoning) → H* → MambaDecoder → tokens
//!
//! ## Running
//! ```
//! cargo run --bin demo -- [local|expert|full|both]
//! ```

use std::time::Instant;

use aideen_backbone::{MambaDecoder, MambaSlotReasoning};
use aideen_core::{
    compute::{ComputeBackend, TensorId},
    reasoning::Reasoning,
    state::{ArchitectureConfig, HSlots},
};
use aideen_node::{
    critic::{Critic, CriticConfig},
    expert::{ExpertClient, ExpertPipeline, ExpertService, UniformRouter},
    inference::{run as inference_run, InferenceConfig},
    network::{in_process::InProcessChannel, NetChannel},
    peers::NodeId,
};
use nalgebra::DVector;

// ── Compact mocks ────────────────────────────────────────────────────────────

struct DemoBackend;
impl ComputeBackend for DemoBackend {
    fn load_tensor(&mut self, _: &[f32]) -> Result<TensorId, String> {
        Ok(TensorId(0))
    }
    fn ffn_forward(
        &mut self,
        _: &TensorId,
        _: &TensorId,
        d: usize,
    ) -> Result<DVector<f32>, String> {
        Ok(DVector::zeros(d))
    }
}

// Stable reasoning: very mild decay → converges fast (for local/expert demo)
struct StableReasoning {
    config: ArchitectureConfig,
}
impl Reasoning for StableReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.config
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        let d_r = self.config.d_r;
        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };
        HSlots::from_broadcast(&s_r, &self.config)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn ComputeBackend>,
    ) -> HSlots {
        let h_slots = self.config.h_slots;
        let mut next = HSlots::zeros(&self.config);
        for k in 0..h_slots {
            next.set_slot(k, &(h.slot(k) * 0.9999));
        }
        next
    }
}

// ── Minimal vocabulary for demo ───────────────────────────────────────────────

/// Trivial tokeniser: each ASCII character → token ID.
fn tokenize(text: &str, vocab_size: usize) -> Vec<u32> {
    text.bytes().map(|b| b as u32 % vocab_size as u32).collect()
}

/// Trivial detokeniser: token ID → character.
fn detokenize(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(|&t| {
            let b = (t % 128) as u8;
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '·'
            }
        })
        .collect()
}

// ── MODO 1: Local-only ────────────────────────────────────────────────────────

fn run_local_mode(queries: &[&str]) {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║           MODO 1: Local-Only Inference              ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let config = ArchitectureConfig::default();
    let mut reasoning = StableReasoning {
        config: config.clone(),
    };
    let mut backend = DemoBackend;
    let cfg = InferenceConfig {
        max_iters: 20,
        epsilon: 1e-4,
        ..Default::default()
    };

    for (i, query) in queries.iter().enumerate() {
        let t0 = Instant::now();
        // Note: run_local_mode uses StableReasoning which doesn't need tokens,
        // but we pass the query as a string for compatibility with the inference_run signature.
        let result = inference_run(query, &mut reasoning, &mut backend, None, &cfg);
        let elapsed_ms = t0.elapsed().as_millis();

        println!("Query #{}: {:?}", i + 1, query);
        match result {
            Some(r) => {
                println!("  iters:         {}", r.metrics.iters);
                println!("  converged:     {}", r.metrics.converged);
                println!("  stability:     {:.4}", r.metrics.stability);
                println!("  slot_energy:   {:.4}", r.metrics.slot_energy);
                println!("  slot_diversity:{:.4}", r.metrics.slot_diversity);
                println!("  routing:       {:?}", r.routing);
                println!("  Q_semantic:    {:.4}", r.signal.q_semantic);
            }
            None => {
                println!("  ⚠ DEQ did not converge with Q_min (normal in MVP without training)")
            }
        }
        println!("  elapsed:       {}ms\n", elapsed_ms);
    }
}

// ── MODO 2: Expert-Hop ────────────────────────────────────────────────────────

fn spawn_expert_server() -> Box<dyn NetChannel> {
    let (client_ch, mut server_ch) = InProcessChannel::pair();
    std::thread::spawn(move || {
        let svc = ExpertService {
            reasoning: StableReasoning {
                config: ArchitectureConfig::default(),
            },
            k_max: 5,
            eps_step: 1e-5,
            delta_cap: 10.0,
        };
        loop {
            let Ok(task) = server_ch.recv() else { break };
            if let Ok(result) = svc.process(&task) {
                if server_ch.send(result).is_err() {
                    break;
                }
            }
        }
    });
    Box::new(client_ch)
}

fn run_expert_mode(queries: &[&str]) {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║         MODO 2: Expert-Hop con Critic v0            ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let peer_id: NodeId = [0x42u8; 32];
    let mut pipeline = ExpertPipeline {
        router: Box::new(UniformRouter { k: 1 }),
        client: ExpertClient::new(vec![(peer_id, spawn_expert_server())]),
        bundle_version: 1,
        delta_cap_global: Some(5.0),
        outlier_factor: None,
    };

    let mut critic = Critic::new(CriticConfig {
        temperature: 0.3,
        ucb_c: 0.5,
        q_init: 0.6,
        lr: 0.2,
    });

    let config = ArchitectureConfig::default();
    let mut reasoning = StableReasoning {
        config: config.clone(),
    };
    let mut backend = DemoBackend;
    let cfg = InferenceConfig {
        max_iters: 15,
        epsilon: 1e-4,
        ..Default::default()
    };

    for (i, query) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let result = inference_run(query, &mut reasoning, &mut backend, None, &cfg);
        let local_ms = t0.elapsed().as_millis();

        println!("Query #{}: {:?}", i + 1, query);

        let h_star = match result {
            Some(r) => {
                println!(
                    "  [Local] iters={}, stability={:.4}, routing={:?}",
                    r.metrics.iters, r.metrics.stability, r.routing
                );
                r.h_star
            }
            None => {
                println!("  [Local] DEQ did not converge — using zero H* for expert-hop");
                HSlots::zeros(&config)
            }
        };

        let h_flat = h_star.to_flat();
        let t1 = Instant::now();
        match pipeline.run(&h_flat[..config.d_r]) {
            Ok(run_result) => {
                let expert_ms = t1.elapsed().as_millis();
                let q_expert = run_result.q_mean;
                critic.update(&peer_id, q_expert);
                println!(
                    "  [Expert] delta_norm={:.4}, q_mean={:.4}, drops={}, elapsed={}ms",
                    run_result.delta_norm, q_expert, run_result.drops_count, expert_ms
                );
                println!(
                    "  [Critic] rep[{:02x}..] = {:.4}",
                    peer_id[0],
                    critic.reputation(&peer_id).unwrap_or(0.0)
                );
            }
            Err(e) => println!("  [Expert] Error: {e}"),
        }
        println!(
            "  Total: {}ms  (local: {}ms)\n",
            t0.elapsed().as_millis(),
            local_ms
        );
    }

    println!("Top-1 expert by reputation: {:?}", critic.top_k(1));
}

// ── MODO 3: Full Pipeline (DEQ → H* → MambaDecoder → texto) ──────────────────

fn run_full_mode(queries: &[&str]) {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║   MODO 3: Full Pipeline  DEQ → H* → MambaDecoder   ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("Note: random weights — tokens have no semantic meaning.");
    println!("      The goal is to validate that the PIPELINE WORKS.\n");

    // MambaSlotReasoning as f of the DEQ (cross-slot attn + Mamba SSM)
    let config = ArchitectureConfig::default();
    let mut reasoning = MambaSlotReasoning::new(config.clone());
    let mut backend = DemoBackend;

    // Small vocabulary for demo (ASCII printable)
    let vocab_size = 128usize;
    let mut dec_config = config.clone();
    dec_config.vocab_size = vocab_size;
    let decoder = MambaDecoder::new(4, dec_config);

    let cfg = InferenceConfig {
        max_iters: 30,
        epsilon: 1e-3, // more tolerant for convergence with random weights
        ..Default::default()
    };

    for (i, query) in queries.iter().enumerate() {
        let t0 = Instant::now();

        // ① DEQ loop con MambaSlotReasoning → H*
        let result = inference_run(query, &mut reasoning, &mut backend, None, &cfg);
        let deq_ms = t0.elapsed().as_millis();

        println!("Query #{}: {:?}", i + 1, query);

        let h_star = match &result {
            Some(r) => {
                println!(
                    "  [DEQ]     iters={}, converged={}, stability={:.4}",
                    r.metrics.iters, r.metrics.converged, r.metrics.stability
                );
                r.h_star.clone()
            }
            None => {
                println!("  [DEQ]     did not converge — using partial H*");
                HSlots::zeros(&config)
            }
        };

        // ② MambaDecoder: H* → tokens
        let prompt_tokens = tokenize(query, vocab_size);
        let t1 = Instant::now();
        let generated = decoder.generate(&h_star, &prompt_tokens[..prompt_tokens.len().min(8)], 20);
        let dec_ms = t1.elapsed().as_millis();

        let text = detokenize(&generated);
        println!("  [Decoder] tokens={:?}", generated);
        println!("  [Decoder] text:   {:?}", text);
        println!(
            "  [Timing]  DEQ={}ms  Decoder={}ms  Total={}ms\n",
            deq_ms,
            dec_ms,
            t0.elapsed().as_millis()
        );
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let queries = [
        "What is a distributed reasoning system?",
        "Explain how reinforcement learning works",
        "What is the difference between inference and training?",
    ];

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("both");

    println!("═══════════════════════════════════════════════════════");
    println!("           AIDEEN AI MVP — Demo End-to-End              ");
    println!("═══════════════════════════════════════════════════════");
    println!("Mode: {mode}  |  Queries: {}", queries.len());

    match mode {
        "local" => run_local_mode(&queries),
        "expert" => run_expert_mode(&queries),
        "full" => run_full_mode(&queries),
        _ => {
            run_local_mode(&queries);
            run_expert_mode(&queries);
            run_full_mode(&queries);
        }
    }

    println!("\n✓ Demo completed.");
}
