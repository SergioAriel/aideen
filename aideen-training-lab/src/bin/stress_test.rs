//! AIDEEN v13.2 Integrity Stress Test
//! Forces Global Memory Fallback (8192 elements) to verify the sync and scalability fixes.

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;
use std::env;
use std::time::Instant;

const PROMPT: &str = "AIDEEN STRESS TEST CORE INTEGRITY GLOBAL MEMORY SYNC SCALABILITY GRADIENT SOLVER CONVERGENCE CHECK.";

fn env_u32(name: &str) -> Option<u32> {
    env::var(name).ok().and_then(|v| v.parse::<u32>().ok())
}

fn env_f32(name: &str) -> Option<f32> {
    env::var(name).ok().and_then(|v| v.parse::<f32>().ok())
}

fn env_u64(name: &str) -> Option<u64> {
    env::var(name).ok().and_then(|v| v.parse::<u64>().ok())
}

fn main() {
    println!("\n[STRESS-TEST] Iniciando Auditoría de Integridad v13.2...");

    let force_global = env::var("AIDEEN_STRESS_FORCE_GLOBAL")
        .ok()
        .as_deref()
        == Some("1");
    let _deq_only = env::var("AIDEEN_DEQ_ONLY").ok().as_deref() == Some("1");
    let _no_mamba = env::var("AIDEEN_DEQ_NO_MAMBA").ok().as_deref() == Some("1");

    let mut config = ArchitectureConfig::default();
    config.d_r = env_u32("AIDEEN_STRESS_DR").unwrap_or(if force_global { 1024 } else { 512 }) as usize;
    config.h_slots = env_u32("AIDEEN_STRESS_HSLOTS").unwrap_or(8) as usize;
    config.max_deq_iters = env_u32("AIDEEN_STRESS_MAX_ITERS").unwrap_or(if force_global {
        12
    } else {
        14
    }) as usize;
    config.adj_iters = env_u32("AIDEEN_STRESS_ADJ_ITERS").unwrap_or(if force_global {
        8
    } else {
        6
    }) as usize;
    config.deq_epsilon = env_f32("AIDEEN_STRESS_EPS").unwrap_or(if force_global {
        1e-4
    } else {
        2e-4
    });
    config.deq_grad_scale = 0.01;
    let stress_iters = env_u32("AIDEEN_STRESS_ITERS").unwrap_or(20) as usize;
    let lr = env_f32("AIDEEN_STRESS_LR").unwrap_or(if force_global { 0.0007 } else { 0.0003 });
    let seed = env_u64("AIDEEN_STRESS_SEED").unwrap_or(42);

    println!(
        "[STRESS-TEST] Config: d_r={} h_slots={} total={} max_iters={} adj_iters={} eps={} lr={} ({})",
        config.d_r,
        config.h_slots,
        config.d_r * config.h_slots,
        config.max_deq_iters,
        config.adj_iters,
        config.deq_epsilon,
        lr,
        if force_global {
            "FORCING GLOBAL"
        } else {
            "STABLE PROFILE"
        }
    );

    let tok = Tokenizer::from_text(PROMPT, config.clone());
    let tokens = tok.encode(PROMPT);
    let mut train_tokens = &tokens[..tokens.len() - 1];
    let mut targets = &tokens[1..];
    if let Some(seq_len) = env_u32("AIDEEN_STRESS_SEQ_LEN") {
        let seq_len = seq_len as usize;
        if seq_len >= 2 && tokens.len() >= seq_len {
            train_tokens = &tokens[..seq_len - 1];
            targets = &tokens[1..seq_len];
        }
    }

    let mut trainer = Trainer::from_tokenizer_seeded(tok, lr, seed);
    trainer.config = config;
    trainer.config.train_deq = true;
    if env::var("AIDEEN_LM_FROZEN").ok().as_deref() == Some("1") {
        trainer.frozen_lm = true;
    }
    if env::var("AIDEEN_EMB_FROZEN").ok().as_deref() == Some("1") {
        trainer.frozen_emb = true;
    }
    if env::var("AIDEEN_DEQ_FROZEN").ok().as_deref() == Some("1") {
        trainer.frozen_deq = true;
    }
    if let Some(v) = env_u32("AIDEEN_STRESS_ADAPTIVE_MAX_ITERS") {
        trainer.adaptive_max_iters = v;
    }

    #[cfg(feature = "wgpu")]
    {
        use aideen_backbone::gpu_backend::WgpuBlockBackend;
        if let Some(gpu) = WgpuBlockBackend::new_blocking() {
            trainer.reasoning.set_backend(gpu);
            println!("[STRESS-TEST] Backend: GPU Metal ✅");
        }
    }

    println!("[STRESS-TEST] Ejecutando ráfagas de entrenamiento...");
    let seq_tokens = train_tokens.len();
    let mut total_ms = 0u128;
    for i in 0..stress_iters {
        let t = Instant::now();
        let loss = trainer.train_sequence(train_tokens, targets, true, trainer.config.deq_epsilon);
        let elapsed_ms = t.elapsed().as_millis();
        total_ms += elapsed_ms;
        let tps = (seq_tokens * (i + 1)) as f64 / (total_ms as f64 / 1000.0);
        println!(
            "[STRESS-TEST] Iter {:>2} | Loss: {:.4} | Time: {:>4}ms | TPS: {:>6.1}",
            i + 1,
            loss,
            elapsed_ms,
            tps,
        );
    }
    println!(
        "[STRESS-TEST] TPS promedio: {:.1} ({} tokens × {} iters en {:.2}s)",
        (seq_tokens * stress_iters) as f64 / (total_ms as f64 / 1000.0),
        seq_tokens,
        stress_iters,
        total_ms as f64 / 1000.0,
    );

    if env::var("AIDEEN_MAMBA_HIST_PROFILE").ok().is_some() {
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = trainer.gpu_deq.as_ref() {
            let seq_len = targets.len() as u32;
            let ((w_hist_mean, w_hist_max), (w_delta_mean, w_delta_max), (b_delta_mean, b_delta_max)) =
                gpu.read_hist_selective_param_stats();
            let ((fwd_delta_mean, fwd_delta_max, fwd_delta_nz), (fwd_a_mean, fwd_a_min, fwd_a_max)) =
                gpu.read_hist_selective_forward_stats(seq_len);
            eprintln!(
                "[GPU-HIST-SELECTIVE-CHECK] w_hist(mean/max)=({:.6e},{:.6e}) w_delta(mean/max)=({:.6e},{:.6e}) b_delta(mean/max)=({:.6e},{:.6e}) fwd_delta(mean/max/nz)={:.6e}/{:.6e}/{}/{} fwd_a(mean/min/max)=({:.6e},{:.6e},{:.6e})",
                w_hist_mean,
                w_hist_max,
                w_delta_mean,
                w_delta_max,
                b_delta_mean,
                b_delta_max,
                fwd_delta_mean,
                fwd_delta_max,
                fwd_delta_nz,
                seq_len as usize * trainer.config.h_slots * trainer.config.d_r,
                fwd_a_mean,
                fwd_a_min,
                fwd_a_max
            );
        }
    }

    println!("\n[STRESS-TEST] Audit Finalizado. Revisa los logs [GPU-ORACLE] arriba.");
    println!("  - Si 'mode' es NORMAL/BOOST y 'conv' es OK -> El Sync de Memoria Global funciona.");
    println!("  - Si 'rs_cg' es bajo -> La escalabilidad del CG Solver funciona.");
    println!("  - Para forzar d_r=1024: AIDEEN_STRESS_FORCE_GLOBAL=1");
    println!("  - Overrides: AIDEEN_STRESS_DR, AIDEEN_STRESS_HSLOTS, AIDEEN_STRESS_MAX_ITERS, AIDEEN_STRESS_ADJ_ITERS, AIDEEN_STRESS_EPS, AIDEEN_STRESS_LR, AIDEEN_STRESS_ITERS");
}
