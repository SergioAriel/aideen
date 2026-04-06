use aideen_block::deq_bridge::{DeqComputeShape, RustDeqBridge};

#[tokio::test]
async fn test_deq_forward_wgpu_dispatch() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Test DEQ Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .unwrap();

    let b = 1;
    let d = 512;
    let h = 8;
    let bridge = RustDeqBridge::new(
        &device,
        d as u32,
        h as u32,
        b as u32,
        1,
        adapter.features().contains(wgpu::Features::SUBGROUP),
    );

    let shape = DeqComputeShape {
        batch_size: b as u32,
        d_model: d as u32,
        h_slots: h as u32,
        max_iters: 10,
        epsilon: 1e-4,
        damping: 0.9,
        seq_len: 1,
        residual_alpha: 0.0,
        debug_enable: 1,
        token_start: 0,
        token_count: 1,
        diag_zero_win: 0,
        diag_one_iter: 0,
        fpm_alpha_m: 0.01,
        fpm_tau: 0.5,
        fpm_persist_beta: 0.01,
    };

    let len_sq = d * d;
    let s_in = vec![0.1f32; d];
    let w_q = vec![0.02f32; len_sq];
    let w_k = vec![0.02f32; len_sq];
    let w_v = vec![0.02f32; len_sq];
    let w_o = vec![0.02f32; len_sq];
    let w_in = vec![0.0f32; len_sq];
    let w_x = vec![0.0f32; len_sq];
    let w_out = vec![0.0f32; len_sq];
    let a_log = vec![-0.5f32; d];
    let norm = vec![1.0f32; d];

    let result = bridge.run_forward(
        &device, &queue, &shape, &s_in, &w_q, &w_k, &w_v, &w_o, &w_in, &w_x, &w_out, &a_log, &norm,
        true,
    );

    assert!(result.is_ok(), "GPU Forward DEQ falló");
    let (h_out, conv) = result.unwrap();
    assert_eq!(h_out.len(), b * h * d);
    assert_eq!(conv.len(), b);

    println!("H_out[0..5]: {:?}", &h_out[0..5]);
    println!("Converged flags: {:?}", conv);
}
