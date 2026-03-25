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
                required_features: wgpu::Features::SUBGROUP,
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
    let bridge = RustDeqBridge::new(&device, d as u32, h as u32, b as u32, 1);

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
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
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

#[tokio::test]
async fn test_cg_solver_wgpu_dispatch() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Test CG Device"),
                required_features: wgpu::Features::SUBGROUP,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .unwrap();

    let b = 1;
    let d = 512;
    let h = 8;
    let cg_bridge =
        aideen_block::cg_bridge::RustCgBridge::new(&device, d as u32, h as u32, b as u32);

    let shape = aideen_block::cg_bridge::CGComputeShape {
        batch_size: b as u32,
        d_model: d as u32,
        h_slots: h as u32,
        adj_iters: 5,
        epsilon: 1e-3,
        curr_iter: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
        damping: 0.9,
    };

    let len_sq = d * d;
    let s_in = vec![0.1f32; d];
    let h_star = vec![0.2f32; h * d];
    let dl_dh_pooled = vec![0.05f32; d];

    let w_empty = vec![0.02f32; len_sq];
    let v_empty = vec![0.1f32; d];

    let result = cg_bridge.run_backward(
        &device,
        &queue,
        &shape,
        &s_in,
        &h_star,
        &dl_dh_pooled,
        &w_empty,
        &w_empty,
        &w_empty,
        &w_empty,
        &w_empty,
        &w_empty,
        &w_empty,
        &v_empty,
        &v_empty,
        true,
    );

    assert!(result.is_ok(), "GPU CG Solver falló");
    let v_out = result.unwrap();

    assert_eq!(v_out.len(), b * h * d);
    println!("V_out (CG) [0..5]: {:?}", &v_out[0..5]);
}
