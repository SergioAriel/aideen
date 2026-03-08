struct UpdateUniforms {
    d_model: u32,
    h_slots: u32,
    lr: f32,
    grad_scale: f32,
    ternary_flag: u32,
    weight_decay: f32,
    seq_len: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> v_adjoint: array<f32>; 
@group(0) @binding(2) var<storage, read> q_input: array<f32>;   // S_in (Batch de tokens)
@group(0) @binding(3) var<storage, read> h_star: array<f32>;
@group(0) @binding(4) var<storage, read_write> debug_log: array<f32>;

// Pesos a actualizar
@group(1) @binding(0) var<storage, read_write> W_q: array<f32>;
@group(1) @binding(1) var<storage, read_write> W_k: array<f32>;
@group(1) @binding(2) var<storage, read_write> W_v: array<f32>;
@group(1) @binding(3) var<storage, read_write> W_o: array<f32>;
@group(1) @binding(4) var<storage, read_write> W_in: array<f32>;
@group(1) @binding(5) var<storage, read_write> W_x: array<f32>;
@group(1) @binding(6) var<storage, read_write> W_out: array<f32>;
@group(1) @binding(7) var<storage, read_write> A_log: array<f32>;
@group(1) @binding(8) var<storage, read_write> NormScale: array<f32>;

@compute
@workgroup_size(16, 16, 1)
fn fused_deq_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let d = params.d_model;

    if (row >= d || col >= d) { return; }
    let idx = row * d + col;

    var g_wq   = 0.0;
    var g_wk   = 0.0;
    var g_wv   = 0.0;
    var g_wo   = 0.0;
    var g_wx   = 0.0;
    var g_wout = 0.0; 
    var g_win  = 0.0;

    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;

    let h_slots = params.h_slots;
    for (var t = 0u; t < params.seq_len; t = t + 1u) {
        let q = q_input[t * d + col];
        let t_off = t * h_slots * d;
        
        for (var s = 0u; s < h_slots; s = s + 1u) {
            let s_off = s * d;
            let v_s = v_adjoint[t_off + s_off + row];
            let h_s = h_star[t_off + s_off + col];

            let v_safe = clamp(v_s, -10.0, 10.0);
            let h_safe = clamp(h_s, -10.0, 10.0);

            // W_in y W_out usan q: rutas que mezclan la entrada externa
            g_win  += v_safe * q;
            g_wout += v_safe * q;

            // Pesos de atención y SSM-input usan h_star: rutas recurrentes puras
            g_wq += v_safe * h_safe;
            g_wk += v_safe * h_safe;
            g_wv += v_safe * h_safe;
            g_wo += v_safe * h_safe;
            g_wx += v_safe * h_safe;

            // Debug v11.0: Capturar señales del último token para diagnóstico
            if (idx == 0u && t == params.seq_len - 1u && s == 0u) {
                debug_log[0] = v_s;
                debug_log[4] = q; 
                debug_log[5] = h_s;  
            }
        }
    }
    
    // Normalize gradients by sequence length
    let seq_scale = 1.0 / max(f32(params.seq_len), 1.0);
    g_win  *= seq_scale;
    g_wout *= seq_scale;
    g_wq   *= seq_scale;
    g_wk   *= seq_scale;
    g_wv   *= seq_scale;
    g_wo   *= seq_scale;
    g_wx   *= seq_scale;

    // Gradient Clipping Hard (per-element)
    let clip = 0.5;
    let s_wq   = clamp(lr * g_wq   * grad_scale, -clip, clip);
    let s_wk   = clamp(lr * g_wk   * grad_scale, -clip, clip);
    let s_wv   = clamp(lr * g_wv   * grad_scale, -clip, clip);
    let s_wo   = clamp(lr * g_wo   * grad_scale, -clip, clip);
    let s_wx   = clamp(lr * g_wx   * grad_scale, -clip, clip);
    let s_wout = clamp(lr * g_wout * grad_scale, -clip, clip);
    let s_win  = clamp(lr * g_win  * grad_scale, -clip, clip);

    // --- ACTUALIZACIÓN CON WEIGHT DECAY ---
    W_q[idx]   = W_q[idx]   * wd_factor - s_wq;
    W_k[idx]   = W_k[idx]   * wd_factor - s_wk;
    W_v[idx]   = W_v[idx]   * wd_factor - s_wv;
    W_o[idx]   = W_o[idx]   * wd_factor - s_wo;
    W_in[idx]  = W_in[idx]  * wd_factor - s_win;
    W_x[idx]   = W_x[idx]   * wd_factor - s_wx;
    W_out[idx] = W_out[idx] * wd_factor - s_wout;

    // A_log (Scalar dynamics) — usa gradiente de W_x (mismo camino SSM)
    if (col == 0u) {
        A_log[row] = A_log[row] * wd_factor - clamp(lr * g_wx * grad_scale, -0.1, 0.1);
    }

    if (idx == 0u) {
        debug_log[8] = 123.456; 
    }
}
