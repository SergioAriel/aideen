// AIDEEN Semba Engine v3: DEQ Forward Solver (v13.0 - "Solver Transparency")
// Optimized Injection, 16KB Shared, Safe Swap, Magic 777.7.

struct RunUniforms {
    batch_size: u32,
    d_model: u32,
    h_slots: u32,
    max_iters: u32,
    epsilon: f32,
    damping: f32,
    seq_len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(2) var<storage, read> W_q:    array<f32>;
@group(0) @binding(3) var<storage, read> W_k:    array<f32>;
@group(0) @binding(4) var<storage, read> W_v:    array<f32>;
@group(0) @binding(5) var<storage, read> W_o:    array<f32>;
@group(0) @binding(6) var<storage, read> W_in:   array<f32>;
@group(0) @binding(7) var<storage, read> W_x:   array<f32>;
@group(0) @binding(8) var<storage, read> W_out: array<f32>;
@group(0) @binding(9) var<storage, read> A_log: array<f32>;
@group(0) @binding(10) var<storage, read> NormScale: array<f32>;
@group(0) @binding(11) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(12) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(13) var<storage, read_write> Scratch: array<f32>; 
@group(0) @binding(14) var<storage, read_write> H_pooled: array<f32>;
@group(0) @binding(15) var<storage, read_write> DebugLog: array<f32>;

const WG_SIZE: u32 = 256u;
const MAX_SHARED_DIM: u32 = 4096u;

var<workgroup> state_shared: array<f32, MAX_SHARED_DIM>;
var<workgroup> shared_sums: array<f32, WG_SIZE>;

// --- v13.1 diagnostics ---
var<workgroup> hit_count: atomic<u32>;
var<workgroup> max_delta_seen: f32;
var<workgroup> last_delta: f32;

@compute @workgroup_size(256, 1, 1)
fn deq_forward_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let tid = lid.x;
    
    if (batch_idx >= shape.batch_size) { return; }

    if (tid == 0u) {
        atomicStore(&hit_count, 0u);
        max_delta_seen = 0.0;
        last_delta = 0.0;
    }
    workgroupBarrier();

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    // ... (rest of the preamble) ...
    let total_elements = h_slots * d_model;
    let h_base = batch_idx * total_elements;
    
    let stride = d_model * (h_slots * 4u + 1u);
    let batch_scratch = batch_idx * stride;
    let qkv_base = batch_scratch;
    let signal_ext_base = qkv_base + (h_slots * 3u * d_model);
    let swap_base = signal_ext_base + d_model;

    // --- 1. Inicialización de Estado Compartido (Warm Start inicial) ---
    for (var i = tid; i < total_elements; i += WG_SIZE) {
        if (i < MAX_SHARED_DIM) {
            state_shared[i] = H_curr[h_base + i];
        }
    }
    workgroupBarrier();

    var total_iters = 0u;

    // --- 2. Bucle Causal de Tokens (v11.0 Causal Drive) ---
    for (var t = 0u; t < shape.seq_len; t++) {
        // ... (existing code for injection) ...
        for (var d_out = tid; d_out < d_model; d_out += WG_SIZE) {
            var inj = 0.0;
            let s_in_base = batch_idx * (shape.seq_len * d_model) + (t * d_model); 
            for (var j = 0u; j < d_model; j++) {
                inj += W_in[j * d_model + d_out] * S_in[s_in_base + j];
            }
            Scratch[signal_ext_base + d_out] = inj;
        }
        workgroupBarrier();

        var iter = 0u;
        var converged = false;

        while (iter < shape.max_iters && !converged) {
            var local_max_delta = 0.0;
            
            for (var s = 0u; s < h_slots; s++) {
                let offset = s * d_model;
                for (var d_out = tid; d_out < d_model; d_out += WG_SIZE) {
                    let inj = Scratch[signal_ext_base + d_out];

                    var x_proj = 0.0;
                    var q = 0.0;
                    var kv = 0.0;
                    var v = 0.0;
                    for (var j = 0u; j < d_model; j++) {
                        let h_val = state_shared[offset + j];
                        let w_idx = j * d_model + d_out;
                        x_proj += W_x[w_idx] * h_val;
                        q      += W_q[w_idx] * h_val;
                        kv     += W_k[w_idx] * h_val;
                        v      += W_v[w_idx] * h_val;
                    }
                    
                    let a_bar = exp(-abs(A_log[d_out]));
                    let attn_gate = 1.0 / (1.0 + exp(-(q * kv)));
                    
                    let mixed = x_proj + (0.25 * attn_gate * v) + inj;
                    let h_prev = state_shared[offset + d_out];
                    let val = a_bar * h_prev + (1.0 - a_bar) * mixed;
                    
                    local_max_delta = max(local_max_delta, abs(val - h_prev));
                    Scratch[swap_base + offset + d_out] = val;
                }
            }
            
            workgroupBarrier();
            
            for (var i = tid; i < total_elements; i += WG_SIZE) {
                if (i < MAX_SHARED_DIM) {
                    state_shared[i] = Scratch[swap_base + i];
                }
            }
            
            workgroupBarrier();

            shared_sums[tid] = local_max_delta;
            workgroupBarrier();
            for (var step = 128u; step > 0u; step >>= 1u) {
                if (tid < step) {
                    shared_sums[tid] = max(shared_sums[tid], shared_sums[tid + step]);
                }
                workgroupBarrier();
            }
            
            if (shared_sums[0] < shape.epsilon) { converged = true; }
            iter++;
        }
        
        total_iters += iter;
        
        // --- v13.1 per-token diagnostics ---
        if (tid == 0u) {
            let d = shared_sums[0];
            last_delta = d;
            max_delta_seen = max(max_delta_seen, d);

            if (!converged) {
                _ = atomicAdd(&hit_count, 1u);
            }
        }
        workgroupBarrier();

        for (var i = tid; i < d_model; i += WG_SIZE) {
            H_pooled[batch_idx * (shape.seq_len * d_model) + (t * d_model) + i] = state_shared[i];
        }
        workgroupBarrier();
    }

    for (var i = tid; i < total_elements; i += WG_SIZE) {
        if (i < MAX_SHARED_DIM) {
            H_next[h_base + i] = state_shared[i];
        }
    }

    if (batch_idx == 0u && tid == 0u) {
        DebugLog[0] = 777.7; 
        DebugLog[1] = f32(shape.batch_size);
        DebugLog[2] = f32(shape.d_model);
        
        var max_h = 0.0;
        for (var i = 0u; i < total_elements; i++) {
             max_h = max(max_h, abs(state_shared[i]));
        }
        DebugLog[10] = f32(shape.seq_len); 
        DebugLog[11] = max_h;
        DebugLog[12] = state_shared[0]; 
        DebugLog[13] = f32(total_iters) / f32(shape.seq_len);
        DebugLog[14] = select(0.0, 1.0, atomicLoad(&hit_count) == 0u);
        DebugLog[15] = f32(atomicLoad(&hit_count));
        DebugLog[16] = max_delta_seen;
        DebugLog[17] = last_delta;
        DebugLog[20] = 999.9; 
    }
}
