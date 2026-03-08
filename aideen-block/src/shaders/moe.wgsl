// Loxi V8 — Mixture of Experts (MoE)
//
// Implements:
//   1. RMSNorm
//   2. Router (logits = x @ router_w^T)
//   3. Adaptive Top-K selection (CPU-side)
//   4. Sequential expert execution (Top-k) with RMSNorm matching PyTorch

struct Shape {
    d_model: u32,
    num_experts: u32,
    expert_hidden: u32,
    _pad: u32,
};

struct Params {
    weight: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> hidden: array<f32>; // [d_model]
@group(0) @binding(2) var<storage, read> norm_w: array<f32>; // [d_model]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [d_model] (residual)

@group(1) @binding(0) var<storage, read> router_fc1: array<f32>; // [num_experts, d_model]
@group(1) @binding(1) var<storage, read> router_fc2: array<f32>; // Not used here if expert

@group(2) @binding(0) var<uniform> params: Params;

var<workgroup> shared_sq_sum: f32;
var<workgroup> local_sums: array<f32, 256>;

// --- Router Kernel ---
@compute @workgroup_size(256, 1, 1)
fn moe_router(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d_model = shape.d_model;
    let num_experts = shape.num_experts;

    // 1. RMSNorm
    var local_sum: f32 = 0.0;
    for (var i: u32 = lid.x; i < d_model; i += 256u) {
        let v = hidden[i];
        local_sum += v * v;
    }
    local_sums[lid.x] = local_sum;
    workgroupBarrier();

    if (lid.x == 0u) {
        var total_sum: f32 = 0.0;
        for (var i = 0u; i < 256u; i++) { total_sum += local_sums[i]; }
        shared_sq_sum = total_sum;
    }
    workgroupBarrier();

    let rms_inv = 1.0 / sqrt(shared_sq_sum / f32(d_model) + 1e-8);

    // 2. Router GEMV: logits[e] = sum_d(router_w[e,d] * (x[d] * norm_w[d] * rms_inv))
    if (gid.x < num_experts) {
        let e = gid.x;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < d_model; d++) {
            let x_normed = hidden[d] * norm_w[d] * rms_inv;
            acc += router_fc1[e * d_model + d] * x_normed;
        }
        output[e] = acc;
    }
}

// --- Expert Kernel (FC1 -> SiLU -> FC2) ---
@group(1) @binding(0) var<storage, read> fc1: array<f32>; // [expert_hidden, d_model]
@group(1) @binding(1) var<storage, read> fc2: array<f32>; // [d_model, expert_hidden]

@compute @workgroup_size(256, 1, 1)
fn expert_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d_model = shape.d_model;
    let expert_hidden = shape.expert_hidden;
    let gate_weight = params.weight;

    // 1. RMSNorm
    var local_sum: f32 = 0.0;
    for (var i: u32 = lid.x; i < d_model; i += 256u) {
        let v = hidden[i];
        local_sum += v * v;
    }
    local_sums[lid.x] = local_sum;
    workgroupBarrier();

    if (lid.x == 0u) {
        var total_sum: f32 = 0.0;
        for (var i = 0u; i < 256u; i++) { total_sum += local_sums[i]; }
        shared_sq_sum = total_sum;
    }
    workgroupBarrier();

    let rms_inv = 1.0 / sqrt(shared_sq_sum / f32(d_model) + 1e-8);

    // 2. FC1 -> SiLU -> FC2 (Streamed per thread gid.x < d_model)
    // To match PyTorch: out = FC2(SiLU(FC1(normed_x)))
    if (gid.x < d_model) {
        let out_idx = gid.x;
        var total_expert_out: f32 = 0.0;

        for (var h: u32 = 0u; h < expert_hidden; h++) {
            // FC1 row for neuron h
            var inner_acc: f32 = 0.0;
            for (var d: u32 = 0u; d < d_model; d++) {
                let x_normed = hidden[d] * norm_w[d] * rms_inv;
                inner_acc += fc1[h * d_model + d] * x_normed;
            }
            
            // SiLU
            let silu_out = inner_acc / (1.0 + exp(-inner_acc));
            
            // FC2 accumulation for output gid.x
            total_expert_out += fc2[out_idx * expert_hidden + h] * silu_out;
        }

        output[out_idx] += total_expert_out * gate_weight;
    }
}
