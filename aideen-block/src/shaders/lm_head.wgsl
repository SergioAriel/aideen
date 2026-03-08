// Loxi V8 — LM Head Shader with RMSNorm
//
// Computes output logits from the final hidden state.
// 1. RMSNorm: x_normed = x * final_norm.weight / sqrt(sum(x^2)/D + eps)
// 2. Linear: logits[v] = dot(x_normed, vocab_head.weight[v])

struct LmHeadShape {
    d_model:    u32,
    vocab_size: u32,
    _pad0:      u32,
    _pad1:      u32,
};

@group(0) @binding(0) var<uniform>           shape:   LmHeadShape;
@group(0) @binding(1) var<storage, read>     hidden:  array<f32>;      // [d_model]
@group(0) @binding(2) var<storage, read>     norm_w:  array<f32>;      // [d_model]
@group(0) @binding(3) var<storage, read>     vocab_w: array<f32>;      // [vocab_size × d_model]
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;     // [vocab_size]

var<workgroup> shared_sq_sum: f32;
var<workgroup> local_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn lm_head_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>
) {
    let d_model = shape.d_model;
    let vocab_size = shape.vocab_size;

    // 1. RMSNorm (Parallel reduction)
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

    // 2. Linear Projection: logits[v] = dot(normed_x, vocab_w[v])
    // We repurpose the workgroup threads to compute logits in parallel
    // Since workgroup size is 256, and we want to compute all vocab_size logits.
    for (var v: u32 = gid.x; v < vocab_size; v += 256u * 1u) { // gid.x is fine for simple dispatch
        let row_offset = v * d_model;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < d_model; d++) {
            let x_normed = hidden[d] * norm_w[d] * rms_inv;
            acc += vocab_w[row_offset + d] * x_normed;
        }
        logits[v] = acc;
    }
}
