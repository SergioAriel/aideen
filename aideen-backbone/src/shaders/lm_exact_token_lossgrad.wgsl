struct ExactTokenParams {
    d_model: u32,
    vocab_size: u32,
    token_index: u32,
    target_index: u32,
    seq_scale_bits: u32,
    _pad1: array<vec4<u32>, 4>,
};

@group(0) @binding(0) var<uniform> params: ExactTokenParams;
@group(0) @binding(1) var<storage, read> logits_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> dl_scaled_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> loss_out: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn lm_exact_token_lossgrad_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }
    let vocab = params.vocab_size;
    let target_idx = params.target_index;
    let seq_scale = bitcast<f32>(params.seq_scale_bits);
    var max_logit = -3.402823e38;
    for (var v = 0u; v < vocab; v = v + 1u) {
        max_logit = max(max_logit, logits_in[v]);
    }
    var sum_exp = 0.0;
    for (var v = 0u; v < vocab; v = v + 1u) {
        sum_exp = sum_exp + exp(logits_in[v] - max_logit);
    }
    let target_logit = logits_in[target_idx];
    loss_out[0] = -target_logit + max_logit + log(sum_exp);
    for (var v = 0u; v < vocab; v = v + 1u) {
        let p = exp(logits_in[v] - max_logit) / sum_exp;
        let y = select(0.0, 1.0, v == target_idx);
        dl_scaled_out[v] = (p - y) * seq_scale;
    }
}
