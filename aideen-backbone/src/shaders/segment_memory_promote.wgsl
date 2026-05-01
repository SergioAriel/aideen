override ASSOC_BANKS: u32 = 1u;
override ASSOC_RANK: u32 = 32u;
override SEGMENT_MEMORY_BETA: f32 = 0.5;

struct PromoteParams {
    d_model: u32,
    h_slots: u32,
    batch_size: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> Params: PromoteParams;
@group(0) @binding(1) var<storage, read_write> MState: array<f32>;
@group(0) @binding(2) var<storage, read_write> AssocBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> HistWeights: array<f32>;

fn hist_mat_len(d: u32) -> u32 { return d * d; }
fn hist_scale_base(d: u32, h_slots: u32) -> u32 { return hist_mat_len(d); }
fn hist_bias_base(d: u32, h_slots: u32) -> u32 { return hist_scale_base(d, h_slots) + h_slots * d; }
fn hist_gate_base(d: u32, h_slots: u32) -> u32 { return hist_bias_base(d, h_slots) + h_slots * d; }
fn slot_anchor_base(d: u32, h_slots: u32) -> u32 { return hist_gate_base(d, h_slots) + h_slots; }
fn w_k_write_base(d: u32, h: u32) -> u32 { return slot_anchor_base(d, h) + h * d; }
fn w_v_write_base(d: u32, h: u32) -> u32 { return w_k_write_base(d, h) + h * d * ASSOC_RANK; }
fn b_delta_base(d: u32, h: u32) -> u32 { return w_v_write_base(d, h) + h * d * ASSOC_RANK; }
fn hist_selective_flag_base(d: u32, h: u32) -> u32 { return b_delta_base(d, h) + h * d; }
fn hist_gate_query_base(d: u32, h: u32) -> u32 { return hist_selective_flag_base(d, h) + 21u; }
fn w_write_gate_base(d: u32, h_slots: u32) -> u32 { return hist_gate_query_base(d, h_slots) + h_slots * d; }
fn b_write_mem_base(d: u32, h_slots: u32) -> u32 { return w_write_gate_base(d, h_slots) + h_slots * d; }
fn hhist_gamma_base(d: u32, h: u32) -> u32 { return b_write_mem_base(d, h) + h; }
fn w_retain_up_base(d: u32, h: u32) -> u32 { return hhist_gamma_base(d, h) + h; }
fn w_retain_down_base(d: u32, h: u32) -> u32 { return w_retain_up_base(d, h) + h * d * ASSOC_RANK; }
fn b_retain_base(d: u32, h: u32) -> u32 { return w_retain_down_base(d, h) + h * d * ASSOC_RANK; }
fn w_q_mem_base(d: u32, h: u32) -> u32 { return b_retain_base(d, h) + h * d; }
fn w_k_mem_base(d: u32, h: u32) -> u32 { return w_q_mem_base(d, h) + h * d * ASSOC_RANK; }
fn b_read_mem_base(d: u32, h: u32) -> u32 { return w_k_mem_base(d, h) + h * d * ASSOC_RANK; }
fn w_k_assoc_base(d: u32, h: u32) -> u32 { return b_read_mem_base(d, h) + h; }
fn w_v_assoc_base(d: u32, h: u32) -> u32 { return w_k_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn w_q_assoc_base(d: u32, h: u32) -> u32 { return w_v_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn alpha_assoc_base(d: u32, h: u32) -> u32 { return w_q_assoc_base(d, h) + h * d * ASSOC_RANK; }

@compute @workgroup_size(64, 1, 1)
fn segment_memory_promote_main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let d = gid.x;
    let batch = gid.y;
    if (d >= Params.d_model || batch >= Params.batch_size || Params.h_slots < 2u) {
        return;
    }

    let reserved_slot = Params.h_slots - 1u;
    let assoc_bank_stride = ASSOC_RANK + Params.d_model + 1u;
    let assoc_slot_stride = ASSOC_BANKS * assoc_bank_stride;
    let usage_idx = assoc_bank_stride - 1u;
    let reserved_m_base = (batch * Params.h_slots + reserved_slot) * Params.d_model;
    let qassoc_base = w_q_assoc_base(Params.d_model, Params.h_slots) + reserved_slot * Params.d_model * ASSOC_RANK;
    let alpha_idx = alpha_assoc_base(Params.d_model, Params.h_slots) + reserved_slot;

    var query: array<f32, 32>;
    for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
        var acc = 0.0;
        for (var j = 0u; j < Params.d_model; j = j + 1u) {
            let m = MState[reserved_m_base + j];
            acc = acc + HistWeights[qassoc_base + j * ASSOC_RANK + r] * m;
        }
        query[r] = tanh(acc);
    }

    var max_score = -1.0e30;
    for (var slot = 0u; slot < reserved_slot; slot = slot + 1u) {
        let slot_base = (batch * Params.h_slots + slot) * assoc_slot_stride;
        for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
            let bank_base = slot_base + bank * assoc_bank_stride;
            let usage = max(AssocBuf[bank_base + usage_idx], 0.0);
            if (usage <= 0.0) {
                continue;
            }
            var score = 0.0;
            for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                score = score + query[r] * AssocBuf[bank_base + r];
            }
            score = score + log(max(usage, 1.0e-6));
            if (score > max_score) {
                max_score = score;
            }
        }
    }

    var weighted_value = 0.0;
    var total_weight = 0.0;

    for (var slot = 0u; slot < reserved_slot; slot = slot + 1u) {
        let slot_base = (batch * Params.h_slots + slot) * assoc_slot_stride;
        for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
            let bank_base = slot_base + bank * assoc_bank_stride;
            let usage = max(AssocBuf[bank_base + usage_idx], 0.0);
            if (usage <= 0.0) {
                continue;
            }
            var score = 0.0;
            for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                score = score + query[r] * AssocBuf[bank_base + r];
            }
            score = score + log(max(usage, 1.0e-6));
            let weight = exp(score - max_score);
            let value_idx = bank_base + ASSOC_RANK + d;
            weighted_value = weighted_value + weight * AssocBuf[value_idx];
            total_weight = total_weight + weight;
        }
    }

    let prev = MState[reserved_m_base + d];
    if (total_weight <= 1.0e-8) {
        return;
    }

    let candidate = weighted_value / total_weight;
    let alpha = 1.0 / (1.0 + exp(-HistWeights[alpha_idx]));
    let beta = SEGMENT_MEMORY_BETA * alpha;
    MState[reserved_m_base + d] = (1.0 - beta) * prev + beta * candidate;
}
