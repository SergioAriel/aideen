override ASSOC_BANKS: u32 = 1u;
override ASSOC_RANK: u32 = 32u;

const ASSOC_OCCUPIED_THRESHOLD: f32 = 1.0e-4;
const ASSOC_USAGE_DECAY: f32 = 0.999;

struct PromoteParams {
    d_model: u32,
    h_slots: u32,
    batch_size: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> Params: PromoteParams;
@group(0) @binding(1) var<storage, read_write> AssocPersistent: array<f32>;
@group(0) @binding(2) var<storage, read_write> AssocLocal: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn assoc_persistent_promote_main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let assoc_bank_stride = ASSOC_RANK + Params.d_model + 1u;
    let total = Params.batch_size * Params.h_slots * ASSOC_BANKS * assoc_bank_stride;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let elem_in_bank = idx % assoc_bank_stride;
    let usage_idx = assoc_bank_stride - 1u;
    let bank_base = idx - elem_in_bank;

    let local_usage = max(AssocLocal[bank_base + usage_idx], 0.0);
    let persistent_usage = max(AssocPersistent[bank_base + usage_idx], 0.0);
    let local_occupied = local_usage > ASSOC_OCCUPIED_THRESHOLD;

    if (elem_in_bank == usage_idx) {
        let decayed = persistent_usage * ASSOC_USAGE_DECAY;
        let next_usage = select(decayed, max(decayed, clamp(local_usage, 0.0, 1.0)), local_occupied);
        AssocPersistent[idx] = next_usage;
        return;
    }

    if (!local_occupied) {
        if (persistent_usage <= ASSOC_OCCUPIED_THRESHOLD) {
            AssocPersistent[idx] = 0.0;
        }
        return;
    }

    let prev = AssocPersistent[idx];
    let src = AssocLocal[idx];
    let beta = clamp(local_usage, 0.0, 1.0);
    AssocPersistent[idx] = select(src, (1.0 - beta) * prev + beta * src, persistent_usage > ASSOC_OCCUPIED_THRESHOLD);
}

@compute @workgroup_size(256, 1, 1)
fn assoc_persistent_load_main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let assoc_bank_stride = ASSOC_RANK + Params.d_model + 1u;
    let total = Params.batch_size * Params.h_slots * ASSOC_BANKS * assoc_bank_stride;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    // Simple copy from Persistent back to Local.
    // This allows the DEQ solver to start with the consolidated memory state.
    AssocLocal[idx] = AssocPersistent[idx];
}

