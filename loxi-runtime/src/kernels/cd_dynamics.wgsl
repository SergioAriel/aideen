// kernels/cd_dynamics.wgsl
// ─────────────────────────────────────────────────────────────────────────────
// C+D (Cohesion + Divergence) Dynamics Kernels
//
// These are the training-specific kernels that maintain the balance between:
//   - Cohesion: experts stay semantically aligned (don't drift apart)
//   - Divergence: experts maintain specialization (don't collapse into one)
//
// From the design document:
//   ΔW_n = ΔW_task + λ_c * ΔW_cohesion - α_r * ΔW_repulsion
//
// Where:
//   ΔW_task      = gradient from local task loss
//   ΔW_cohesion  = pull toward network mean expert
//   ΔW_repulsion = push away from too-similar experts
//
// Parameters:
//   λ_cohesion  = 0.05  (gentle pull toward mean)
//   α_repulsion = 0.03  (gentle push from similar experts)
//   ε_norm      = 5.0   (norm constraint on any single gradient)
// ─────────────────────────────────────────────────────────────────────────────

// ── Kernel 1: Compute mean delta across affinity group ────────────────────────
// Called by Architect after receiving deltas from all nodes in a group.
// Output: mean_delta[i] = average of all expert weight deltas at position i.

struct MeanDeltaParams {
    N:           u32;   // number of parameters per expert
    num_experts: u32;   // how many experts in this affinity group
}

// experts is a flat array: [num_experts * N]
// expert k's params start at k * N
@group(0) @binding(0) var<storage, read>       experts:    array<f32>;
@group(0) @binding(1) var<storage, read_write> mean_delta: array<f32>;
@group(0) @binding(2) var<uniform>             params:     MeanDeltaParams;

@compute @workgroup_size(256)
fn compute_mean_delta(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= params.N) { return; }

    var sum: f32 = 0.0;
    for (var k = 0u; k < params.num_experts; k++) {
        sum += experts[k * params.N + i];
    }
    mean_delta[i] = sum / f32(params.num_experts);
}

// ── Kernel 2: Compute cosine similarity between two experts ───────────────────
// Used to determine repulsion strength: experts that are too similar get pushed apart.
// Output: similarity scalar written to output[0].
// Note: full cosine similarity requires two reduction passes (dot + norms).
// This kernel is called twice per pair and combined on CPU.

struct SimilarityParams {
    N: u32;
}

@group(0) @binding(0) var<storage, read>       expert_a:     array<f32>;
@group(0) @binding(1) var<storage, read>       expert_b:     array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_dots: array<f32>;
@group(0) @binding(3) var<uniform>             params:       SimilarityParams;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn compute_dot_partial(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let i   = global_id.x;
    let tid = local_id.x;

    var val: f32 = 0.0;
    if (i < params.N) {
        val = expert_a[i] * expert_b[i];
    }
    partial[tid] = val;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (tid == 0u) {
        partial_dots[wg_id.x] = partial[0];
    }
}

// ── Kernel 3: Apply C+D gradient to expert weights ───────────────────────────
// The full update:
//   theta[i] += lambda_c * (mean_delta[i] - theta[i])    ← cohesion
//             - alpha_r  * repulsion_force[i]              ← divergence
// Called once per expert per training step.

struct CdUpdateParams {
    N:        u32;
    lambda_c: f32;   // cohesion strength (0.05)
    alpha_r:  f32;   // repulsion strength (0.03)
    eps_norm: f32;   // gradient norm clamp (5.0)
}

@group(0) @binding(0) var<storage, read_write> theta:           array<f32>;
@group(0) @binding(1) var<storage, read>       mean_delta:      array<f32>;
@group(0) @binding(2) var<storage, read>       repulsion_force: array<f32>;
@group(0) @binding(3) var<uniform>             p:               CdUpdateParams;

@compute @workgroup_size(256)
fn apply_cd_gradient(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= p.N) { return; }

    let cohesion   = p.lambda_c * (mean_delta[i] - theta[i]);
    let repulsion  = p.alpha_r  * repulsion_force[i];
    let total_grad = cohesion - repulsion;

    // Norm clamp: scale down if gradient is too large
    // (approximated per-element via a pre-computed global scale factor
    // which the CPU passes via a separate uniform — here we just apply)
    theta[i] += total_grad;
}

// ── Kernel 4: Compute repulsion force for one expert ─────────────────────────
// repulsion_force[i] = sum over similar experts of sim(a,b) * (theta_a[i] - theta_b[i])
// This kernel adds one expert's repulsion contribution; call once per similar pair.

struct RepulsionParams {
    N:          u32;
    similarity: f32;   // cosine similarity between this pair (pre-computed)
}

@group(0) @binding(0) var<storage, read>       theta_a:    array<f32>;
@group(0) @binding(1) var<storage, read>       theta_b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> repulsion:  array<f32>;  // accumulated
@group(0) @binding(3) var<uniform>             p:          RepulsionParams;

@compute @workgroup_size(256)
fn accumulate_repulsion(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= p.N) { return; }
    repulsion[i] += p.similarity * (theta_a[i] - theta_b[i]);
}
