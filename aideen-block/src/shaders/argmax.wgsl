// Loxi V8 — Argmax Shader
//
// Reduces [vocab_size] logit buffer to a single u32 (argmax token index).
// This allows a tiny 4-byte readback instead of 200KB, avoiding GPU→CPU bandwidth issues.
//
// Two-pass parallel reduction:
//   Pass 1: Each workgroup finds its local max (value + index) using shared memory.
//   Pass 2: CPU reduces the partial results (64 entries max).

struct ArgmaxShape {
    vocab_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ArgmaxResult {
    max_val:   f32,
    max_idx:   u32,
};

@group(0) @binding(0) var<uniform>             shape:    ArgmaxShape;
@group(0) @binding(1) var<storage, read>       logits:   array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<ArgmaxResult>; // one per workgroup

var<workgroup> sh_val: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;

// Dispatch: (ceil(vocab_size / 256), 1, 1) workgroups of (256, 1, 1)
@compute @workgroup_size(256, 1, 1)
fn argmax_reduce(
    @builtin(global_invocation_id) gid:      vec3<u32>,
    @builtin(local_invocation_id)  lid:      vec3<u32>,
    @builtin(workgroup_id)         wid:      vec3<u32>,
) {
    let local_idx  = lid.x;
    let global_idx = gid.x;

    // Load into shared memory (clamp out-of-bounds to -inf)
    if (global_idx < shape.vocab_size) {
        sh_val[local_idx] = logits[global_idx];
        sh_idx[local_idx] = global_idx;
    } else {
        sh_val[local_idx] = -3.402823e+38; // -FLT_MAX
        sh_idx[local_idx] = 0u;
    }
    workgroupBarrier();

    // Parallel reduction within workgroup (tree reduction)
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local_idx < stride) {
            if (sh_val[local_idx + stride] > sh_val[local_idx]) {
                sh_val[local_idx] = sh_val[local_idx + stride];
                sh_idx[local_idx] = sh_idx[local_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Lane 0 writes the workgroup's result to the partials buffer
    if (local_idx == 0u) {
        partials[wid.x] = ArgmaxResult(sh_val[0], sh_idx[0]);
    }
}
