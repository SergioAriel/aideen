// LOXI V8: NATIVE MAMBA WGSL KERNEL (ENGINEERING LEVEL)
// A hardware-level State Space Model (SSM) parallel associative scan implementation.
// Math Reference: Parallel Prefix Scan (Bleloch / Hillis-Steele algorithm)
// Operates over the associative pair: (A_bar, B_bar * x_t)

struct ComputeShape {
    batch_size: u32,
    seq_len: u32,
    d_model: u32,
    d_state: u32,
};

@group(0) @binding(0) var<uniform> shape: ComputeShape;

// The Sequence Tokens (Input)
@group(0) @binding(1) var<storage, read> X_in: array<f32>;

// The SSM Discretized Matrices
@group(0) @binding(2) var<storage, read> dt: array<f32>;
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read> C: array<f32>;

// The output (Scan Result)
@group(0) @binding(6) var<storage, read_write> Y_out: array<f32>;

// Internal Workgroup SRAM for fast Prefix Scan (Shared Memory)
// Capped at typical workgroup limits (e.g., 256 for optimal occupancy)
const TILE_SIZE: u32 = 256u;
var<workgroup> s_A: array<f32, TILE_SIZE>;
var<workgroup> s_Bx: array<f32, TILE_SIZE>;

@compute
@workgroup_size(256, 1, 1) // 256 threads working on a sequence chunk
fn mamba_parallel_scan(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let t = local_id.x; 
    let channel = group_id.y; 

    if (channel >= shape.d_model) {
        return;
    }

    // Enable f16 precision extension here (Uncomment in production WebGPU environments)
    // enable f16;
    // We currently use f32 to match Rust's memory alignment, but f16 halves VRAM footprint precisely as noted.
    
    // Calculate how many 256-token tiles we must process to cover the sequence
    let num_chunks = (shape.seq_len + TILE_SIZE - 1u) / TILE_SIZE;
    
    // The continuous hidden state carried over between sequential tiles
    var running_h_prev: f32 = 0.0;

    for (var chunk: u32 = 0u; chunk < num_chunks; chunk = chunk + 1u) {
        let global_t = (chunk * TILE_SIZE) + t;
        let is_valid_token = global_t < shape.seq_len;

        var a_bar = 0.0;
        var b_x = 0.0;
        var offset = 0u;
        var c_val = 0.0;

        if (is_valid_token) {
            offset = (channel * shape.seq_len) + global_t;
            let x_val = X_in[offset];
            let a_val = A[channel];
            let b_val = B[channel];
            let delta = dt[offset];
            c_val = C[channel];
            
            a_bar = exp(delta * a_val); 
            b_x = ((a_bar - 1.0) / a_val) * b_val * x_val;
        }

        // Bridge the state from the previous chunk to the very first token of this new chunk
        if (t == 0u) {
            b_x = b_x + (a_bar * running_h_prev);
        }

        s_A[t] = a_bar;
        s_Bx[t] = b_x;

        workgroupBarrier();

        // HILLIS-STEELE PARALLEL PREFIX SCAN
        for (var step: u32 = 1u; step < TILE_SIZE; step = step * 2u) {
            var new_a: f32 = s_A[t];
            var new_bx: f32 = s_Bx[t];

            if (t >= step) {
                let left_a = s_A[t - step];
                let left_bx = s_Bx[t - step];
                new_a = s_A[t] * left_a;
                new_bx = (s_A[t] * left_bx) + s_Bx[t];
            }

            workgroupBarrier(); 

            s_A[t] = new_a;
            s_Bx[t] = new_bx;

            workgroupBarrier(); 
        }

        // Write the chunk history to VRAM 
        if (is_valid_token) {
            let h_t = s_Bx[t];
            Y_out[offset] = c_val * h_t;
        }
        
        // Save the absolutely final temporal state of this chunk to carry to the next chunk
        if (t == TILE_SIZE - 1u) {
            running_h_prev = s_Bx[t];
        }
        
        workgroupBarrier();
    }
}
