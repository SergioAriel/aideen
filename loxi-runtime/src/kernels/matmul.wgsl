// kernels/matmul.wgsl
// ─────────────────────────────────────────────────────────────────────────────
// General Matrix Multiply: C = A × B
//
// A: [M, K]   (row-major, f32)
// B: [K, N]   (row-major, f32)
// C: [M, N]   (row-major, f32)
//
// Strategy: tiled GEMM with shared memory (workgroup-local cache).
// Tile size 16×16 gives good occupancy on both Metal and Vulkan.
// Works on M1 Metal shader core and any Vulkan-capable GPU.
//
// Usage: dispatch(ceil(N/16), ceil(M/16), 1) workgroups
// ─────────────────────────────────────────────────────────────────────────────

struct MatMulDims {
    M: u32,   // rows of A and C
    K: u32,   // cols of A = rows of B
    N: u32,   // cols of B and C
}

@group(0) @binding(0) var<storage, read>       A:    array<f32>;
@group(0) @binding(1) var<storage, read>       B:    array<f32>;
@group(0) @binding(2) var<storage, read_write> C:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: MatMulDims;

// Workgroup tile cache — 16×16 tiles for A and B
var<workgroup> tile_A: array<f32, 256>;  // 16×16
var<workgroup> tile_B: array<f32, 256>;  // 16×16

const TILE: u32 = 16u;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)   global_id:     vec3<u32>,
    @builtin(local_invocation_id)    local_id:      vec3<u32>,
    @builtin(workgroup_id)           workgroup_id:  vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;

    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc: f32 = 0.0;

    // Number of tiles along K dimension
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile of A: A[row, t*TILE + local_col]
        let a_col = t * TILE + local_col;
        if (row < dims.M && a_col < dims.K) {
            tile_A[local_row * TILE + local_col] = A[row * dims.K + a_col];
        } else {
            tile_A[local_row * TILE + local_col] = 0.0;
        }

        // Load tile of B: B[t*TILE + local_row, col]
        let b_row = t * TILE + local_row;
        if (b_row < dims.K && col < dims.N) {
            tile_B[local_row * TILE + local_col] = B[b_row * dims.N + col];
        } else {
            tile_B[local_row * TILE + local_col] = 0.0;
        }

        // Sync: all threads have loaded their tiles
        workgroupBarrier();

        // Accumulate dot product for this tile
        for (var k = 0u; k < TILE; k++) {
            acc += tile_A[local_row * TILE + k] * tile_B[k * TILE + local_col];
        }

        // Sync before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
