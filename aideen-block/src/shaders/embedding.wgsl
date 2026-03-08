// Loxi V8 — Embedding Gather Shader
//
// Performs a row-selection from the embedding weight matrix.
// For each token in the sequence, reads the corresponding embedding row.
//
// Weight matrix: [vocab_size × D_embed] stored row-major in f32.
// Input: token_ids array of u32.
// Output: hidden[T × D_embed] f32 — one row per token.

struct EmbedShape {
    seq_len:    u32,   // T — number of tokens
    d_embed:    u32,   // D — embedding dimension (256 for Block A input)
    vocab_size: u32,
    _pad:       u32,
};

@group(0) @binding(0) var<uniform>         shape:      EmbedShape;
@group(0) @binding(1) var<storage, read>   token_ids:  array<u32>;
@group(0) @binding(2) var<storage, read>   weights:    array<f32>;  // [vocab_size × d_embed]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;  // [seq_len × d_embed]

// Each thread handles one (token, dim) pair.
// Dispatch: (seq_len, d_embed / 64, 1) workgroups of (1, 64, 1)
@compute @workgroup_size(1, 64, 1)
fn embedding_gather(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let token_idx = gid.x;   // which position in the sequence
    let dim_chunk = gid.y;   // which 64-wide chunk of the embedding dimension
    let local_dim = gid.z;   // unused (we pack 64 per workgroup via gid.y iterations)

    if (token_idx >= shape.seq_len) { return; }

    let token_id = token_ids[token_idx];

    // Guard: clamp out-of-bound token IDs (UNK fallback to token 0)
    let safe_id = select(token_id, 0u, token_id >= shape.vocab_size);

    let src_row = safe_id * shape.d_embed;
    let dst_row = token_idx * shape.d_embed;

    // Each thread copies one f32 element
    let d = dim_chunk;
    if (d < shape.d_embed) {
        output[dst_row + d] = weights[src_row + d];
    }
}
