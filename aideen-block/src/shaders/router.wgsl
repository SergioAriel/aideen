// LOXI V8: THE ENTROPY ROUTER SHADER (PRODUCTION MATH)
// Computes logits (MatMul), softmax, routing probabilities, and token entropy.
// Determines the 'Elastic Bottleneck' tier (Dim16 or Dim64).
// Capped at 32 experts for VRAM loop stability per single Workgroup invocation.

struct ComputeShape {
    batch_size: u32,
    seq_len: u32,
    d_model: u32,
    num_experts: u32,
};

@group(0) @binding(0) var<uniform> shape: ComputeShape;

// Token inputs (embeddings processed by Mamba)
@group(0) @binding(1) var<storage, read> Mamba_Out: array<f32>;

// The Router's Heavy Weights (NUM_EXPERTS x D_MODEL)
@group(0) @binding(2) var<storage, read> Router_Weights: array<f32>;

// Output probabilities for the MoE experts [seq_len * num_experts]
@group(0) @binding(3) var<storage, read_write> MoE_Probs: array<f32>;

// Output mask block for Block-Sparse Dispatch (1 if Dim64, 0 if Dim16)
@group(0) @binding(4) var<storage, read_write> Attention_Tier_Mask: array<u32>;

const MAX_EXPERTS: u32 = 32u;
var<workgroup> logits: array<f32, MAX_EXPERTS>;

// Calculate Entropy: -sum(p * log(p))
// Executed by 1 thread per token (Simulated Vector ALU behavior)
@compute
@workgroup_size(1, 1, 1) // 1 Thread = 1 Token Routing Analysis
fn calculate_routing(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let token_idx = global_id.x;
    if (token_idx >= shape.seq_len) {
        return;
    }

    let d_model = shape.d_model;
    let num_experts = shape.num_experts;
    
    // 1. MatMul: Calculate logits (Token [1 x D_MODEL] dot Weights [NUM_EXPERTS x D_MODEL])
    var max_logit: f32 = -99999.0;
    
    for (var e: u32 = 0u; e < num_experts; e = e + 1u) {
        var dot_product: f32 = 0.0;
        let weight_row_offset = e * d_model;
        let token_offset = token_idx * d_model;
        
        for (var d: u32 = 0u; d < d_model; d = d + 1u) {
            dot_product = dot_product + (Mamba_Out[token_offset + d] * Router_Weights[weight_row_offset + d]);
        }
        
        logits[e] = dot_product;
        if (dot_product > max_logit) {
            max_logit = dot_product;
        }
    }

    // 2. Compute Softmax for routing probabilities (with numerical stability `max_logit`)
    var sum_exp: f32 = 0.0;
    for (var e: u32 = 0u; e < num_experts; e = e + 1u) {
        let p_exp = exp(logits[e] - max_logit);
        logits[e] = p_exp; // Re-using logits array to store exp temporarily
        sum_exp = sum_exp + p_exp;
    }

    // 3. Normalize into Probabilities and compute Entropy
    var token_entropy: f32 = 0.0;
    for (var e: u32 = 0u; e < num_experts; e = e + 1u) {
        let p = logits[e] / sum_exp;
        
        // Write the final probability out to VRAM (for the MoE Gating phase to use later)
        MoE_Probs[(token_idx * num_experts) + e] = p;
        
        // Accumulate Entropy: H(p) = -sum(p * log2(p))
        if (p > 0.000001) { // Prevent log(0) NaN
            token_entropy = token_entropy - (p * log2(p));
        }
    }
    
    // 4. THE TIER DISPATCH DECISION (Elastic Bottleneck Logic)
    // If entropy is low (model is certain), assign to cheap 16-Dim hardware block.
    // If entropy is high (uncertainty), assign to 64-Dim hardware block.
    // The threshold is a hyperparameter defined by the architecture.
    let entropy_threshold: f32 = 1.0; 
    
    if (token_entropy < entropy_threshold) {
        Attention_Tier_Mask[token_idx] = 0u; // Will be batched into Dim16
    } else {
        Attention_Tier_Mask[token_idx] = 1u; // Will be batched into Dim64
    }
}
