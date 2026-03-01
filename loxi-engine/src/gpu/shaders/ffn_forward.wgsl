// ffn_forward.wgsl
// Función PURA: (inputs, weights) -> outputs
// Sin estado persistente ni mutación de pesos.

@group(0) @binding(0)
var<storage, read> inputs: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

@group(0) @binding(2)
var<storage, read_write> outputs: array<f32>;

struct PushConstants {
    in_dim: u32,
    out_dim: u32,
}

var<push_constant> pc: PushConstants;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    if (row >= pc.out_dim) {
        return;
    }
    
    var sum = 0.0;
    for (var i = 0u; i < pc.in_dim; i = i + 1u) {
        let input_val = inputs[i];
        let weight_val = weights[i * pc.out_dim + row]; // Column-major layout (Nalgebra)
        sum = sum + input_val * weight_val;
    }
    
    outputs[row] = sum;
}
