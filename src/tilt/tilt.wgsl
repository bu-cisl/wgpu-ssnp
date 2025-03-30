@group(0) @binding(0) var<storage, read> angles : array<f32>;  // Flattened (N, 2) tensor
@group(0) @binding(1) var<storage, read> shape : array<u32>;   // Shape [height, width]
@group(0) @binding(2) var<storage, read> res : array<f32>;     // Resolution [x, y, z]
@group(0) @binding(3) var<storage, read_write> factors : array<f32>; // Output (2, 2, N) tensor
@group(0) @binding(4) var<uniform> NA : f32;
@group(0) @binding(5) var<uniform> trunc_flag : u32; // 0 for false, 1 for true

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    let N = arrayLength(&factors) / 4; // Total number of angle pairs (2x2xN)
    
    if (idx >= N) {
        return;
    }
    
    // Get the angle pair (θ1, θ2) for this index
    let angle1 = angles[2u * idx];
    let angle2 = angles[2u * idx + 1u];
    
    // Calculate sine and cosine components
    let sin_comp = sin(angle1);
    let cos_comp = cos(angle2);
    
    // Calculate norm factors (using only y and z resolutions)
    let norm_x = f32(shape[0u]) * res[1u]; // shape[0] * res[1] (y resolution)
    let norm_y = f32(shape[1u]) * res[2u]; // shape[1] * res[2] (z resolution)
    
    // Calculate all four components
    var factor00 = NA * sin_comp * norm_x; // First sin component
    var factor01 = NA * sin_comp * norm_y; // Second sin component
    var factor10 = NA * cos_comp * norm_x; // First cos component
    var factor11 = NA * cos_comp * norm_y; // Second cos component
    
    // Apply truncation if needed
    if (trunc_flag == 1u) {
        factor00 = trunc(factor00);
        factor01 = trunc(factor01);
        factor10 = trunc(factor10);
        factor11 = trunc(factor11);
    }
    
    // Store results in (2, 2, N) layout
    // First dimension (sin/cos):
    //   [0][*][*] = sin components
    //   [1][*][*] = cos components
    let base_idx = 4u * idx;
    factors[base_idx] = factor00;      // [0][0][idx]
    factors[base_idx + 1u] = factor01; // [0][1][idx]
    factors[base_idx + 2u] = factor10; // [1][0][idx]
    factors[base_idx + 3u] = factor11; // [1][1][idx]
}