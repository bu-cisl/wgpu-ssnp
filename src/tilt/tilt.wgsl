@group(0) @binding(0) var<storage, read> angles : array<f32>;
@group(0) @binding(1) var<storage, read> shape : array<u32>;
@group(0) @binding(2) var<storage, read> res : array<f32>;
@group(0) @binding(3) var<storage, read_write> factors : array<f32>;
@group(0) @binding(4) var<uniform> NA : f32;
@group(0) @binding(5) var<uniform> trunc_flag : u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    
    // Total outputs = 2 × number of angles
    let total_factors = arrayLength(&angles) * 2;
    if (idx >= total_factors) {
        return; // Exit if out of bounds
    }

    let angle_idx = idx / 2;
    let angle = angles[angle_idx];
    let is_sin_component = (idx % 2 == 0); // sin for even indices, cos for odd

    let c_ba = NA * select(cos(angle), sin(angle), is_sin_component);
    let norm = select(
        f32(shape[1]) * res[2], // y-component (cos)
        f32(shape[0]) * res[1], // x-component (sin)
        is_sin_component
    );

    var factor = c_ba * norm;
    if (trunc_flag == 1u) {
        factor = trunc(factor);
    }

    factors[idx] = factor;
}