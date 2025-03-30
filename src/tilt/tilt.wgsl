@group(0) @binding(0) var<storage, read> angles : array<f32>;
@group(0) @binding(1) var<storage, read> shape : array<u32>;
@group(0) @binding(2) var<storage, read> res : array<f32>;
@group(0) @binding(3) var<storage, read_write> factors : array<f32>;
@group(0) @binding(4) var<uniform> NA : f32;
@group(0) @binding(5) var<uniform> trunc_flag : u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    
    // Total output elements = shape[0] * shape[1] * 2 (2 components per position)
    let total_elements = shape[0u] * shape[1u] * 2u;
    if (idx >= total_elements) {
        return;
    }

    // Calculate spatial coordinates (i,j) and component index (0=x, 1=y)
    let i = idx / (shape[1u] * 2u);
    let j = (idx / 2u) % shape[1u];
    let component = idx % 2u; // 0 for x, 1 for y

    // Get angle (only first angle pair is used for spatial position [i,j])
    let angle_idx = i * shape[1u] + j;
    let angle = angles[angle_idx % arrayLength(&angles)];

    // Compute c_ba component (sin for x, cos for y)
    let c_ba = NA * select(cos(angle), sin(angle), component == 0u);

    // Compute norm component
    let norm = select(
        f32(shape[1u]) * res[2u], // y-component uses res[2]
        f32(shape[0u]) * res[1u], // x-component uses res[1]
        component == 0u
    );

    // Compute and optionally truncate
    var factor = c_ba * norm;
    if (trunc_flag == 1u) {
        factor = trunc(factor);
    }

    factors[idx] = factor;
}