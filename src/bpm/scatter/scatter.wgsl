@group(0) @binding(0) var<storage, read> n: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: vec3<f32>; // res_z, dz, n0

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&n)) {
        return;
    }
    
    let res_z = params.x;
    let dz = params.y;
    let n0 = params.z;

    // α = (2π * res_z / n0) * dz
    let pi = radians(180.0);
    let alpha = (2.0 * pi * res_z / n0) * dz;

    // n = a + i b
    let a = n[idx].x; // Re(n)
    let b = n[idx].y; // Im(n)

    // exp(i * α * (a + i b)) = exp(-α b) * (cos(α a) + i sin(α a))
    let phase = alpha * a;
    let exparg = -alpha * b;
    let exparg_cl = clamp(exparg, -60.0, 60.0); // avoid overflow/underflow
    let mag = exp(exparg_cl);

    output[idx] = vec2<f32>(mag * cos(phase), mag * sin(phase));
}