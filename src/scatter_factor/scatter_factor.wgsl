@group(0) @binding(0) var<storage, read> input_n: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_result: array<f32>;
@group(0) @binding(2) var<uniform> params: vec3<f32>; // res_z, dz, n0

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input_n)) {
        return;
    }
 
    let pi = radians(180.0); 
    let factor = pow(2.0 * pi * params.x / params.z, 2.0) * params.y * (2.0 * params.z + input_n[i]);
    output_result[i] = factor * input_n[i];
}