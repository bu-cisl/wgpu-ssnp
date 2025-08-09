@group(0) @binding(0) var<storage, read> input_n: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output_result: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: vec3<f32>; // res_z, dz, n0

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input_n)) {
        return;
    }

    let pi = radians(180.0);
    let a = input_n[i].x; // real
    let b = input_n[i].y; // imag

    let real_part = a * (2.0 * params.z + a) - b * b;
    let imag_part = b * (2.0 * params.z + 2.0 * a);

    let const_factor = pow(2.0 * pi * params.x / params.z, 2.0) * params.y;

    output_result[i] = const_factor * vec2<f32>(real_part, imag_part);
}
