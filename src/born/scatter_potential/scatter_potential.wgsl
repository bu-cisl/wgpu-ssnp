struct Params {
    res_z: f32,
    n0: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<storage, read> input_n: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output_potential: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_potential)) {
        return;
    }

    let pi = radians(180.0);
    let factor = pow((2.0 * pi * params.res_z / params.n0), 2.0);
    let delta_n = input_n[idx].x;
    let potential = factor * delta_n * (2.0 * params.n0 + delta_n);

    output_potential[idx] = vec2<f32>(potential, 0.0);
}
