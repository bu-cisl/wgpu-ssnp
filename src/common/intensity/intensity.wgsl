@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> intensity: u32;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&output)) {
        return;
    }

    output[i] = length(input[i]); // abs(complex)
    if (intensity==1) {
        output[i] = pow(output[i], 2.0);
    }
}
