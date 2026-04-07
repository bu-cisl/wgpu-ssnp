@group(0) @binding(0) var<storage, read> input_value: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output_result: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: f32;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&output_result)) {
        return;
    }

    output_result[i] = input_value[i] * params;
}
