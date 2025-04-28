@group(0) @binding(0) var<storage, read> forward: array<vec2<f32>>; 
@group(0) @binding(1) var<storage, read> pupil: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_result: array<vec2<f32>>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&output_result)) {
        return;
    }

    output_result[i] = forward[i] * vec2<f32>(pupil[i], pupil[i]);
}