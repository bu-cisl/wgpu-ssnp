@group(0) @binding(0) var<storage, read> ud: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> fft: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&out)) {
        return;
    }

    out[i] = ud[i] - fft[i];
}
