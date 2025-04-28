@group(0) @binding(0) var<storage, read> scatter: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> u: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&out)) {
        return;
    }

    out[i] = vec2<f32>(
        scatter[i].x * u[i].x - scatter[i].y * u[i].y, // Real part
        scatter[i].x * u[i].y + scatter[i].y * u[i].x  // Imaginary part
    );
}
