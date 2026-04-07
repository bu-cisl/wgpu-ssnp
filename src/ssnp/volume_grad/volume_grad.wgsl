@group(0) @binding(0) var<storage, read> dq: array<f32>;
@group(0) @binding(1) var<storage, read> grad_value: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> u_value: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output_result: array<f32>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&output_result)) {
        return;
    }

    let grad = grad_value[i];
    let u = u_value[i];
    let real_part = grad.x * u.x + grad.y * u.y;
    output_result[i] = dq[i] * real_part;
}
