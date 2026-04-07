@group(0) @binding(0) var<storage, read> forward_grad: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> res: array<f32>;
@group(0) @binding(2) var<storage, read> cgamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_grad: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> ud_grad: array<vec2<f32>>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&forward_grad)) {
        return;
    }

    let pi = radians(180.0);
    let kz = cgamma[idx] * (2.0 * pi * res[2]);
    let scale = 0.5 / max(kz, 1e-6);
    let grad = forward_grad[idx];

    u_grad[idx] = grad * 0.5;
    ud_grad[idx] = vec2<f32>(-grad.y * scale, grad.x * scale);
}
