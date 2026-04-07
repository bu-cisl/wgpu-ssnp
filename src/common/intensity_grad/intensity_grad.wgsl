@group(0) @binding(0) var<storage, read> field: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> measured: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_grad: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: f32;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&output_grad)) {
        return;
    }

    let value = field[i];
    let predicted = value.x * value.x + value.y * value.y;
    let residual = (predicted - measured[i]) * params;
    output_grad[i] = vec2<f32>(2.0 * residual * value.x, 2.0 * residual * value.y);
}
