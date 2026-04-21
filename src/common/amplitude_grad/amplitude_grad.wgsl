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
    let pred_intensity = value.x * value.x + value.y * value.y;
    let pred_amp = sqrt(pred_intensity + 1e-8);
    let meas_amp = sqrt(measured[i] + 1e-8);
    let residual = (pred_amp - meas_amp) * params;
    let scale = residual / pred_amp;

    output_grad[i] = vec2<f32>(
        value.x * scale,
        value.y * scale
    );
}