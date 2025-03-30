@group(0) @binding(0) var<storage, read> angles : array<f32>;
@group(0) @binding(1) var<storage, read> shape : array<u32>;
@group(0) @binding(2) var<storage, read> res : array<f32>;
@group(0) @binding(3) var<storage, read_write> factor : array<f32>;
@group(0) @binding(4) var<uniform> params: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&angles)) {
        return;
    }

    let sin_angle = sin(angles[idx]);
    let cos_angle = cos(angles[idx]);
    let c_ba = vec2<f32>(sin_angle, cos_angle);

    let norm = vec2<f32>(f32(shape[1]), f32(shape[2])) * res[1..2];
    
    var factor_value = c_ba * norm;
    if (params != 0.0) {
        factor_value = trunc(factor_value);
    }

    factor[idx] = factor_value.x;
}