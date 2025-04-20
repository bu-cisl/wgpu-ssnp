@group(0) @binding(0) var<storage, read> angles : array<f32>;
@group(0) @binding(1) var<storage, read> shape  : array<i32>;
@group(0) @binding(2) var<storage, read> res    : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<vec2<f32>>;
@group(0) @binding(4) var<uniform> NA         : f32;
@group(0) @binding(5) var<uniform> trunc_flag : u32;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y,
                     a.x * b.y + a.y * b.x);
}
fn cexp(theta: f32) -> vec2<f32> {
    return vec2<f32>(cos(theta), sin(theta));
}

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_angles = arrayLength(&angles);
    let shape_x = u32(shape[1]);
    let shape_y = u32(shape[0]);
    let total_pixels = shape_x * shape_y;
    let total_out = num_angles * total_pixels;
    if (idx >= total_out) { return; }

    // decode (angle, y, x)
    let angle_idx = idx / total_pixels;
    let pixel_idx = idx % total_pixels;
    let x = pixel_idx % shape_x;
    let y = pixel_idx / shape_x;

    // constants
    let pi     = radians(180.0);
    let two_pi = 2.0 * pi;
    let shape_x_f = f32(shape_x);
    let shape_y_f = f32(shape_y);

    // 1) WRAP the raw angle into [0, 2π) to improve sin/cos precision
    let raw_angle    = angles[angle_idx];
    let angle_wrapped = fract(raw_angle / two_pi) * two_pi;

    // 2) compute base sin/cos on the reduced angle
    let c_sin = NA * sin(angle_wrapped);
    let c_cos = NA * cos(angle_wrapped);

    // 3) form your “spatial frequency” and immediately normalize/divide
    //    (this keeps the numbers small), THEN optionally truncate,
    //    THEN take fract() so everything is in [0,1).
    var f_sin_norm = c_sin * res[1];
    var f_cos_norm = c_cos * res[2];
    if (trunc_flag == 1u) {
        f_sin_norm = trunc(f_sin_norm * shape_y_f) / shape_y_f;
        f_cos_norm = trunc(f_cos_norm * shape_x_f) / shape_x_f;
    }
    f_sin_norm = fract(f_sin_norm);
    f_cos_norm = fract(f_cos_norm);

    // 4) build the per‐pixel phase entirely inside [0, 2π)
    let xf = f32(x);
    let yf = f32(y);
    let phase_x = two_pi * fract(f_cos_norm * xf);
    let phase_y = two_pi * fract(f_sin_norm * yf);

    let xr = cexp(phase_x);
    let yr = cexp(phase_y);
    let val = cmul(xr, yr);

    // 5) do the same for the center‐point
    let cx = shape_x / 2u;
    let cy = shape_y / 2u;
    let cf_x = f32(cx);
    let cf_y = f32(cy);
    let center_phase_x = two_pi * fract(f_cos_norm * cf_x);
    let center_phase_y = two_pi * fract(f_sin_norm * cf_y);
    let center_x = cexp(center_phase_x);
    let center_y = cexp(center_phase_y);
    let center  = cmul(center_x, center_y);

    // apply normalization
    let inv_center = vec2<f32>(center.x, -center.y);
    out[idx] = cmul(val, inv_center);
}
