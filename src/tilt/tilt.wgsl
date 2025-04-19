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
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    let num_angles = arrayLength(&angles);
    let shape_x = u32(shape[1]);
    let shape_y = u32(shape[0]);
    let total_pixels = shape_x * shape_y;
    let total_out = num_angles * total_pixels;
    let pi = radians(180.0);
    if (idx >= total_out) {
        return;
    }

    // decode (angle, y, x) from idx
    let angle_idx = idx / total_pixels;
    let pixel_idx = idx % total_pixels;
    let x = pixel_idx % shape_x;
    let y = pixel_idx / shape_x;

    // compute factors
    let angle = angles[angle_idx];
    let c_sin = NA * sin(angle);
    let c_cos = NA * cos(angle);
    let norm_sin = f32(shape[0]) * res[1];
    let norm_cos = f32(shape[1]) * res[2];
    var f_sin = c_sin * norm_sin;
    var f_cos = c_cos * norm_cos;
    if (trunc_flag == 1u) {
        f_sin = trunc(f_sin);
        f_cos = trunc(f_cos);
    }

    let two_pi = 2.0 * pi;
    // compute 
    let xr = cexp(two_pi * f_cos * f32(x) / f32(shape_x));
    let yr = cexp(two_pi * f_sin * f32(y) / f32(shape_y));
    let val = cmul(xr, yr);

    // normalization: divide by center‑point value
    let cx = shape_x / 2u;
    let cy = shape_y / 2u;
    let center_x = cexp(two_pi * f_cos * f32(cx) / f32(shape_x));
    let center_y = cexp(two_pi * f_sin * f32(cy) / f32(shape_y));
    let center = cmul(center_x, center_y);

    let inv_center = vec2<f32>(center.x, -center.y);
    let norm = cmul(val, inv_center);

    out[idx] = norm;
}
