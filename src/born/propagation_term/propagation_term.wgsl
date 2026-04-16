struct Params {
    dims: vec4<f32>,
    physics: vec4<f32>,
    angle: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> input_slice: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output_term: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const eps: f32 = 1e-8;

fn modulus(x: i32, y: i32) -> i32 {
    return ((x % y) + y) % y;
}

fn near_0(index: i32, size: i32) -> f32 {
    return fract(f32(index) / f32(size) + 0.5) - 0.5;
}

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output_term)) {
        return;
    }

    let height = i32(params.dims.x);
    let width = i32(params.dims.y);
    let shift_y = i32(params.dims.z);
    let shift_x = i32(params.dims.w);

    let y = i32(idx) / width;
    let x = i32(idx) % width;

    let src_y = modulus(y - shift_y, height);
    let src_x = modulus(x - shift_x, width);
    let src_idx = src_y * width + src_x;

    let res_z = params.physics.x;
    let res_y = params.physics.y;
    let res_x = params.physics.z;
    let depth = params.physics.w;
    let pi = radians(180.0);

    let c_alpha = near_0(x, width) / res_x;
    let c_beta = near_0(y, height) / res_y;
    let kz = sqrt(max(1.0 - (c_alpha * c_alpha + c_beta * c_beta), eps)) * (2.0 * pi * res_z);

    let cb = params.angle.x;
    let ca = params.angle.y;
    let kz_in = sqrt(max(1.0 - (cb * cb + ca * ca), 0.0)) * (2.0 * pi * res_z);
    let phase = depth * (kz - kz_in);

    let factor = vec2<f32>(
        -sin(phase) / (2.0 * kz),
        cos(phase) / (2.0 * kz)
    );

    let value = input_slice[src_idx];
    output_term[idx] = vec2<f32>(
        value.x * factor.x - value.y * factor.y,
        value.x * factor.y + value.y * factor.x
    );
}
