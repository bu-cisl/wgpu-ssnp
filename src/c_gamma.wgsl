struct Params {
    res: vec3<f32>,
    shape: vec3<i32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const eps: f32 = 1E-8;

fn near_0(index: i32, size: i32) -> f32 {
    return mod(f32(index) / f32(size) + 0.5, 1.0) - 0.5;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let z = i32(global_id.z);

    if (x >= params.shape.x || y >= params.shape.y || z >= params.shape.z) {
        return;
    }

    let c_alpha = near_0(y, params.shape.y) / params.res.x;
    let c_beta  = near_0(x, params.shape.x) / params.res.y;
    let c_gamma = near_0(z, params.shape.z) / params.res.z;

    let alpha_square = pow(abs(c_alpha), 2.0);
    let beta_square  = pow(abs(c_beta), 2.0);
    let gamma_square = pow(abs(c_gamma), 2.0);

    let value = 1.0 - (alpha_square + beta_square + gamma_square);
    let result = sqrt(max(value, eps));

    let index = (z * params.shape.y * params.shape.x) + (y * params.shape.x) + x;
    output[index] = result;
}