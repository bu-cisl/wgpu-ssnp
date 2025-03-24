@group(0) @binding(0) var<storage, read> shape: array<i32>;
@group(0) @binding(1) var<storage, read> res: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const eps: f32 = 1E-8;

fn modulus(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn near_0(index: i32, size: i32) -> f32 {
    return modulus(f32(index) / f32(size) + 0.5, 1.0) - 0.5;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_dims: i32 = i32(arrayLength(&shape));

    for (var dim: i32 = 0; dim < num_dims; dim++) {
        if (i32(global_id[dim]) >= shape[dim]) {
            return;
        }
    }

    var sum_squares: f32 = 0.0;
    let alpha = near_0(i32(global_id[0]), shape[0]) / res[0];
    let beta = near_0(i32(global_id[1]), shape[1]) / res[1];

    sum_squares = alpha * alpha + beta * beta;

    let value = 1.0 - sum_squares;
    let result = sqrt(max(value, eps));

    var index: i32 = i32(global_id[0]) * shape[1] + i32(global_id[1]);

    output[index] = result;
}