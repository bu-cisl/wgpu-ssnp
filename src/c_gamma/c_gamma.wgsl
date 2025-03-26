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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_dims: i32 = i32(arrayLength(&shape));
    var valid: bool = true;
    var sum_squares: f32 = 0.0;

    for (var dim: i32 = 0; dim < num_dims; dim++) {
        if (i32(global_id[dim]) >= shape[dim]) {
            valid = false;
            break;
        }
        let normalized = near_0(i32(global_id[dim]), shape[dim]) / res[dim];
        sum_squares += normalized * normalized;
    }

    if (!valid) {
        return;
    }

    let value = 1.0 - sum_squares;
    let result = sqrt(max(value, eps));

    var index: i32 = 0;
    var stride: i32 = 1;
    for (var dim: i32 = num_dims - 1; dim >= 0; dim--) {
        index += i32(global_id[dim]) * stride;
        stride *= shape[dim];
    }

    output[index] = result;
}
