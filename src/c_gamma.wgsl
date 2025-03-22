struct Params {
    res: array<f32>,
    shape: array<i32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const eps: f32 = 1E-8;

fn near_0(index: i32, size: i32) -> f32 {
    return mod(f32(index) / f32(size) + 0.5, 1.0) - 0.5;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_dims = arrayLength(&params.shape);

    for (var dim: i32 = 0; dim < num_dims; dim++) {
        if (i32(global_id[dim]) >= params.shape[dim]) {
            return;
        }
    }

    var c_values: array<f32>;
    for (var dim: i32 = 0; dim < num_dims; dim++) {
        c_values[dim] = near_0(i32(global_id[dim]), params.shape[dim]) / params.res[dim];
    }

    var sum_squares: f32 = 0.0;
    for (var dim: i32 = 0; dim < num_dims; dim++) {
        sum_squares += pow(abs(c_values[dim]), 2.0);
    }

    let value = 1.0 - sum_squares;
    let result = sqrt(max(value, eps));

    var index: i32 = 0;
    var stride: i32 = 1;
    for (var dim: i32 = 0; dim < num_dims; dim++) {
        index += i32(global_id[dim]) * stride;
        stride *= params.shape[dim];
    }

    output[index] = result;
}