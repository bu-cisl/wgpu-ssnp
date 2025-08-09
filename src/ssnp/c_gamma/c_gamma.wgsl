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

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index: i32 = i32(global_id.x); // Use only the x-dimension for indexing

    let size_x: i32 = shape[0];
    let size_y: i32 = shape[1];
    let total_size: i32 = size_x * size_y;

    if (index >= total_size) {
        return;
    }

    let resolution_x: f32 = res[1];
    let resolution_y: f32 = res[2];

    let index_x: i32 = index / size_y;
    let index_y: i32 = index % size_y;


    let c_alpha: f32 = near_0(index_y, size_y) / resolution_y;
    let c_beta: f32 = near_0(index_x, size_x) / resolution_x;

    let alpha_square: f32 = c_alpha * c_alpha;
    let beta_square: f32 = c_beta * c_beta;

    let gamma: f32 = sqrt(max(1.0 - (alpha_square + beta_square), eps));

    output[index] = gamma;
}
