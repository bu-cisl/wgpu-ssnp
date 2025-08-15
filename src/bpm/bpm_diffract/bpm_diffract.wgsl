@group(0) @binding(0) var<storage, read> output : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> input : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> res : array<f32>;
@group(0) @binding(3) var<storage, read> cgamma : array<f32>;
@group(0) @binding(6) var<uniform> params: f32; // dz

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&uf)) {
        return;
    }

    let pi = radians(180.0);
    let gamma = cgamma[idx];
    let kz = 2.0 * pi * res[2] * gamma;

    // Clamp exponent to prevent underflow/overflow
    let exponent = clamp((gamma - 0.2) * 5.0, -60.0, 0.0);
    let eva = exp(exponent);

    // θ = kz * eva * dz
    let theta = kz * eva * params;

    // Numerically stable sin/cos for small |θ|
    let small = abs(theta) < 1e-3;
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;

    // For small θ: sinθ ≈ θ - θ^3/6, cosθ ≈ 1 - θ^2/2
    let sin_theta = select(sin(theta), theta - (theta3 * (1.0 / 6.0)), small);
    let cos_theta = select(cos(theta), 1.0 - 0.5 * theta2, small);

    // field * (cosθ + i sinθ)
    let a = input[idx].x; // Re(field)
    let b = input[idx].y; // Im(field)
    let out_real = a * cos_theta - b * sin_theta;
    let out_imag = a * sin_theta + b * cos_theta;

    output[idx] = vec2<f32>(out_real, out_imag);
}