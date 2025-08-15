@group(0) @binding(0) var<storage, read> uf : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> ub : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> res : array<f32>;
@group(0) @binding(3) var<storage, read> cgamma : array<f32>;
@group(0) @binding(4) var<storage, read_write> newUF : array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> newUB : array<vec2<f32>>;
@group(0) @binding(6) var<uniform> params: f32;

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

    // Trig stability: use Taylor approximation for small kz
    let kz_dz = kz * params;

    let cos_kz_dz = cos(kz_dz) * eva;

    let sin_kz_dz =
        select(sin(kz_dz), kz_dz - (kz_dz * kz_dz * kz_dz) / 6.0, abs(kz_dz) < 1e-2) * eva;

    let p_mat_0 = cos_kz_dz;
    let p_mat_1 = sin_kz_dz / max(kz, 1e-6); // avoid divide-by-zero
    let p_mat_2 = -sin_kz_dz * kz;
    let p_mat_3 = cos_kz_dz;

    // FMA-style pattern: helps minimize rounding error
    let uf_x = uf[idx].x;
    let uf_y = uf[idx].y;
    let ub_x = ub[idx].x;
    let ub_y = ub[idx].y;

    newUF[idx] = vec2<f32>(
        p_mat_0 * uf_x + p_mat_1 * ub_x,
        p_mat_0 * uf_y + p_mat_1 * ub_y
    );

    newUB[idx] = vec2<f32>(
        p_mat_2 * uf_x + p_mat_3 * ub_x,
        p_mat_2 * uf_y + p_mat_3 * ub_y
    );
}
