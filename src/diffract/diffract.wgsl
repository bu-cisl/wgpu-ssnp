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
    let kz = 2.0 * pi * res[2] * cgamma[idx];
    let eva = exp(clamp((cgamma[idx] - 0.2) * 5.0, -1e38, 0.0));

    let cos_kz_dz = cos(kz * params) * eva;
    let sin_kz_dz = sin(kz * params) * eva;

    let p_mat_0 = cos_kz_dz;
    let p_mat_1 = sin_kz_dz / kz;
    let p_mat_2 = -sin_kz_dz * kz;
    let p_mat_3 = cos_kz_dz;

    newUF[idx] = vec2<f32>(
        p_mat_0 * uf[idx].x + p_mat_1 * ub[idx].x,
        p_mat_0 * uf[idx].y + p_mat_1 * ub[idx].y
    );

    newUB[idx] = vec2<f32>(
        p_mat_2 * uf[idx].x + p_mat_3 * ub[idx].x,
        p_mat_2 * uf[idx].y + p_mat_3 * ub[idx].y
    );
}