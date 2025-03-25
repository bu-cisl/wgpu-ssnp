@group(0) @binding(0) var<storage, read> uf : array<f32>;
@group(0) @binding(1) var<storage, read> ub : array<f32>;
@group(0) @binding(2) var<storage, read> cgamma : array<f32>;
@group(0) @binding(3) var<storage, read_write> newUF : array<f32>;
@group(0) @binding(4) var<storage, read_write> newUB : array<f32>;
@group(0) @binding(5) var<uniform> params: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;

    // Ensure we don't exceed array bounds
    if (idx >= arrayLength(&uf)) {
        return;
    }

    let kz = 2.0 * 3.14159265359 * 0.1 * cgamma[idx]; // res[2] = 0.1
    let eva = exp(clamp((cgamma[idx] - 0.2) * 5.0, 0.0, 1e10));
    
    let cos_kz_dz = cos(kz * params);
    let sin_kz_dz = sin(kz * params);

    // Constructing p_mat
    let p_mat_0 = cos_kz_dz * eva;
    let p_mat_1 = (sin_kz_dz / kz) * eva;
    let p_mat_2 = (-sin_kz_dz * kz) * eva;
    let p_mat_3 = cos_kz_dz * eva;

    // Calculating uf_new and ub_new
    newUF[idx] = p_mat_0 * uf[idx] + p_mat_1 * ub[idx];
    newUB[idx] = p_mat_2 * uf[idx] + p_mat_3 * ub[idx];
}
