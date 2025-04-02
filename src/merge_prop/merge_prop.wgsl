@group(0) @binding(0) var<storage, read> uf : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> ub : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> res : array<f32>;
@group(0) @binding(3) var<storage, read> cgamma : array<f32>;
@group(0) @binding(4) var<storage, read_write> uf_new : array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> ub_new : array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&uf)) {
        return;
    }

    let pi = radians(180.0); 
    let kz = cgamma[idx] * (2.0 * pi * res[2]);

    // Complex addition: uf_new = uf + ub
    uf_new[idx] = uf[idx] + ub[idx];
    
    // Complex multiplication: ub_new = (uf - ub) * 1j * kz
    let uf_minus_ub = uf[idx] - ub[idx];
    let imaginary_kz = vec2<f32>(-uf_minus_ub.y * kz, uf_minus_ub.x * kz);
    ub_new[idx] = imaginary_kz;
}