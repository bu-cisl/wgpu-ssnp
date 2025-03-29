@group(0) @binding(0) var<storage, read> cgamma : array<f32>;
@group(0) @binding(1) var<storage, read_write> mask : array<u32>;
@group(0) @binding(2) var<uniform> na: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&cgamma)) {
        return;
    }

    let threshold = sqrt(1.0 - na * na);
    mask[idx] = select(0u, 1u, cgamma[idx] > threshold);
}
