@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: vec3<i32>; // x=rows, y=cols, z=stage
@group(0) @binding(2) var<uniform> doInverse: u32;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = i32(global_id.x);
    let row = i32(global_id.y);
    let cols = params.y;
    let rows = params.x;
    let stage = u32(params.z);
    
    if (col >= cols || row >= rows) {
        return;
    }

    // Cooley-Tukey butterfly operation for row FFT
    let m_u32 = 1u << (stage + 1u);
    let half_m_u32 = 1u << stage;
    let m = i32(m_u32);
    let half_m = i32(half_m_u32);
    
    let first_idx = (col / half_m) * m;
    let offset = col % half_m;
    let idx1 = first_idx + offset;
    let idx2 = first_idx + offset + half_m;
    
    if (idx2 >= cols) {
        return;
    }

    let pi = radians(180.0);
    let k = f32(offset);
    
    // Compute twiddle factor: exp(-2πi * k / m) for forward, exp(2πi * k / m) for inverse
    let sign = select(-1.0, 1.0, doInverse == 1u);
    let angle = sign * 2.0 * pi * k / f32(m);
    let w_real = cos(angle);
    let w_imag = sin(angle);
    
    // Get data values
    let a = data[row * cols + idx1];
    let b = data[row * cols + idx2];
    
    // Compute b * w
    let b_w = vec2<f32>(
        b.x * w_real - b.y * w_imag,
        b.x * w_imag + b.y * w_real
    );
    
    // Butterfly: t = a + b*w, b_new = a - b*w
    data[row * cols + idx1] = a + b_w;
    data[row * cols + idx2] = a - b_w;
    
    // For inverse FFT, divide by 2 at each stage per element
    if (doInverse == 1u) {
        data[row * cols + idx1] = data[row * cols + idx1] * 0.5;
        data[row * cols + idx2] = data[row * cols + idx2] * 0.5;
    }
}
