@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dims: vec2<i32>; // rows, cols
@group(0) @binding(3) var<uniform> doInverse: u32; // IDFT flag

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = i32(global_id.x); 
    let row = i32(global_id.y); 
    if (col >= dims.y || row >= dims.x) {
        return;
    }
    
    var sum = vec2<f32>(0.0, 0.0);
    let pi = radians(180.0);
    
    // Determine sign based on inverse flag
    let sign = select(-2.0, 2.0, doInverse == 1);
    
    // Compute DFT/IDFT
    for (var x = 0; x < dims.y; x = x + 1) {
        let phase = fract(f32(col * x) / f32(dims.y)); // shrink phase to preserve precision
        let angle = sign * pi * phase;
        let euler = vec2<f32>(cos(angle), sin(angle));
        let idx = row * dims.y + x;
        let val = input[idx];

        // Euler rule
        sum = sum + vec2<f32>(
            val.x * euler.x - val.y * euler.y,
            val.x * euler.y + val.y * euler.x
        );
    }
    
    // For IDFT, we need to divide by N (dims.y in this case)
    if (doInverse == 1) {
        sum = sum / f32(dims.y);
    }
    
    let outIndex = row * dims.y + col;
    output[outIndex] = sum;
}