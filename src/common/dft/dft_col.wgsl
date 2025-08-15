@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dims: vec2<i32>; // rows, cols
@group(0) @binding(3) var<uniform> doInverse: u32; // IDFT flag

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = i32(global_id.x); 
    let l = i32(global_id.y);   // freq index
    if (col >= dims.y || l >= dims.x) {
        return;
    }
    
    var sum = vec2<f32>(0.0, 0.0);
    let pi = radians(180.0);
    
    // Determine sign based on inverse flag
    let sign = select(-2.0, 2.0, doInverse == 1);

    // Compute DFT/IDFT
    for (var row = 0; row < dims.x; row = row + 1) {
        let phase = fract(f32(row * l) / f32(dims.x)); // shrink phase to preserve precision
        let angle = sign * pi * phase;
        let euler = vec2<f32>(cos(angle), sin(angle));
        let idx = row * dims.y + col;
        let val = input[idx];

        // Euler Rule
        sum = sum + vec2<f32>(
            val.x * euler.x - val.y * euler.y,
            val.x * euler.y + val.y * euler.x
        );
    }
    
    // For IDFT, normalize by N (dims.x in this case)
    if (doInverse == 1) {
        sum = sum / f32(dims.x);
    }
    
    let outIndex = l * dims.y + col;
    output[outIndex] = sum;
}