@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: vec3<i32>; // x=rows, y=cols, z=stage
@group(0) @binding(2) var<uniform> doInverse: u32;

// Bit-reverse permutation for rows
@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = i32(global_id.x);
    let row = i32(global_id.y);
    let cols = params.y;
    let rows = params.x;

    if (col >= cols || row >= rows) {
        return;
    }

    // Get bit-reversed index for this column
    let n = u32(cols);
    var reversed = 0u;
    var col_bit = u32(col);
    var temp = n;
    
    while (temp > 1u) {
        reversed = (reversed << 1u) | (col_bit & 1u);
        col_bit = col_bit >> 1u;
        temp = temp >> 1u;
    }

    // Only swap if reversed > col to avoid double swaps
    if (i32(reversed) > col) {
        let idx1 = row * cols + col;
        let idx2 = row * cols + i32(reversed);
        
        let temp_val = data[idx1];
        data[idx1] = data[idx2];
        data[idx2] = temp_val;
    }
}
