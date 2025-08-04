#include "forward.h"

// SSNP FORWARD FUNCTION
vector<vector<vector<float>>> forward(
    WebGPUContext& context, 
    vector<vector<vector<float>>> n, 
    vector<float> res, 
    float na, 
    vector<vector<float>> angles, 
    float n0,
    bool intensity
) {
    vector<int> shape = {int(n[0].size()), int(n[0][0].size())};

    // initialize the final result output
    vector<vector<vector<float>>> result; // angle_size x shape[0] x shape[1]

    for(vector<float> c_ba : angles) {
        // CONFIGURING INPUT FIELD
        // Generate Forward/Backward
        size_t buffer_len = shape[0] * shape[1];
        wgpu::Buffer tiltResultBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        tilt(context, tiltResultBuffer, c_ba, shape, res);
        wgpu::Buffer forwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        dft(context, forwardBuffer, tiltResultBuffer, buffer_len, shape[0], shape[1], 0); // dft
        vector<float> backward(shape[0]*shape[1]*2, 0.0);
        wgpu::Buffer backwardBuffer = createBuffer(context.device, backward.data(), sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        tiltResultBuffer.release();
        
        // Get U/UD
        wgpu::Buffer U = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        wgpu::Buffer UD = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        merge_prop(context, U, UD, forwardBuffer, backwardBuffer, buffer_len, shape, res);
        forwardBuffer.release();
        backwardBuffer.release();

        // Traverse slices
        for(vector<vector<float>> slice : n) {
            // Propogate the wave
            wgpu::Buffer U2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            wgpu::Buffer UD2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            diffract(context, U2, UD2, U, UD, buffer_len, shape, res, 1.0);
            U.release();
            UD.release();
            U = U2; // reassign U value for next iter

            // Field to spatial domain
            wgpu::Buffer u = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            dft(context, u, U2, buffer_len, shape[0], shape[1], 1); // idft

            // Scattering effects
            UD = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            wgpu::Buffer scatterBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            vector<float> flatSlice;
            for (const auto& inner : slice) {
                flatSlice.insert(flatSlice.end(), inner.begin(), inner.end());
            }
            vector<float> complexSlice;
            for (float value : flatSlice) {
                complexSlice.push_back(value); // real part
                complexSlice.push_back(0); // 0 for imag part
            }        
            wgpu::Buffer sliceBuffer = createBuffer(context.device, complexSlice.data(), sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            scatter_factor(context, scatterBuffer, sliceBuffer, buffer_len, res[0], 1, n0); // compute scatter factor output
            scatter_effects(context, UD, scatterBuffer, u, UD2, buffer_len, shape); // compute ud - fft(scatter*u)
            u.release();
            scatterBuffer.release();
            sliceBuffer.release();
            UD2.release();
        }

        // Propagate the wave back to the focal plane
        wgpu::Buffer U2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        wgpu::Buffer UD2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));  
        diffract(context, U2, UD2, U, UD, buffer_len, shape, res, -1*float(n.size())/2);
        U.release();
        UD.release();

        // Merge the forward and backward fields from u and ∂u
        forwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        wgpu::Buffer _ = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        split_prop(context, forwardBuffer, _, U2, UD2, buffer_len, shape, res);
        _.release();
        U2.release();
        UD2.release();
        wgpu::Buffer pupilBuffer = createBuffer(context.device, nullptr, sizeof(int) * buffer_len, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        binary_pupil(context, pupilBuffer, shape, na, res);
        wgpu::Buffer finalForwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        mult(context, finalForwardBuffer, forwardBuffer, pupilBuffer, buffer_len);
        forwardBuffer.release();
        pupilBuffer.release();

        wgpu::Buffer complexSlice = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        dft(context, complexSlice, finalForwardBuffer, buffer_len, shape[0], shape[1], 1); // idft
        finalForwardBuffer.release();
        wgpu::Buffer sliceBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        intense(context, sliceBuffer, complexSlice, buffer_len, intensity);
        vector<float> slice = readBack(context.device, context.queue, buffer_len, sliceBuffer);
        complexSlice.release();
        sliceBuffer.release();

        // reshape for final result
        vector<vector<float>> reshapedSlice(shape[0], vector<float>(shape[1], 0.0f));
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                reshapedSlice[i][j] = slice[i * shape[1] + j];
            }
        }
        result.push_back(reshapedSlice);
    }

    return result;
}