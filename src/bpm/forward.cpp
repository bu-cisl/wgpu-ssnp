#include "forward.h"

// BPM FORWARD FUNCTION
namespace bpm {
    vector<vector<vector<float>>> forward(
        WebGPUContext& context, 
        vector<vector<vector<float>>> n, 
        vector<float> res, 
        float na, 
        vector<vector<float>> angles, 
        float n0,
        int outputType
    ) {
        vector<int> shape = {int(n[0].size()), int(n[0][0].size())};

        // initialize the final result output
        vector<vector<vector<float>>> result; // angle_size x shape[0] x shape[1]

        for(vector<float> c_ba : angles) {
            // Configure input field
            size_t buffer_len = shape[0] * shape[1];
            wgpu::Buffer fieldBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            tilt(context, fieldBuffer, c_ba, shape, res);

            // Propagate the wave through RI distribution
            for(vector<vector<float>> slice : n) {
                // propagate the wave 1.0*Δz
                wgpu::Buffer fieldBuffer2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
                diffract(context, fieldBuffer2, fieldBuffer, buffer_len, shape, res, 1.0);
                fieldBuffer.release();

                // compute scattering
                scatter(context, fieldBuffer, fieldBuffer2, buffer_len, shape, res[0], 1.0, n0);
                fieldBuffer2.release();
            }

            // Propagate the wave back to the focal plane
            wgpu::Buffer fieldBuffer2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            diffract(context, fieldBuffer2, fieldBuffer, buffer_len, shape, res, -1*float(n.size())/2);
            fieldBuffer.release();
            
            // Apply binary pupil
            wgpu::Buffer pupilBuffer = createBuffer(context.device, nullptr, sizeof(int) * buffer_len, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            binary_pupil(context, pupilBuffer, shape, na, res);
            wgpu::Buffer finalForwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            mult(context, finalForwardBuffer, fieldBuffer2, pupilBuffer, buffer_len);
            fieldBuffer2.release();
            pupilBuffer.release();
            
            // Get real space of field
            wgpu::Buffer complexSlice = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            dft(context, complexSlice, finalForwardBuffer, buffer_len, shape[0], shape[1], 1); // idft
            finalForwardBuffer.release();
            
            // Complex output
            if (outputType == 2) {
                vector<float> complexData = readBack(context.device, context.queue, buffer_len * 2, complexSlice);
                complexSlice.release();
                
                // reshape for final result - 2 x H x W (real, imag)
                vector<vector<float>> realSlice(shape[0], vector<float>(shape[1], 0.0f));
                vector<vector<float>> imagSlice(shape[0], vector<float>(shape[1], 0.0f));
                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        int idx = i * shape[1] + j;
                        realSlice[i][j] = complexData[idx * 2];     // real part
                        imagSlice[i][j] = complexData[idx * 2 + 1]; // imag part
                    }
                }
                result.push_back(realSlice);
                result.push_back(imagSlice);
            } 
            
            // Default output
            else { 
                wgpu::Buffer sliceBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
                intense(context, sliceBuffer, complexSlice, buffer_len, outputType == 1);
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
        }

        return result;
    }
}