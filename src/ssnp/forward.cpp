#include "forward.h"

namespace ssnp {
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
        size_t buffer_len = shape[0] * shape[1];
        vector<vector<vector<float>>> result;

        // TRAVERSING EACH ILLUMINATION ANGLE
        for (const vector<float>& c_ba : angles) {
            // PROPAGATING THROUGH THE VOLUME
            SSNPState exitState = propagate_to_object_exit(
                context,
                initialize_angle_state(context, c_ba, shape, res),
                n,
                shape,
                res,
                n0
            );
            // PROJECTING TO THE SENSOR PLANE
            wgpu::Buffer complexSlice = project_state_to_sensor_field(
                context,
                exitState,
                shape,
                res,
                na,
                -1.0f * float(n.size()) / 2.0f
            );
            release_state(exitState);
            
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
