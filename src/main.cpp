#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
#include "tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h"
#include "c_gamma/c_gamma.h"  
#include "webgpu_utils.h"
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    initWebGPU(context);

    // default init params for ssnp model
    vector<float> res = {0.1,0.1,0.1};
    float na = 0.65;
    int angles_size = 32;
    bool intensity = true;

    // angles_size vectors of c_ba values
    vector<vector<float>> angles(angles_size, vector<float>(2, 0.0));

    // input matrix
    vector<vector<vector<float>>> n(3, vector<vector<float>>(16, vector<float>(16, 1.0f)));

    // ssnp forward function
    vector<int> shape = {int(n[0].size()), int(n[0][0].size())};
    
    for(vector<float> c_ba : angles) {
        // CONFIGURING INPUT FIELD

        // Generate Forward/Backward
        size_t buffer_len = shape[0] * shape[1];
        wgpu::Buffer tiltResultBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        tilt(context, tiltResultBuffer, c_ba, shape, res);
        wgpu::Buffer forwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        // NEED TO forward = fft(tiltResult)
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

            // Field to spatial domain
            // NEED TO u = ifft(U)

            // Scattering effects
            // NEED TO COMPUTE SCATTERING EFFECTS

            // Reassign U/UD values
            U = U2;
            UD = UD2;
        }

        // Propagate the wave back to the focal plane
        wgpu::Buffer U2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        wgpu::Buffer UD2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));  
        diffract(context, U2, UD2, U, UD, buffer_len, shape, res, -1*n.size()/2);
        U.release();
        UD.release();

        // Merge the forward and backward fields from u and ∂u
        forwardBuffer = createBuffer(context.device, backward.data(), sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        wgpu::Buffer _ = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        split_prop(context, forwardBuffer, _, U2, UD2, buffer_len, shape, res);
        // NEED TO FORWARD * BINARY PUPIL

        // NEED TO temp_result = torch.fft.ifft2(Forward)
        // read back temp_result
        // temp_result = abs(temp_result)
        // append to result 3d Matrix
    }
    // return result**2 if self.intensity else result

    cout << na << " " << intensity << endl;
}