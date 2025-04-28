#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
#include "tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h" 
#include "dft/dft.h"
#include "mult/mult.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>
#include <algorithm>
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

    // initialize the final result output
    vector<vector<vector<complex<float>>>> result;
    
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
            // NEED TO COMPUTE SCATTERING EFFECTS
            UD = UD2;
            u.release();
        }

        // Propagate the wave back to the focal plane
        wgpu::Buffer U2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        wgpu::Buffer UD2 = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));  
        diffract(context, U2, UD2, U, UD, buffer_len, shape, res, -1*n.size()/2);
        U.release();
        UD.release();

        // Merge the forward and backward fields from u and ∂u
        forwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        wgpu::Buffer _ = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        split_prop(context, forwardBuffer, _, U2, UD2, buffer_len, shape, res);
        _.release();
        U2.release();
        UD2.release();
        wgpu::Buffer pupilBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        binary_pupil(context, pupilBuffer, shape, na, res);
        wgpu::Buffer finalForwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        mult(context, finalForwardBuffer, forwardBuffer, pupilBuffer, buffer_len);
        forwardBuffer.release();
        pupilBuffer.release();

        wgpu::Buffer slice_result = createBuffer(context.device, backward.data(), sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        dft(context, slice_result, finalForwardBuffer, buffer_len, shape[0], shape[1], 1); // idft
        finalForwardBuffer.release();
        vector<float> flatSlice = readBack(context.device, context.queue, buffer_len * 2, slice_result);
        slice_result.release();
        vector<complex<float>> complexSlice; // convert real,imag floats to complex pairs
        for (size_t i = 0; i < flatSlice.size(); i += 2) {
            complexSlice.push_back(complex<float>(flatSlice[i], flatSlice[i + 1]));
        }
        transform(complexSlice.begin(), complexSlice.end(), complexSlice.begin(), [](complex<float> x) { return abs(x); });
        // reshape for final result
        vector<vector<complex<float>>> reshapedSlice;
        // for (int i = 0; i < shape[0]; i++) {
        //     for (int j = 0; j < shape[1]; j++) {
        //         reshapedSlice[i][j] = complexSlice[i * shape[1] + j];
        //     }
        // }
        // result.push_back(reshapedSlice);
    }
    // return result**2 if self.intensity else result

    cout << na << " " << intensity << endl;
}