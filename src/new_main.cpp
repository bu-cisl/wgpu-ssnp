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
        wgpu::Buffer tiltResultBuffer = createBuffer(context.device, nullptr, sizeof(float) * shape[0] * shape[1] * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        tilt(context, tiltResultBuffer, c_ba, shape, res);
        vector<float> backward(shape[0]*shape[1]*2, 0.0);
        wgpu::Buffer backwardBuffer = createBuffer(context.device, backward.data(), sizeof(float) * shape[0] * shape[1] * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        backwardBuffer = backwardBuffer;

        continue;
    }

    cout << na << " " << intensity << endl;
}