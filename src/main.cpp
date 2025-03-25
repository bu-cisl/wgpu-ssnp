#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "c_gamma/c_gamma.h"
#include "diffract/diffract.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    initWebGPU(context);

    // Test scatter_factor
    vector<float> input = {5, 21, 65};
    wgpu::Buffer scatterFactorResultBuffer = createBuffer(context.device, nullptr, input.size() * sizeof(float), static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    scatter_factor(context, scatterFactorResultBuffer, input);
    cout << "scatter_factor output: " << endl;
    vector<float> output = readBack(context.device, context.queue, input.size(), scatterFactorResultBuffer);
    for (float o : output) cout << fixed << setprecision(8) << o << " ";
    cout << endl;

    // Test c_gamma
    vector<float> res = {5.2f, 2.2f};
    vector<int> shape = {3, 2};
    wgpu::Buffer cgammaResultBuffer = createBuffer(context.device, nullptr, sizeof(float) * shape[0]*shape[1], static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    c_gamma(context, cgammaResultBuffer, res, shape);
    cout << "c_gamma output:" << endl;
    vector<float> cgamma = readBack(context.device, context.queue, shape[0]*shape[1], cgammaResultBuffer);
    for (float c : cgamma) cout << fixed << setprecision(4) << c << " ";
    cout << endl;

    // Test diffract
    vector<float> uf = {1};
    vector<float> ub = {1};
    wgpu::Buffer newUFBuffer = createBuffer(context.device, nullptr, sizeof(float) * uf.size(), static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    wgpu::Buffer newUBBuffer = createBuffer(context.device, nullptr, sizeof(float) * ub.size(), static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    diffract(context, newUFBuffer, newUBBuffer, uf, ub);

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    scatterFactorResultBuffer.release();
    cgammaResultBuffer.release();

    return 0;
}