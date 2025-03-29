#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
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
    wgpu::Buffer scatterFactorResultBuffer = createBuffer(context.device, nullptr, input.size() * sizeof(float), WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    scatter_factor(context, scatterFactorResultBuffer, input);
    cout << "scatter_factor output: " << endl;
    vector<float> scatter = readBack(context.device, context.queue, input.size(), scatterFactorResultBuffer);
    for (float s : scatter) cout << fixed << setprecision(8) << s << " ";
    cout << endl;

    // Test c_gamma


    // Test diffract
    vector<float> uf = {1,2,3,4,5,6,7,8,9};
    vector<float> ub = {9,8,7,6,5,4,3,2,1};
    wgpu::Buffer newUFBuffer = createBuffer(context.device, nullptr, sizeof(float) * uf.size(), WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    wgpu::Buffer newUBBuffer = createBuffer(context.device, nullptr, sizeof(float) * ub.size(), WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    diffract(context, newUFBuffer, newUBBuffer, uf, ub);
    cout << "diffract output (new uf):" << endl;
    vector<float> ufbuff = readBack(context.device, context.queue, uf.size(), newUFBuffer);
    for (float uf : ufbuff) cout << fixed << setprecision(4) << uf << " ";
    cout << endl;
    cout << "diffract output (new ub):" << endl;
    vector<float> ubbuff = readBack(context.device, context.queue, ub.size(), newUBBuffer);
    for (float ub : ubbuff) cout << fixed << setprecision(4) << ub << " ";
    cout << endl;

    // Test binary_pupil
    vector<int> shape = {4, 4};
    wgpu::Buffer maskBuffer = createBuffer(context.device, nullptr, sizeof(uint32_t) * shape[0] * shape[1], WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    std::vector<float> res = {1.1f, 0.4f, 0.1f};
    float na = 0.9f;
    binary_pupil(context, maskBuffer, res, na, shape);
    cout << "binary_pupil output: ";
    vector<uint32_t> maskBuff = readBack2(context.device, context.queue, shape[0] * shape[1], maskBuffer);
    for (uint32_t val : maskBuff) cout << val << " ";
    cout << endl;

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    scatterFactorResultBuffer.release();
    newUFBuffer.release();
    newUBBuffer.release();
    maskBuffer.release();

    return 0;
}