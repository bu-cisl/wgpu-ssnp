#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
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
    wgpu::Buffer scatterFactorResultBuffer = createBuffer(context.device, nullptr, input.size() * sizeof(float), WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    scatter_factor(context, scatterFactorResultBuffer, input);
    cout << "scatter_factor output: " << endl;
    vector<float> scatter = readBack(context.device, context.queue, input.size(), scatterFactorResultBuffer);
    for (float s : scatter) cout << fixed << setprecision(8) << s << " ";
    cout << endl;

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

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    scatterFactorResultBuffer.release();
    newUFBuffer.release();
    newUBBuffer.release();

    return 0;
}