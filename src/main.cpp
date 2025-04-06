#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
#include "tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h"
#include "webgpu_utils.h"
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    initWebGPU(context);

    // Test scatter_factor
    vector<float> scatter_input = {5, 21, 65};
    wgpu::Buffer scatterResultBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * scatter_input.size(), 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    scatter_factor(context, scatterResultBuffer, scatter_input);
    cout << "scatter_factor output: " << endl;
    vector<float> scatter = readBack(context.device, context.queue, scatter_input.size(), scatterResultBuffer);
    for (float s : scatter) cout << fixed << setprecision(8) << s << " ";
    cout << endl;

    // Test c_gamma
    vector<int> c_gamma_shape = {3, 3};
    vector<float> c_gamma_res = {0.1f, 0.4f, 0.1f};
    size_t c_gamma_len = c_gamma_shape[0] * c_gamma_shape[1];
    wgpu::Buffer cGammaResultBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * c_gamma_len, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    c_gamma(context, cGammaResultBuffer, c_gamma_res, c_gamma_shape);
    cout << "c_gamma output:" << endl;
    vector<float> cgamma = readBack(context.device, context.queue, c_gamma_len, cGammaResultBuffer);
    for (float c : cgamma) cout << fixed << setprecision(4) << c << " ";
    cout << endl;

    // Test diffract
    vector<float> diffract_uf = {1,2,3,4,5,6,7,8,9};
    vector<float> diffract_ub = {9,8,7,6,5,4,3,2,1};
    size_t diffract_size = diffract_uf.size();
    vector<int> diffract_shape = {3,3}; // we need to note original shape of matrix before flattening
    wgpu::Buffer diffractUFBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * diffract_size, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer diffractUBBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * diffract_size, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    diffract(context, diffractUFBuffer, diffractUBBuffer, diffract_uf, diffract_ub, diffract_shape);
    cout << "diffract output (new uf):" << endl;
    vector<float> diffractUF = readBack(context.device, context.queue, diffract_size, diffractUFBuffer);
    for (float uf : diffractUF) cout << fixed << setprecision(4) << uf << " ";
    cout << endl;
    cout << "diffract output (new ub):" << endl;
    vector<float> diffractUB = readBack(context.device, context.queue, diffract_size, diffractUBBuffer);
    for (float ub : diffractUB) cout << fixed << setprecision(4) << ub << " ";
    cout << endl;

    // Test binary_pupil
    vector<int> binary_pupil_shape = {3, 3};
    float binary_pupil_na = 0.9f;
    vector<float> binary_pupil_res = {0.1f, 0.4f, 0.1f};
    size_t binary_pupil_len = c_gamma_shape[0] * c_gamma_shape[1];
    wgpu::Buffer binaryPupilResultBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(uint32_t) * binary_pupil_len, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    binary_pupil(context, binaryPupilResultBuffer, binary_pupil_shape, binary_pupil_na, binary_pupil_res);
    cout << "binary_pupil output:" << endl;
    vector<uint32_t> binarypupil = readBackInt(context.device, context.queue, binary_pupil_len, binaryPupilResultBuffer);
    for (uint32_t b : binarypupil) cout << b << " ";
    cout << endl;

    // Test tilt
    vector<int> tilt_shape = {8, 8};
    vector<float> tilt_angles = {2*M_PI, M_PI/2, M_PI/6};
    float tilt_NA = 0.5f;
    vector<float> tilt_res = {0.69f, 0.2f, 0.1f};
    bool trunc = false;
    size_t tilt_output_size = tilt_angles.size() * 2;  
    wgpu::Buffer tiltResultBuffer = createBuffer(
        context.device, nullptr, 
        sizeof(float) * tilt_output_size, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    tilt(context, tiltResultBuffer, tilt_angles, tilt_shape, tilt_NA, tilt_res, trunc);
    cout << "tilt (factor) output:" << endl;
    vector<float> tilt = readBack(context.device, context.queue, tilt_output_size, tiltResultBuffer);
    for (float t : tilt) cout << fixed << scientific << setprecision(4) << t << " ";
    cout << endl;

    // Test merge_prop
    vector<float> merge_res = {0.1f, 0.1f, 0.1f};
    vector<complex<float>> uf_merge = {
        {1.0f, 9.0f}, {2.0f, 8.0f}, {3.0f, 7.0f},
        {4.0f, 6.0f}, {5.0f, 5.0f}, {6.0f, 4.0f},
        {7.0f, 3.0f}, {8.0f, 2.0f}, {9.0f, 1.0f}
    };
    vector<complex<float>> ub_merge = {
        {9.0f, 1.0f}, {8.0f, 2.0f}, {7.0f, 3.0f},
        {6.0f, 4.0f}, {5.0f, 5.0f}, {4.0f, 6.0f},
        {3.0f, 7.0f}, {2.0f, 8.0f}, {1.0f, 9.0f}
    };
    vector<int> merge_shape = {3,3};
    wgpu::Buffer mergeUFBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * uf_merge.size(), // ×2 for complex
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer mergeUBBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * ub_merge.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    merge_prop(context, mergeUFBuffer, mergeUBBuffer, uf_merge, ub_merge, merge_shape, merge_res);
    cout << "merge_prop output (new uf):" << endl;
    vector<float> mergeUF = readBack(context.device, context.queue, 2 * uf_merge.size(), mergeUFBuffer);
    for (float uf : mergeUF) cout << fixed << setprecision(4) << uf << " ";
    cout << endl;
    cout << "merge_prop output (new ub):" << endl;
    vector<float> mergeUB = readBack(context.device, context.queue, 2 * ub_merge.size(), mergeUBBuffer);
    for (float ub : mergeUB) cout << fixed << setprecision(4) << ub << " ";
    cout << endl;

    // Test split_prop
    vector<float> split_res = {0.1f, 0.1f, 0.1f};
    vector<complex<float>> uf_split = {
        {1.0f, 9.0f}, {2.0f, 8.0f}, {3.0f, 7.0f},
        {4.0f, 6.0f}, {5.0f, 5.0f}, {6.0f, 4.0f},
        {7.0f, 3.0f}, {8.0f, 2.0f}, {9.0f, 1.0f}
    };
    vector<complex<float>> ub_split = {
        {9.0f, 1.0f}, {8.0f, 2.0f}, {7.0f, 3.0f},
        {6.0f, 4.0f}, {5.0f, 5.0f}, {4.0f, 6.0f},
        {3.0f, 7.0f}, {2.0f, 8.0f}, {1.0f, 9.0f}
    };
    vector<int> split_shape = {3,3};
    wgpu::Buffer splitUFBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * uf_merge.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer splitUBBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * ub_merge.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    split_prop(context, splitUFBuffer, splitUBBuffer, uf_split, ub_split, split_shape, split_res);
    cout << "split_prop output (new uf):" << endl;
    vector<float> splitUF = readBack(context.device, context.queue, 2 * uf_merge.size(), splitUFBuffer);
    for (float uf : splitUF) cout << fixed << scientific << setprecision(4) << uf << " ";
    cout << endl;
    cout << "split_prop output (new ub):" << endl;
    vector<float> splitUB = readBack(context.device, context.queue, 2 * ub_merge.size(), splitUBBuffer);
    for (float ub : splitUB) cout << fixed << scientific <<  setprecision(4) << ub << " ";
    cout << endl;

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    scatterResultBuffer.release();
    cGammaResultBuffer.release();
    diffractUFBuffer.release();
    diffractUBBuffer.release();
    binaryPupilResultBuffer.release();
    tiltResultBuffer.release();
    mergeUFBuffer.release();
    mergeUBBuffer.release();
    splitUFBuffer.release();
    splitUBBuffer.release();

    return 0;
}