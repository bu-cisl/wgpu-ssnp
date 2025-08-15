#include "scatter_effects.h"

static size_t buffer_len;

void scatter_effects(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& scatterBuffer, 
    wgpu::Buffer& uBuffer, 
    wgpu::Buffer& udBuffer,
    size_t bufferlen,
    std::vector<int> shape
) {
    buffer_len = bufferlen;

    // perform scatter factor * u
    wgpu::Buffer fftInputBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    complex_mult(context, fftInputBuffer, scatterBuffer, uBuffer, buffer_len);

    // perform fft(scatter*u)
    wgpu::Buffer fftBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    dft(context, fftBuffer, fftInputBuffer, buffer_len, shape[0], shape[1], 0);
    fftInputBuffer.release();

    // perform ud - fft(scatter*u)
    complex_sub(context, outputBuffer, udBuffer, fftBuffer, buffer_len);

    // Cleanup Resources
    fftBuffer.release();
}