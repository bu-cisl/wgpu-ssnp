#ifndef FFT_H
#define FFT_H

#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"
#include "fft_utils.h"

// Barebones API entry point allowing forced DFT
void fft(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    uint32_t doInverse,
    bool forceDft = false
);

// Internal Cooley-Tukey implementation for power-of-2 dimensions.
void fftPowerOfTwo(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    uint32_t doInverse
);

void fft_adjoint_forward(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    bool forceDft = false
);

void fft_adjoint_inverse(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    bool forceDft = false
);

#endif // FFT_H
