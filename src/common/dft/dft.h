#ifndef DFT_H
#define DFT_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <complex>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void dft(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    uint32_t doInverse
);

void dft_adjoint_forward(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols
);

void dft_adjoint_inverse(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols
);

#endif 
