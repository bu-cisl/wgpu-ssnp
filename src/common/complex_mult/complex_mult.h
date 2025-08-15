#ifndef COMPLEX_MULT_H
#define COMPLEX_MULT_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void complex_mult(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer1, 
    wgpu::Buffer& inputBuffer2,
    size_t bufferlen
);

#endif 
