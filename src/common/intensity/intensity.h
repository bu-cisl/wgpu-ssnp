#ifndef INTENSITY_H
#define INTENSITY_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void intense(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    bool intensity
);

#endif 
