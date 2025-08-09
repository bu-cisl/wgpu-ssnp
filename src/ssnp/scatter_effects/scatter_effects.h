#ifndef SCATTER_EFFECTS_H
#define SCATTER_EFFECTS_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"
#include "../dft/dft.h"

void scatter_effects(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& scatterBuffer, 
    wgpu::Buffer& uBuffer, 
    wgpu::Buffer& udBuffer,
    size_t bufferlen,
    std::vector<int> shape
);

#endif 
