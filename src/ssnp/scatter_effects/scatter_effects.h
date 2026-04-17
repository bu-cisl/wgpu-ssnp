#ifndef SCATTER_EFFECTS_H
#define SCATTER_EFFECTS_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../../common/webgpu_utils.h"
#include "../../common/fft/fft.h"
#include "../../common/complex_mult/complex_mult.h"
#include "../../common/complex_sub/complex_sub.h"

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
