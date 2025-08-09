#ifndef C_GAMMA_H
#define C_GAMMA_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void c_gamma(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    std::vector<float> res, 
    std::vector<int> shape
);

#endif 
