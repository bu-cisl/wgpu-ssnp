#ifndef C_GAMMA_H
#define C_GAMMA_H
#include "../webgpu_utils.h"
#include <vector>

void c_gamma(WebGPUContext& context, wgpu::Buffer& outputBuffer, const std::vector<float>& res, const std::vector<int>& shape);

#endif 
