#ifndef C_GAMMA_H
#define C_GAMMA_H
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void c_gamma(WebGPUContext& context, wgpu::Buffer& outputBuffer, const std::vector<float>& res, const std::vector<int>& shape);

#endif 
