#ifndef SCATTER_FACTOR_H
#define SCATTER_FACTOR_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <optional>
#include "../webgpu_utils.h"
#include <webgpu/webgpu.hpp>

void scatter_factor(WebGPUContext& context, wgpu::Buffer& outputBuffer, std::vector<float> inputData, std::optional<float> res_z = 0.1, std::optional<float> dz = 1, std::optional<float> n0 = 1);

#endif 
