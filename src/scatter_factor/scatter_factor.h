#ifndef SCATTER_FACTOR_H
#define SCATTER_FACTOR_H
#include "../webgpu_utils.h"
#include <vector>
#include <optional>

void scatter_factor(WebGPUContext& context, wgpu::Buffer& outputBuffer, std::vector<float> inputData, std::optional<float> res_z = 0.1, std::optional<float> dz = 1, std::optional<float> n0 = 1);

#endif 
