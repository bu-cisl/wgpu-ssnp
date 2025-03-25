#ifndef DIFFRACT_H
#define DIFFRACT_H
#include "../webgpu_utils.h"
#include <vector>
#include <optional>

void diffract(WebGPUContext& context, wgpu::Buffer& newUFBuffer, wgpu::Buffer& newUDBuffer, std::vector<float> uf, std::vector<float> ub, std::optional<std::vector<float>> ub = {0.1, 0.1, 0.1}, std::optional<float> dz = 1);

#endif 