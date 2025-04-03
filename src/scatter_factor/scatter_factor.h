#ifndef SCATTER_FACTOR_H
#define SCATTER_FACTOR_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <optional>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void scatter_factor(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    std::vector<float> n, 
    std::optional<float> res_z = 0.1, 
    std::optional<float> dz = 1, 
    std::optional<float> n0 = 1
);

#endif 
