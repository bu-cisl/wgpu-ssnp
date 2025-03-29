#ifndef DIFFRACT_H
#define DIFFRACT_H
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <optional>
#include <cmath>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"
#include "../c_gamma/c_gamma.h"

void diffract(
    WebGPUContext& context, 
    wgpu::Buffer& newUFBuffer, 
    wgpu::Buffer& newUBBuffer, 
    std::vector<float> uf, 
    std::vector<float> ub, 
    std::optional<std::vector<float>> res = std::vector<float>{0.1, 0.1, 0.1}, 
    std::optional<float> dz = 1
);

#endif 