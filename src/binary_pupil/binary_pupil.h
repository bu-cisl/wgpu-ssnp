#ifndef BINARY_PUPIL_H
#define BINARY_PUPIL_H
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <optional>
#include <cmath>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"
#include "../c_gamma/c_gamma.h"

void binary_pupil(
    WebGPUContext& context,
    wgpu::Buffer& maskBuffer,
    std::optional<std::vector<float>> res = std::vector<float>{0.1, 0.1, 0.1},
    std::optional<float> na = 1.0f,
    const std::vector<int>& shape = {}
);

#endif 