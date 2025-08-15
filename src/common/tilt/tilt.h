#ifndef TILT_H
#define TILT_H
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <optional>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void tilt(
    WebGPUContext& context,
    wgpu::Buffer& outBuffer,
    std::vector<float> c_ba,
    std::vector<int> shape,
    std::optional<std::vector<float>> res = std::vector<float>{0.1, 0.1, 0.1}, 
    std::optional<bool> trunc = true
);

#endif