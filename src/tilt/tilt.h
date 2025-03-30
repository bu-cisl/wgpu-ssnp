#ifndef TILT_H
#define TILT_H

#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <optional>
#include <cmath>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void tilt(
    WebGPUContext& context,
    wgpu::Buffer& factorsBuffer,
    const std::vector<float>& angles,      // Flattened (N, 2) tensor
    const std::vector<uint32_t>& shape,    // Always 2D: [height, width]
    float NA,
    const std::vector<float>& res,
    bool trunc = false,
    uint32_t numAnglePairs = 0
);

#endif