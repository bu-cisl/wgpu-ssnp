#ifndef BINARY_PUPIL_H
#define BINARY_PUPIL_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <optional>
#include <webgpu/webgpu.hpp>
#include "../../webgpu_utils.h"
#include "../c_gamma/c_gamma.h"

void binary_pupil(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    std::vector<int> shape,
    float na,
    std::optional<std::vector<float>> res = std::vector<float>{0.1, 0.1, 0.1}
);

#endif 