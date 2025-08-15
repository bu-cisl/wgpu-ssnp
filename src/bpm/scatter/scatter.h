#ifndef SCATTER_H
#define SCATTER_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <optional>
#include <webgpu/webgpu.hpp>
#include "../../common/webgpu_utils.h"
#include "../../common/dft/dft.h"
#include "../../common/complex_mult/complex_mult.h"

void scatter(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<float> res_z = 0.1, 
    std::optional<float> dz = 1, 
    std::optional<float> n0 = 1.33
);

#endif 
