#ifndef BPM_DIFFRACT_H
#define BPM_DIFFRACT_H
#include <webgpu/webgpu.hpp>
#include <optional>
#include "../../common/webgpu_utils.h"
#include "../../common/c_gamma/c_gamma.h"

void diffract(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer, 
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<std::vector<float>> res = std::vector<float>{0.1, 0.1, 0.1}, 
    std::optional<float> dz = 1.0
);

#endif