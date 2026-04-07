#ifndef SSNP_SCATTER_DERIVATIVE_H
#define SSNP_SCATTER_DERIVATIVE_H

#include "../../common/webgpu_utils.h"
#include <optional>

void scatter_derivative(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    std::optional<float> res_z,
    std::optional<float> dz,
    std::optional<float> n0
);

#endif
