#ifndef SSNP_DIFFRACT_GRAD_H
#define SSNP_DIFFRACT_GRAD_H

#include "../../common/webgpu_utils.h"
#include "../../common/c_gamma/c_gamma.h"
#include <optional>
#include <vector>

void diffract_grad(
    WebGPUContext& context,
    wgpu::Buffer& newUFBuffer,
    wgpu::Buffer& newUBBuffer,
    wgpu::Buffer& ufBuffer,
    wgpu::Buffer& ubBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<std::vector<float>> res,
    std::optional<float> dz
);

#endif
