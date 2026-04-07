#ifndef SSNP_SPLIT_PROP_GRAD_H
#define SSNP_SPLIT_PROP_GRAD_H

#include "../../common/webgpu_utils.h"
#include "../../common/c_gamma/c_gamma.h"
#include <optional>
#include <vector>

void split_prop_grad(
    WebGPUContext& context,
    wgpu::Buffer& uGradBuffer,
    wgpu::Buffer& udGradBuffer,
    wgpu::Buffer& forwardGradBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<std::vector<float>> res
);

#endif
