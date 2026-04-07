#ifndef SSNP_IDIFFRACT_H
#define SSNP_IDIFFRACT_H

#include "ssnp_diffract.h"

inline void idiffract(
    WebGPUContext& context,
    wgpu::Buffer& newUFBuffer,
    wgpu::Buffer& newUBBuffer,
    wgpu::Buffer& ufBuffer,
    wgpu::Buffer& ubBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<std::vector<float>> res,
    std::optional<float> dz
) {
    diffract(context, newUFBuffer, newUBBuffer, ufBuffer, ubBuffer, bufferlen, shape, res, -dz.value());
}

#endif
