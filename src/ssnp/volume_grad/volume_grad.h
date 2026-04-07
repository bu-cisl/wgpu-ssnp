#ifndef SSNP_VOLUME_GRAD_H
#define SSNP_VOLUME_GRAD_H

#include "../../common/webgpu_utils.h"

void volume_grad(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& dqBuffer,
    wgpu::Buffer& gradBuffer,
    wgpu::Buffer& uBuffer,
    size_t bufferlen
);

#endif
