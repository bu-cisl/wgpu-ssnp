#ifndef INTENSITY_GRAD_H
#define INTENSITY_GRAD_H

#include "../webgpu_utils.h"

void intensity_grad(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& fieldBuffer,
    wgpu::Buffer& measuredBuffer,
    size_t bufferlen,
    float scale
);

#endif
