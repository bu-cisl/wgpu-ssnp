#ifndef COMPLEX_SCALE_H
#define COMPLEX_SCALE_H

#include "../webgpu_utils.h"

void complex_scale(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    float scale
);

#endif
