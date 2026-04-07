#ifndef COMPLEX_ADD_H
#define COMPLEX_ADD_H

#include "../webgpu_utils.h"

void complex_add(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer1,
    wgpu::Buffer& inputBuffer2,
    size_t bufferlen
);

#endif
