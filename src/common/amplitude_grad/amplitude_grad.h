#ifndef AMPLITUDE_GRAD_H
#define AMPLITUDE_GRAD_H

#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void amplitude_grad(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& fieldBuffer,
    wgpu::Buffer& measuredBuffer,
    size_t bufferlen,
    float inv_pixels
);

#endif