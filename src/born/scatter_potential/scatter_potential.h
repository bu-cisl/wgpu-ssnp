#ifndef BORN_SCATTER_POTENTIAL_H
#define BORN_SCATTER_POTENTIAL_H

#include "../../common/webgpu_utils.h"

namespace born {

void scatter_potential(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    float res_z,
    float n0
);

} // namespace born

#endif
