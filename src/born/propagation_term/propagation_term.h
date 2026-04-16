#ifndef BORN_PROPAGATION_TERM_H
#define BORN_PROPAGATION_TERM_H

#include "../../common/webgpu_utils.h"

namespace born {

void propagation_term(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::vector<float> res,
    std::vector<float> c_ba,
    float depth
);

} // namespace born

#endif
