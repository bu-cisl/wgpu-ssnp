#ifndef C_GAMMA_H
#define C_GAMMA_H
#include "webgpu_utils.h"
#include <vector>

std::vector<float> c_gamma(WebGPUContext& context, const std::vector<float>& res, const std::vector<int>& shape);

#endif 
