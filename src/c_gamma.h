#ifndef C_GAMMA_H
#define C_GAMMA_H
#include "webgpu_utils.h"
#include <vector>

std::vector<float> c_gamma(WebGPUContext& context, int width, int height, float res_x, float res_y, float res_z);

#endif 
