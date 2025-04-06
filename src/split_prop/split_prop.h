#ifndef SPLIT_PROP_H
#define SPLIT_PROP_H
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <complex>
#include <optional>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"
#include "../c_gamma/c_gamma.h"

void split_prop(
    WebGPUContext& context,
    wgpu::Buffer& newUFBuffer,
    wgpu::Buffer& newUBBuffer,
    std::vector<std::complex<float>> uf,
    std::vector<std::complex<float>> ub,
    std::vector<int> shape,
    std::optional<std::vector<float>> res = std::vector<float>{0.1, 0.1, 0.1}
);

#endif