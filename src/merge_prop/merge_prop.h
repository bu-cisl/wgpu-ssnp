#ifndef MERGE_PROP_H
#define MERGE_PROP_H

#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <complex>
#include <optional>
#include <cmath>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"
#include "../c_gamma/c_gamma.h"

void merge_prop(
    WebGPUContext& context,
    wgpu::Buffer& newUFBuffer,
    wgpu::Buffer& newUBBuffer,
    const std::vector<std::complex<float>>& uf,
    const std::vector<std::complex<float>>& ub,
    std::vector<int> shape,
    std::optional<std::vector<float>> res = std::vector<float>{0.1, 0.1, 0.1}
);

#endif