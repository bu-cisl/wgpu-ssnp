#ifndef BORN_FORWARD_H
#define BORN_FORWARD_H

#include "../common/binary_pupil/binary_pupil.h"
#include "../common/complex_add/complex_add.h"
#include "../common/dft/dft.h"
#include "../common/intensity/intensity.h"
#include "../common/mult/mult.h"
#include "../common/webgpu_utils.h"
#include "propagation_term/propagation_term.h"
#include "scatter_potential/scatter_potential.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace born {

std::vector<std::vector<std::vector<float>>> forward(
    WebGPUContext& context,
    std::vector<std::vector<std::vector<float>>> n,
    std::vector<float> res,
    float na,
    std::vector<std::vector<float>> angles,
    float n0,
    int outputType
);

} // namespace born

#endif
