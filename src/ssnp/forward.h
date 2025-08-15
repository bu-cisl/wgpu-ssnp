#ifndef SSNP_FORWARD_H
#define SSNP_FORWARD_H

#include "scatter_factor/scatter_factor.h"
#include "ssnp_diffract/ssnp_diffract.h"
#include "../common/binary_pupil/binary_pupil.h"
#include "../common/tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h" 
#include "../common/dft/dft.h"
#include "../common/mult/mult.h"
#include "scatter_effects/scatter_effects.h"
#include "../common/intensity/intensity.h"
#include "../common/webgpu_utils.h"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

namespace ssnp {

    vector<vector<vector<float>>> forward(
        WebGPUContext& context, 
        vector<vector<vector<float>>> n, 
        vector<float> res, 
        float na, 
        vector<vector<float>> angles, 
        float n0,
        int outputType
    );
}

#endif