#ifndef BPM_FORWARD_H
#define BPM_FORWARD_H

#include "../common/webgpu_utils.h"
#include "../common/dft/dft.h"
#include "../common/tilt/tilt.h"
#include "bpm_diffract/bpm_diffract.h"
#include "scatter/scatter.h"
#include "../common/intensity/intensity.h"
#include "../common/binary_pupil/binary_pupil.h"
#include "../common/mult/mult.h"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

namespace bpm {

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