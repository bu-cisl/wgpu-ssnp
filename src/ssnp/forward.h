#ifndef SSNP_FORWARD_H
#define SSNP_FORWARD_H

#include "pipeline.h"
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
