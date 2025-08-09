#ifndef FORWARD_H
#define FORWARD_H
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
#include "tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h" 
#include "dft/dft.h"
#include "mult/mult.h"
#include "scatter_effects/scatter_effects.h"
#include "intensity/intensity.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

vector<vector<vector<float>>> forward(
    WebGPUContext& context, 
    vector<vector<vector<float>>> n, 
    vector<float> res, 
    float na, 
    vector<vector<float>> angles, 
    float n0,
    bool intensity
);

#endif