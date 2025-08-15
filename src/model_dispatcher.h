#ifndef MODEL_DISPATCHER_H
#define MODEL_DISPATCHER_H

#include "webgpu_utils.h"
#include <vector>
#include <string>
#include <map>

typedef std::vector<std::vector<std::vector<float>>> (*ModelFunction)(
    WebGPUContext&, 
    std::vector<std::vector<std::vector<float>>>, 
    std::vector<float>, 
    float, 
    std::vector<std::vector<float>>, 
    float, 
    int
);

std::vector<std::vector<std::vector<float>>> dispatch_model(
    const std::string& model_name,
    WebGPUContext& context,
    std::vector<std::vector<std::vector<float>>> n,
    std::vector<float> res,
    float na,
    std::vector<std::vector<float>> angles,
    float n0,
    int outputType
);

#endif
