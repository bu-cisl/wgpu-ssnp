#include "model_dispatcher.h"

std::map<std::string, ModelFunction> model_registry = {
    {"ssnp", ssnp::forward}
};

std::vector<std::vector<std::vector<float>>> dispatch_model(
    const std::string& model_name,
    WebGPUContext& context,
    std::vector<std::vector<std::vector<float>>> n,
    std::vector<float> res,
    float na,
    std::vector<std::vector<float>> angles,
    float n0,
    int outputType
) {
    return model_registry[model_name](context, n, res, na, angles, n0, outputType);
}
