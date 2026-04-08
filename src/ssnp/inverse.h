#ifndef SSNP_INVERSE_H
#define SSNP_INVERSE_H

#include "pipeline.h"
#include "ssnp_diffract/idiffract.h"
#include "diffract_grad/diffract_grad.h"
#include "split_prop_grad/split_prop_grad.h"
#include "scatter_derivative/scatter_derivative.h"
#include "volume_grad/volume_grad.h"
#include "../common/complex_add/complex_add.h"
#include "../common/complex_mult/complex_mult.h"
#include "../common/complex_scale/complex_scale.h"
#include "../common/intensity_grad/intensity_grad.h"

namespace ssnp {

struct ReconstructionOptions {
    int max_iterations = 50;
    float learning_rate = 1e-2f;
    float abs_tol = 1e-8f;
    float rel_tol = 1e-6f;
    int print_every = 10;
    bool verbose = false;
};

struct ReconstructionResult {
    std::vector<std::vector<std::vector<float>>> volume;
    std::vector<float> loss_history;
    float best_loss = 0.0f;
    float final_loss = 0.0f;
    int iterations_run = 0;
    float final_learning_rate = 0.0f;
};

ReconstructionResult reconstruct(
    WebGPUContext& context,
    const std::vector<std::vector<std::vector<float>>>& measured,
    const std::vector<std::vector<float>>& angles,
    std::vector<std::vector<std::vector<float>>> initial_volume,
    const std::vector<float>& res,
    float na,
    float n0,
    const ReconstructionOptions& options
);

ReconstructionResult reconstruct(
    WebGPUContext& context,
    const std::vector<std::vector<std::vector<float>>>& measured,
    const std::vector<std::vector<float>>& angles,
    std::vector<std::vector<std::vector<float>>> initial_volume,
    const std::vector<float>& res,
    float na,
    float n0,
    int iterations,
    float learning_rate
);

}

#endif
