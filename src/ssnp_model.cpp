#include "ssnp_model.h"
#include <cmath>  // for M_PI

std::vector<float> scatter_factor(const std::vector<float>& n, float res_z, float dz, float n0) {
    float factor = std::pow(2 * M_PI * res_z / n0, 2) * dz;
    std::vector<float> result(n.size());

    for (size_t i = 0; i < n.size(); ++i) {
        result[i] = factor * n[i] * (2 * n0 + n[i]);
    }

    return result;
}
