#include "ssnp_model.h"
#include <cmath>
#include <vector>
#include <complex>

std::vector<float> scatter_factor(const std::vector<float>& n, float res_z, float dz, float n0) {
    float factor = std::pow(2 * M_PI * res_z / n0, 2) * dz;
    std::vector<float> result(n.size());
    for (size_t i = 0; i < n.size(); ++i) {
        result[i] = factor * n[i] * (2 * n0 + n[i]);
    }
    return result;
}

std::vector<float> near_0(int size) {
    std::vector<float> result(size);
    for (int i = 0; i < size; ++i) {
        result[i] = std::fmod((i / float(size)) + 0.5f, 1.0f) - 0.5f;
    }
    return result;
}

std::vector<std::vector<std::complex<float>>> c_gamma(const std::vector<float>& res, const std::vector<int>& shape) {
    // Generate c_beta and c_alpha
    int size_alpha = shape[0];
    int size_beta = shape[1];
    std::vector<float> c_alpha = near_0(size_alpha);
    std::vector<float> c_beta = near_0(size_beta);

    // Normalize by resolution
    for (int i = 0; i < size_alpha; ++i) {
        c_alpha[i] /= res[1];
    }

    for (int i = 0; i < size_beta; ++i) {
        c_beta[i] /= res[0];
    }

    std::vector<std::vector<std::complex<float>>> result(1, std::vector<std::complex<float>>(size_alpha * size_beta));
    float eps = 1E-8f; // epsilon to avoid sqrt of negative numbers
    for (int i = 0; i < size_alpha; ++i) {
        for (int j = 0; j < size_beta; ++j) {
            float val = 1.0f - (std::pow(c_alpha[i], 2) + std::pow(c_beta[j], 2));
            result[0][i * size_beta + j] = std::sqrt(std::max(val, eps));
        }
    }
    // Unsqueeze
    return {result};
}