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

std::pair<std::vector<std::vector<std::complex<double>>>, std::vector<std::vector<std::complex<double>>>> 
diffract(const std::vector<std::vector<std::complex<double>>>& uf,
         const std::vector<std::vector<std::complex<double>>>& ub,
         const std::vector<float>& res = {0.1f, 0.1f, 0.1f}, 
         float dz = 1.0f) {
    
    int size_alpha = uf[0].size();  // assuming 1 batch, uf[0] is 3x3
    int size_beta = ub[0].size();
    int size = std::sqrt(size_alpha);  // matrix side length (3x3)

    // Generate cgamma
    auto cgamma = c_gamma(res, {size, size});

    // kz = 2 * pi * res[2] * cgamma
    std::vector<std::vector<std::complex<double>>> kz(1, std::vector<std::complex<double>>(size_alpha));
    for (int i = 0; i < size_alpha; ++i) {
        kz[0][i] = 2.0 * M_PI * res[2] * cgamma[0][i];
    }

    // eva = exp(clamp((cgamma - 0.2) * 5, max=0))
    std::vector<std::vector<std::complex<double>>> eva(1, std::vector<std::complex<double>>(size_alpha));
    for (int i = 0; i < size_alpha; ++i) {
        auto val = std::real(cgamma[0][i]) - 0.2;
        val = std::min(val * 5.0, 0.0);
        eva[0][i] = std::exp(val);
    }

    // p_mat = [cos(kz * dz), sin(kz * dz) / kz, -sin(kz * dz) * kz, cos(kz * dz)]
    std::vector<std::vector<std::complex<double>>> p_mat(4, std::vector<std::complex<double>>(size_alpha));

    for (int i = 0; i < size_alpha; ++i) {
        auto kz_val = kz[0][i] * dz;
        p_mat[0][i] = std::cos(kz_val);
        p_mat[1][i] = std::sin(kz_val) / (kz[0][i] + 1e-8);  // avoid division by zero
        p_mat[2][i] = -std::sin(kz_val) * kz[0][i];
        p_mat[3][i] = std::cos(kz_val);
    }

    // Scale p_mat by eva
    for (int i = 0; i < size_alpha; ++i) {
        for (int j = 0; j < 4; ++j) {
            p_mat[j][i] *= eva[0][i];
        }
    }

    // Forward diffraction
    std::vector<std::vector<std::complex<double>>> uf_new(1, std::vector<std::complex<double>>(size_alpha));
    std::vector<std::vector<std::complex<double>>> ub_new(1, std::vector<std::complex<double>>(size_alpha));

    for (int i = 0; i < size_alpha; ++i) {
        uf_new[0][i] = p_mat[0][i] * uf[0][i] + p_mat[1][i] * ub[0][i];
        ub_new[0][i] = p_mat[2][i] * uf[0][i] + p_mat[3][i] * ub[0][i];
    }

    return {uf_new, ub_new};
}