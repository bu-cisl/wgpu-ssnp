#include "ssnp_model.h"
#include <cmath>
#include <vector>
#include <complex>
#include <cassert>
#include <iostream>

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

std::pair<std::vector<std::vector<std::vector<std::complex<double>>>>, std::vector<std::vector<std::vector<std::complex<double>>>>>
diffract(
    const std::vector<std::vector<std::complex<double>>>& uf,
    const std::vector<std::vector<std::complex<double>>>& ub,
    const std::vector<float>& res, 
    float dz) {
    assert(uf.size() == ub.size() && uf[0].size() == ub[0].size());

    int batch_size = uf.size();
    int size = uf[0].size();  // Flattened total size

    // Calculate shape dimensions (height x width)
    auto shape = std::vector<int>{static_cast<int>(std::sqrt(size)), static_cast<int>(std::sqrt(size))};
    while (shape[0] * shape[1] != size) {
        shape[1] += 1;
        if (shape[0] * shape[1] > size) {
            shape[0] += 1;
            shape[1] = size / shape[0];
        }
    }
    int height = shape[0];
    int width = shape[1];

    // Compute cgamma and kz
    auto cgamma = c_gamma(res, shape);
    std::vector<std::vector<std::complex<double>>> kz(batch_size, std::vector<std::complex<double>>(size));
    std::vector<std::vector<std::complex<double>>> eva(batch_size, std::vector<std::complex<double>>(size));

    double pi = 3.141592653589793;
    for (int b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < size; ++i) {
            kz[b][i] = std::complex<float>(2.0f * pi * res[2]) * cgamma[0][i];
            eva[b][i] = std::exp(std::min(std::abs((cgamma[0][i] - std::complex<float>(0.2)) * 5.0f), 0.0f));
        }
    }

    // Calculate p_mat
    std::vector<std::vector<std::vector<std::complex<double>>>> p_mat(4,
        std::vector<std::vector<std::complex<double>>>(batch_size, 
            std::vector<std::complex<double>>(size)));

    for (int b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < size; ++i) {
            std::complex<double> kz_val = kz[b][i];
            p_mat[0][b][i] = std::cos(kz_val * std::complex<double>(dz));
            p_mat[1][b][i] = std::sin(kz_val * std::complex<double>(dz)) / (kz_val + 1e-8);
            p_mat[2][b][i] = -std::sin(kz_val * std::complex<double>(dz)) * (kz_val);
            p_mat[3][b][i] = std::cos(kz_val * std::complex<double>(dz));

            for (int j = 0; j < 4; ++j) {
                p_mat[j][b][i] *= eva[b][i];
            }
        }
    }

    // Initialize reshaped uf_new and ub_new
    std::vector<std::vector<std::vector<std::complex<double>>>> uf_new(batch_size,
        std::vector<std::vector<std::complex<double>>>(height, std::vector<std::complex<double>>(width)));

    std::vector<std::vector<std::vector<std::complex<double>>>> ub_new(batch_size,
        std::vector<std::vector<std::complex<double>>>(height, std::vector<std::complex<double>>(width)));

    // Compute uf_new and ub_new and reshape
    for (int b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < size; ++i) {
            int row = i / width;
            int col = i % width;

            uf_new[b][row][col] = p_mat[0][b][i] * uf[b][i] + p_mat[1][b][i] * ub[b][i];
            ub_new[b][row][col] = p_mat[2][b][i] * uf[b][i] + p_mat[3][b][i] * ub[b][i];
        }
    }

    return {uf_new, ub_new};
}

std::vector<std::vector<bool>> binary_pupil(
    const std::vector<int>& shape, 
    float na, 
    const std::vector<float>& res) {
    auto cgamma = c_gamma(res, shape);
    size_t height = shape[0];
    size_t width = shape[1];

    std::vector<std::vector<bool>> mask(height, std::vector<bool>(width));
    double threshold = std::sqrt(1 - na * na);
    const auto& cgamma_batch = cgamma[0];

    // Compare each value of cgamma with the threshold and assign to mask
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            // Use std::abs to compute the magnitude of the complex number
            mask[i][j] = std::abs(cgamma_batch[i * width + j]) > threshold;
        }
    }

    return mask;
}

std::vector<std::vector<std::vector<std::complex<double>>>> tilt(
    const std::vector<int>& shape,
    const std::vector<double>& angles,
    double NA,
    const std::vector<double>& res,
    bool trunc) {
    std::vector<std::vector<double>> c_ba(shape[0], std::vector<double>(2));

    // Compute c_ba (sine and cosine components)
    for (size_t i = 0; i < angles.size(); ++i) {
        c_ba[i][0] = NA * std::sin(angles[i]);
        c_ba[i][1] = NA * std::cos(angles[i]);
    }
    std::cout << "c_ba (sine and cosine components):\n";
    for (size_t i = 0; i < angles.size(); ++i) {
        std::cout << "(" << c_ba[i][0] << ", " << c_ba[i][1] << ")\n";
    }

    // Compute norm (shape * resolution)
    std::vector<double> norm = {shape[1] * res[1], shape[0] * res[0]};
    std::cout << "norm (shape * resolution):\n";
    std::cout << "(" << norm[0] << ", " << norm[1] << ")\n";

    // Compute factor (after truncation check)
    std::vector<std::vector<double>> factor(angles.size(), std::vector<double>(2));

    for (size_t i = 0; i < angles.size(); ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double value = c_ba[i][j] * norm[j];
            if (trunc) {
                // Apply truncation if trunc is true
                factor[i][j] = std::trunc(value);  // Truncate the value
            } else {
                // Otherwise, store the non-truncated value
                factor[i][j] = value;
            }
        }
    }

    std::cout << "factor (after truncation check):\n";
    for (size_t i = 0; i < angles.size(); ++i) {
        std::cout << "(" << factor[i][0] << ", " << factor[i][1] << ")\n";
    }
}