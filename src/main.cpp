#include <iostream>
#include "ssnp_model.h"

int main() {
    // Test scatter_factor
    std::vector<float> n = {1.0f, 2.0f, 3.0f};
    float res_z = 0.1f;
    float dz = 1.0f;
    float n0 = 1.0f;
    auto result = scatter_factor(n, res_z, dz, n0);

    std::cout << "C++ scatter_factor Results:\n";
    for (auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Test c_gamma
    std::vector<float> res = {1.0f, 1.0f, 1.0f};
    std::vector<int> shape = {3, 3};
    auto gamma_result = c_gamma(res, shape);

    std::cout << "C++ c_gamma Results: \n";
    for (const auto& batch : gamma_result) {
        size_t size_alpha = shape[0];
        size_t size_beta = shape[1];

        // Print in grid format
        for (size_t i = 0; i < size_alpha; ++i) {
            for (size_t j = 0; j < size_beta; ++j) {
                const auto& elem = batch[i * size_beta + j];  // Flattened access
                std::cout << "(" << elem.real() << "," << elem.imag() << ") ";
            }
            std::cout << "\n";
        }
    }

    // Test diffract
    std::vector<std::complex<double>> uf = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0},
        {4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
        {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}
    };

    std::vector<std::complex<double>> ub = {
        {9.0, 0.0}, {8.0, 0.0}, {7.0, 0.0},
        {6.0, 0.0}, {5.0, 0.0}, {4.0, 0.0},
        {3.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}
    };

    std::vector<std::vector<std::complex<double>>> uf_batch = {uf};
    std::vector<std::vector<std::complex<double>>> ub_batch = {ub};

    auto [uf_new, ub_new] = diffract(uf_batch, ub_batch, {1.0, 1.0, 1.0}, 1.0);

    std::cout << "C++ diffract Result (uf_new):\n";
    for (size_t i = 0; i < uf_new[0].size(); ++i) {
        std::cout << "(" << uf_new[0][i].real() << "," << uf_new[0][i].imag() << ") ";
        if ((i + 1) % 3 == 0) std::cout << "\n";
    }

    std::cout << "C++ diffract Result (ub_new):\n";
    for (size_t i = 0; i < ub_new[0].size(); ++i) {
        std::cout << "(" << ub_new[0][i].real() << "," << ub_new[0][i].imag() << ") ";
        if ((i + 1) % 3 == 0) std::cout << "\n";
    }
}
