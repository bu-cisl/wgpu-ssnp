#include <iostream>
#include "ssnp_model.h"

int test() {
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
    std::cout << "\nOutput shape: (" << result.size() << ")\n";

    // Test c_gamma
    std::vector<float> res = {1.0f, 1.0f, 1.0f};
    std::vector<int> shape = {3, 3};
    auto gamma_result = c_gamma(res, shape);

    std::cout << "C++ c_gamma Results: \n";
    size_t batch_size = gamma_result.size();
    for (size_t b = 0; b < batch_size; ++b) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                const auto& elem = gamma_result[b][i * shape[1] + j];
                std::cout << "(" << elem.real() << "," << elem.imag() << ") ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "Output shape: (" << batch_size << ", " << shape[0] << ", " << shape[1] << ")\n";

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

    // Print results as 2D grids (batch x height x width)
    std::cout << "C++ diffract Result (uf_new):\n";
    for (const auto& batch : uf_new) {
        for (const auto& row : batch) {
            for (const auto& elem : row) {
                std::cout << "(" << elem.real() << "," << elem.imag() << ") ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "Output shape (uf_new): (" << uf_new.size() << ", " << shape[0] << ", " << shape[1] << ")\n";
    std::cout << "C++ diffract Result (ub_new):\n";
    for (const auto& batch : ub_new) {
        for (const auto& row : batch) {
            for (const auto& elem : row) {
                std::cout << "(" << elem.real() << "," << elem.imag() << ") ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "Output shape (ub_new): (" << ub_new.size() << ", " << shape[0] << ", " << shape[1] << ")\n";

    // Test binary_pupil
    std::vector<int> shape2 = {5, 4};
    float na = 0.7f;
    std::vector<float> res2 = {0.1f, 0.1f, 0.1f};
    auto mask = binary_pupil(shape2, na, res2);

    std::cout << "C++ binary_pupil result:\n";
    for (const auto& row : mask) {
        for (const auto& val : row) {
            std::cout << (val ? 1 : 0) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Output shape: (" << mask.size() << ", " << mask[0].size() << ")\n";
    

    // Test tilt
    std::vector<int> shape3 = {2, 2};
    std::vector<double> angles = {50.0, 100.0};
    double NA = 0.65;
    std::vector<double> res3 = {0.1f, 0.1f, 0.1f};
    bool trunc = false;
    auto tilt_result = tilt(shape3, angles, NA, res3, trunc);

    std::cout << "C++ tilt Result:\n";
    for (size_t i = 0; i < tilt_result.size(); ++i) {
        for (int y = 0; y < shape3[0]; ++y) { 
            for (int x = 0; x < shape3[1]; ++x) {
                std::cout << "(" << tilt_result[i][y][x].real() << ", "
                          << tilt_result[i][y][x].imag() << ") ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "Output shape: (" << tilt_result.size() << ", " << shape3[0] << ", " << shape3[1] << ")\n";

    return 0;
}