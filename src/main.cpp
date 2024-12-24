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
    for (const auto& batch : gamma_result) {  // Iterate over the batch dimension (size 1)
        size_t size_alpha = shape[0];
        size_t size_beta = shape[1];

        // Print in grid format
        for (size_t i = 0; i < size_alpha; ++i) {
            for (size_t j = 0; j < size_beta; ++j) {
                const auto& elem = batch[i * size_beta + j];  // Flattened access
                std::cout << "(" << elem.real() << "," << elem.imag() << ") ";
            }
            std::cout << "\n";  // Newline after each row
        }
        std::cout << "\n";  // Extra newline for batch separation
    }
}
