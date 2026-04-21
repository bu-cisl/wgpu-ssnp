#include "../../src/ssnp/forward.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <D> <H> <W>" << std::endl;
        return 1;
    }

    const int D = std::atoi(argv[1]);
    const int H = std::atoi(argv[2]);
    const int W = std::atoi(argv[3]);
    if (D <= 0 || H <= 0 || W <= 0) {
        std::cerr << "All dimensions must be positive." << std::endl;
        return 1;
    }

    std::vector<std::vector<std::vector<float>>> input_tensor(
        static_cast<size_t>(D),
        std::vector<std::vector<float>>(
            static_cast<size_t>(H),
            std::vector<float>(static_cast<size_t>(W), 0.0f)
        )
    );

    WebGPUContext context;
    initWebGPU(context);

    const std::vector<float> res = {0.1f, 0.1f, 0.1f};
    constexpr float na = 0.65f;
    constexpr float n0 = 1.33f;
    constexpr int output_type = 1;
    const std::vector<std::vector<float>> angles(1, std::vector<float>(2, 0.0f));

    const auto start = std::chrono::high_resolution_clock::now();
    const auto result = ssnp::forward(context, input_tensor, res, na, angles, n0, output_type);
    const auto end = std::chrono::high_resolution_clock::now();

    if (result.empty()) {
        throw std::runtime_error("SSNP benchmark returned no output.");
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << duration.count() << std::endl;

    return 0;
}
