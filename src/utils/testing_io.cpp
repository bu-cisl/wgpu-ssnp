#include "testing_io.h"

#include <cstdint>
#include <fstream>
#include <iostream>

namespace testing_io {
namespace {

bool read_tensor(
    std::ifstream& in,
    std::vector<std::vector<std::vector<float>>>& tensor,
    int D,
    int H,
    int W
) {
    size_t total = static_cast<size_t>(D) * H * W;
    std::vector<float> buffer(total);
    in.read(reinterpret_cast<char*>(buffer.data()), total * sizeof(float));
    if (!in) {
        return false;
    }

    tensor.resize(D, std::vector<std::vector<float>>(H, std::vector<float>(W)));
    size_t idx = 0;
    for (int d = 0; d < D; ++d)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                tensor[d][i][j] = buffer[idx++];

    return true;
}

} // namespace

bool read_input_tensor(
    const std::string& filename,
    std::vector<std::vector<std::vector<float>>>& tensor,
    int& D,
    int& H,
    int& W
) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return false;
    }

    in.read(reinterpret_cast<char*>(&D), sizeof(int));
    in.read(reinterpret_cast<char*>(&H), sizeof(int));
    in.read(reinterpret_cast<char*>(&W), sizeof(int));
    if (!in) {
        std::cerr << "Failed to read tensor header from: " << filename << std::endl;
        return false;
    }

    return read_tensor(in, tensor, D, H, W);
}

bool write_output_tensor(
    const std::string& filename,
    const std::vector<std::vector<std::vector<float>>>& tensor
) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return false;
    }

    int D = tensor.size();
    int H = tensor[0].size();
    int W = tensor[0][0].size();

    out.write(reinterpret_cast<char*>(&D), sizeof(int));
    out.write(reinterpret_cast<char*>(&H), sizeof(int));
    out.write(reinterpret_cast<char*>(&W), sizeof(int));

    for (int d = 0; d < D; ++d)
        for (int i = 0; i < H; ++i)
            out.write(reinterpret_cast<const char*>(tensor[d][i].data()), W * sizeof(float));

    return true;
}

bool read_reconstruction_input(const std::string& filename, ReconstructionInput& input) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open reconstruction input file: " << filename << std::endl;
        return false;
    }

    int D, H, W, A;
    in.read(reinterpret_cast<char*>(&D), sizeof(int));
    in.read(reinterpret_cast<char*>(&H), sizeof(int));
    in.read(reinterpret_cast<char*>(&W), sizeof(int));
    in.read(reinterpret_cast<char*>(&A), sizeof(int));

    input.res.resize(3);
    in.read(reinterpret_cast<char*>(input.res.data()), sizeof(float) * 3);
    in.read(reinterpret_cast<char*>(&input.na), sizeof(float));
    in.read(reinterpret_cast<char*>(&input.n0), sizeof(float));
    in.read(reinterpret_cast<char*>(&input.max_iterations), sizeof(int));
    in.read(reinterpret_cast<char*>(&input.learning_rate), sizeof(float));
    in.read(reinterpret_cast<char*>(&input.abs_tol), sizeof(float));
    in.read(reinterpret_cast<char*>(&input.rel_tol), sizeof(float));
    in.read(reinterpret_cast<char*>(&input.print_every), sizeof(int));

    uint32_t verbose_flag = 0;
    in.read(reinterpret_cast<char*>(&verbose_flag), sizeof(uint32_t));
    input.verbose = verbose_flag != 0;

    std::vector<float> angle_buffer(static_cast<size_t>(A) * 2);
    in.read(reinterpret_cast<char*>(angle_buffer.data()), sizeof(float) * angle_buffer.size());
    if (!in) {
        std::cerr << "Failed to read reconstruction metadata." << std::endl;
        return false;
    }

    input.angles.assign(A, std::vector<float>(2, 0.0f));
    for (int a = 0; a < A; ++a) {
        input.angles[a][0] = angle_buffer[static_cast<size_t>(a) * 2];
        input.angles[a][1] = angle_buffer[static_cast<size_t>(a) * 2 + 1];
    }

    if (!read_tensor(in, input.measured, A, H, W)) {
        std::cerr << "Failed to read measured stack." << std::endl;
        return false;
    }

    if (!read_tensor(in, input.initial_volume, D, H, W)) {
        std::cerr << "Failed to read initial volume." << std::endl;
        return false;
    }

    return true;
}

} // namespace testing_io
