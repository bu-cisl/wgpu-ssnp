#ifndef TESTING_IO_H
#define TESTING_IO_H

#include "../ssnp/inverse.h"
#include <string>
#include <vector>

namespace testing_io {

struct ReconstructionInput {
    std::vector<std::vector<std::vector<float>>> measured;
    std::vector<std::vector<float>> angles;
    std::vector<std::vector<std::vector<float>>> initial_volume;
    std::vector<float> res;
    float na;
    float n0;
    int max_iterations;
    float learning_rate;
    float abs_tol;
    float rel_tol;
    int print_every;
    bool verbose;
};

bool read_input_tensor(
    const std::string& filename,
    std::vector<std::vector<std::vector<float>>>& tensor,
    int& D,
    int& H,
    int& W
);

bool write_output_tensor(
    const std::string& filename,
    const std::vector<std::vector<std::vector<float>>>& tensor
);

bool read_reconstruction_input(const std::string& filename, ReconstructionInput& input);

} // namespace testing_io

#endif
