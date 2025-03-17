#ifndef SCATTER_FACTOR_H
#define SCATTER_FACTOR_H
#include <vector>
#include <optional>

std::vector<float> scatter_factor(std::vector<float> inputData, std::optional<float> res_z = 0.1, std::optional<float> dz = 1, std::optional<float> n0 = 1);

#endif 
