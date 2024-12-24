#include <iostream>
#include "ssnp_model.h"

int main() {
    std::vector<float> n = {1.0, 2.0, 3.0};
    auto result = scatter_factor(n);

    std::cout << "Scatter Factor Results:\n";
    for (auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
