#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/webgpu_utils.h"
#include "ssnp/forward.h"
#include "ssnp/inverse.h"

namespace {

using Volume = std::vector<std::vector<std::vector<float>>>;
using Stack = std::vector<std::vector<std::vector<float>>>;

float measurement_mse(const Stack& a, const Stack& b) {
    if (a.size() != b.size() || a.empty()) {
        throw std::runtime_error("Measurement stacks must have matching non-empty shapes.");
    }

    float sum = 0.0f;
    size_t count = 0;
    for (size_t angle = 0; angle < a.size(); ++angle) {
        for (size_t y = 0; y < a[angle].size(); ++y) {
            for (size_t x = 0; x < a[angle][y].size(); ++x) {
                float diff = a[angle][y][x] - b[angle][y][x];
                sum += diff * diff;
                ++count;
            }
        }
    }
    return sum / static_cast<float>(count);
}

float volume_mse(const Volume& a, const Volume& b) {
    float sum = 0.0f;
    size_t count = 0;

    for (size_t z = 0; z < a.size(); ++z) {
        for (size_t y = 0; y < a[z].size(); ++y) {
            for (size_t x = 0; x < a[z][y].size(); ++x) {
                float diff = a[z][y][x] - b[z][y][x];
                sum += diff * diff;
                ++count;
            }
        }
    }
    return sum / static_cast<float>(count);
}

bool all_finite(const Volume& volume) {
    for (const auto& slice : volume) {
        for (const auto& row : slice) {
            for (float value : row) {
                if (!std::isfinite(value)) {
                    return false;
                }
            }
        }
    }
    return true;
}

void add_blob(
    Volume& volume,
    float z_center,
    float y_center,
    float x_center,
    float radius,
    float amplitude
) {
    for (size_t z = 0; z < volume.size(); ++z) {
        for (size_t y = 0; y < volume[z].size(); ++y) {
            for (size_t x = 0; x < volume[z][y].size(); ++x) {
                float dz = static_cast<float>(z) - z_center;
                float dy = static_cast<float>(y) - y_center;
                float dx = static_cast<float>(x) - x_center;
                float r2 = dx * dx + dy * dy + dz * dz;
                volume[z][y][x] += amplitude * std::exp(-0.5f * r2 / (radius * radius));
            }
        }
    }
}

Volume make_target_volume(int depth, int height, int width) {
    Volume volume(
        depth,
        std::vector<std::vector<float>>(height, std::vector<float>(width, 0.002f))
    );

    add_blob(volume, 0.25f * depth, 0.30f * height, 0.30f * width, 0.28f * depth, 0.025f);
    add_blob(volume, 0.60f * depth, 0.65f * height, 0.60f * width, 0.32f * depth, 0.020f);
    add_blob(volume, 0.45f * depth, 0.50f * height, 0.40f * width, 0.45f * depth, 0.008f);

    return volume;
}

Volume make_initial_volume(int depth, int height, int width) {
    return Volume(
        depth,
        std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f))
    );
}

} // namespace

int main() {
    try {
        WebGPUContext context;
        initWebGPU(context);

        const std::vector<float> res = {0.20f, 0.20f, 0.20f};
        const float na = 0.65f;
        const float n0 = 1.33f;

        // circular angle pattern for test
        std::vector<std::vector<float>> angles;
        angles.push_back({0.0f, 0.0f});
        int angle_count = 15;
        float r = 0.2f;
        for (int i = 0; i < angle_count; ++i) {
            float theta = 2.0f * M_PI * i / angle_count;
            angles.push_back({r * std::cos(theta), r * std::sin(theta)});
        }

        constexpr int depth = 64;
        constexpr int height = 100;
        constexpr int width = 100;

        Volume target = make_target_volume(depth, height, width);
        Volume initial = make_initial_volume(depth, height, width);

        Stack measured = ssnp::forward(context, target, res, na, angles, n0, 1);
        Stack initial_prediction = ssnp::forward(context, initial, res, na, angles, n0, 1);
        float initial_measurement_error = measurement_mse(initial_prediction, measured);
        float initial_volume_error = volume_mse(initial, target);

        ssnp::ReconstructionOptions options;
        options.max_iterations = 100;
        options.learning_rate = 5e-1f;
        options.abs_tol = 1e-10f;
        options.rel_tol = 1e-6f;
        options.print_every = 1;
        options.verbose = true;

        ssnp::ReconstructionResult result = ssnp::reconstruct(
            context,
            measured,
            angles,
            initial,
            res,
            na,
            n0,
            options
        );

        Stack reconstructed_prediction = ssnp::forward(context, result.volume, res, na, angles, n0, 1);
        float final_measurement_error = measurement_mse(reconstructed_prediction, measured);
        float final_volume_error = volume_mse(result.volume, target);

        bool measurement_improved =
            all_finite(result.volume) &&
            !result.loss_history.empty() &&
            final_measurement_error < initial_measurement_error;

        std::cout << std::setprecision(10);
        std::cout << "reconstruction result";
        if (measurement_improved) {
            std::cout << " (PASS)";
        } else {
            std::cout << " (FAIL)";
        }
        std::cout << "\nmeasurement_mse: " << initial_measurement_error
                  << " -> " << final_measurement_error
                  << "\nvolume_mse: " << initial_volume_error
                  << " -> " << final_volume_error
                  << "\niterations: " << result.iterations_run
                  << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL " << ex.what() << std::endl;
        return 1;
    }
}
