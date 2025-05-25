#define WEBGPU_CPP_IMPLEMENTATION
#include "forward.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

bool read_input_tensor(const string& filename, vector<vector<vector<float>>>& tensor, int& D, int& H, int& W) {
    ifstream in(filename, ios::binary);
    if (!in) {
        cerr << "Failed to open input file: " << filename << endl;
        return false;
    }

    // Read dimensions (3 x int32)
    in.read(reinterpret_cast<char*>(&D), sizeof(int));
    in.read(reinterpret_cast<char*>(&H), sizeof(int));
    in.read(reinterpret_cast<char*>(&W), sizeof(int));

    size_t total = static_cast<size_t>(D) * H * W;
    vector<float> buffer(total);
    in.read(reinterpret_cast<char*>(buffer.data()), total * sizeof(float));
    in.close();

    // Convert to 3D tensor
    tensor.resize(D, vector<vector<float>>(H, vector<float>(W)));
    size_t idx = 0;
    for (int d = 0; d < D; ++d)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                tensor[d][i][j] = buffer[idx++];

    return true;
}

bool write_output_tensor(const string& filename, const vector<vector<vector<float>>>& tensor) {
    ofstream out(filename, ios::binary);
    if (!out) {
        cerr << "Failed to open output file: " << filename << endl;
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

    out.close();
    return true;
}

int main() {
    return 0;
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <sstream>

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    void runForwardFromFile() {
        try {
            std::vector<std::vector<std::vector<float>>> tensor;
            int D, H, W;
            if (!read_input_tensor("input.bin", tensor, D, H, W)) {
                printf("Failed to read input.bin\n");
                return;
            }

            WebGPUContext context;
            initWebGPU(context);
            auto result = forward(
                context,
                tensor,
                {0.1f, 0.1f, 0.1f}, // default res
                0.65f, // default na
                {
                    {0.0f, 0.0f}, 
                    {0.0f, 0.0f}
                }, // default angle twice
                true // default intensity
            );

            // Flatten the tensor into a string for JS
            std::ostringstream oss;
            oss << "[";
            for (size_t d = 0; d < result.size(); ++d)
                for (size_t i = 0; i < result[d].size(); ++i)
                    for (size_t j = 0; j < result[d][i].size(); ++j)
                        oss << result[d][i][j] << (d == result.size()-1 && i == result[d].size()-1 && j == result[d][i].size()-1 ? "" : ",");

            oss << "]";

            int outD = result.size();
            int outH = result[0].size();
            int outW = result[0][0].size();

            float globalMin = result[0][0][0];
            float globalMax = result[0][0][0];

            for (const auto& slice : result) {
                for (const auto& row : slice) {
                    for (float val : row) {
                        if (val < globalMin) globalMin = val;
                        if (val > globalMax) globalMax = val;
                    }
                }
            }

            // Flatten and send to JS as before...
            std::ostringstream jsCall;
            jsCall << "plotSlices(" << oss.str() << "," << outD << "," << outH << "," << outW
                << "," << globalMin << "," << globalMax << ");";
            emscripten_run_script(jsCall.str().c_str());

        } catch (const std::exception &e) {
            printf("C++ exception: %s\n", e.what());
        } catch (...) {
            printf("Unknown C++ exception\n");
        }
    }
}
#endif
