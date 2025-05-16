#define WEBGPU_CPP_IMPLEMENTATION
#include "forward.h"
#include <iostream>
#include <vector>
#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

using namespace std;

// In-memory tensor forwarding (shared by both native and wasm)
vector<vector<vector<float>>> run_forward(
    const vector<vector<vector<float>>>& input_tensor
) {
    WebGPUContext context;
    initWebGPU(context);

    vector<float> res = {0.1f, 0.1f, 0.1f};
    float na = 0.65f;
    bool intensity = true;
    vector<vector<float>> angles(1, vector<float>(2, 0.0f));

    return forward(context, input_tensor, res, na, angles, intensity);
}

#ifndef __EMSCRIPTEN__
// Native (CLI) binary: file I/O path
#include <fstream>

bool read_input_tensor(const string& filename, vector<vector<vector<float>>>& tensor, int& D, int& H, int& W) {
    ifstream in(filename, ios::binary);
    if (!in) return false;

    in.read(reinterpret_cast<char*>(&D), sizeof(int));
    in.read(reinterpret_cast<char*>(&H), sizeof(int));
    in.read(reinterpret_cast<char*>(&W), sizeof(int));

    size_t total = static_cast<size_t>(D) * H * W;
    vector<float> buffer(total);
    in.read(reinterpret_cast<char*>(buffer.data()), total * sizeof(float));
    in.close();

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
    if (!out) return false;

    int D = tensor.size();
    int H = tensor[0].size();
    int W = tensor[0][0].size();

    out.write(reinterpret_cast<const char*>(&D), sizeof(int));
    out.write(reinterpret_cast<const char*>(&H), sizeof(int));
    out.write(reinterpret_cast<const char*>(&W), sizeof(int));

    for (int d = 0; d < D; ++d)
        for (int i = 0; i < H; ++i)
            out.write(reinterpret_cast<const char*>(tensor[d][i].data()), W * sizeof(float));

    out.close();
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input.bin> <output.bin>" << endl;
        return 1;
    }

    string input_filename = argv[1];
    string output_filename = argv[2];

    vector<vector<vector<float>>> input_tensor;
    int D, H, W;
    if (!read_input_tensor(input_filename, input_tensor, D, H, W)) return 1;

    auto result = run_forward(input_tensor);

    if (!write_output_tensor(output_filename, result)) return 1;

    return 0;
}
#else
// WebAssembly version: JS ↔ C++ memory interaction
extern "C" {

EMSCRIPTEN_KEEPALIVE
float* run_forward_wasm(const float* input_flat, int D, int H, int W) {
    static vector<vector<vector<float>>> input_tensor;
    input_tensor.resize(D, vector<vector<float>>(H, vector<float>(W)));

    size_t idx = 0;
    for (int d = 0; d < D; ++d)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                input_tensor[d][i][j] = input_flat[idx++];

    auto result = run_forward(input_tensor);

    static vector<float> output_flat;
    output_flat.clear();

    for (const auto& mat : result)
        for (const auto& row : mat)
            output_flat.insert(output_flat.end(), row.begin(), row.end());

    return output_flat.data(); // pointer to heap buffer
}
}
#endif