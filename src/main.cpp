#define WEBGPU_CPP_IMPLEMENTATION
#include "forward.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

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

    WebGPUContext context;
    initWebGPU(context);

    vector<float> res = {0.1f, 0.1f, 0.1f};
    float na = 0.65f;
    bool intensity = true;
    vector<vector<float>> angles(1, vector<float>(2, 0.0f)); // default [0, 0]

    auto result = forward(context, input_tensor, res, na, angles, intensity);

    if (!write_output_tensor(output_filename, result)) return 1;

    return 0;
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
extern "C" {

// this will be exported into `Module` as `_runForwardOnce`
EMSCRIPTEN_KEEPALIVE
void runForwardOnce() {
    // 1) hard-coded inputs for now
    vector<vector<vector<float>>> input_tensor = {{{1,2,3},{4,5,6},{7,8,9}}};
    float na = 0.65f;
    bool intensity = true;
    vector<float> res = {0.1f,0.1f,0.1f};
    vector<vector<float>> angles = {{0.0f,0.0f}};

    // 2) init WebGPU
    WebGPUContext context;
    initWebGPU(context);

    // 3) run your compute
    auto output = forward(context, input_tensor, res, na, angles, intensity);

    // 4) for demo, print the first element to the browser console
    if (!output.empty() && !output[0][0].empty()) {
      printf("forward[0][0][0] = %f\n", output[0][0][0]);
    }
}

} // extern "C"
#endif