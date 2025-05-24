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

/*
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
*/

int main() {
    vector<vector<vector<float>>> test_input = {
        {
            {1.23f, 4.56f, 7.89f},
            {2.34f, 5.67f, 8.90f},
            {3.45f, 6.78f, 9.01f}
        },
        {
            {0.12f, 3.45f, 6.78f},
            {9.87f, 6.54f, 3.21f},
            {1.11f, 2.22f, 3.33f}
        },
        {
            {7.77f, 8.88f, 9.99f},
            {4.44f, 5.55f, 6.66f},
            {0.01f, 1.02f, 2.03f}
        }
    };

    cout << "Input tensor:" << endl;
    for (auto& slice : test_input) {
        for (auto& row : slice) {
            for (auto val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
    }

    WebGPUContext context;
    initWebGPU(context);

    // Parameters
    vector<float> res = {0.1f, 0.2f, 0.1f};
    float na = 0.69f;
    bool intensity = true;
    vector<vector<float>> angles = {{2.0f, 8.2f}, {1.2f, 5.0f}};
    
    auto result = forward(context, test_input, res, na, angles, intensity);
    
    cout << "Result tensor:" << endl;
    for (auto& slice : result) {
        for (auto& row : slice) {
            for (auto val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
    }
    
    return 0;
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
extern "C" {
    // Reads `input.bin`, runs forward(), prints every element:
    EMSCRIPTEN_KEEPALIVE
    void runForwardFromFile() {
        try {
            // load the bin file into a tensor
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
                /*res*/ {0.1f,0.2f,0.1f},
                /*na*/ 0.69f,
                /*angles*/ {{2.0f, 8.2f}, {1.2f, 5.0f}},
                /*intensity*/ true
            );

            printf("Output dims: %d×%d×%d\n", 
                (int)result.size(),
                result.empty() ? 0 : (int)result[0].size(),
                (result.empty() || result[0].empty()) ? 0 : (int)result[0][0].size()
            );

            // 4) print every element
            for (size_t d = 0; d < result.size(); ++d) {
                printf("Slice %zu:\n", d);
                for (size_t i = 0; i < result[d].size(); ++i) {
                    for (size_t j = 0; j < result[d][i].size(); ++j) {
                        printf("%8.4f ", result[d][i][j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }
        catch (const std::exception &e) {
            printf("C++ exception: %s\n", e.what());
        }
        catch (...) {
            printf("Unknown C++ exception\n");
        }
    }
}
#endif