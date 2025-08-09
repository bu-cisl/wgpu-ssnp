#define WEBGPU_CPP_IMPLEMENTATION
#include "../../src/ssnp/forward.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <D> <H> <W>" << endl;
        return 1;
    }
    
    int D = atoi(argv[1]), H = atoi(argv[2]), W = atoi(argv[3]);
    vector<vector<vector<float>>> input_tensor(D, vector<vector<float>>(H, vector<float>(W, 0.0f)));

    WebGPUContext context;
    initWebGPU(context);

    vector<float> res = {0.1f, 0.1f, 0.1f};
    float na = 0.65f;
    bool intensity = true;
    float n0 = 1.33f;
    vector<vector<float>> angles(1, vector<float>(2, 0.0f));

    auto start = chrono::high_resolution_clock::now();
    auto result = forward(context, input_tensor, res, na, angles, n0, intensity);
    auto end = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << duration.count() << endl;

    return 0;
}