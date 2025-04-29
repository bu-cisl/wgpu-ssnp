#define WEBGPU_CPP_IMPLEMENTATION
#include "forward.h"

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    initWebGPU(context);

    // default init params for ssnp model
    vector<float> res = {0.1,0.1,0.1};
    float na = 0.65;
    int angles_size = 3;
    bool intensity = true;
    vector<vector<float>> angles(angles_size, vector<float>(2, 0.0)); // angles_size vectors of c_ba values, default [0,0]

    // input matrix
    vector<vector<vector<float>>> n(3, vector<vector<float>>(4, vector<float>(4, 1.0f)));

    // compute result with forward function
    vector<vector<vector<float>>> result = forward(context, n, res, na, angles, intensity);

    cout << "Final Result:" << endl;
    for (size_t i = 0; i < result.size(); i++) {
        for (size_t j = 0; j < result[i].size(); j++) {
            for (size_t k = 0; k < result[i][j].size(); k++) {
                cout << result[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

}