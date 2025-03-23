#include "scatter_factor/scatter_factor.h"
#include "c_gamma/c_gamma.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>

using namespace std;

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    if (!initWebGPU(context)) {
        cerr << "Failed to initialize WebGPU!" << endl;
        return -1;
    }

    // Call scatter_factor
    vector<float> input = {1, 2, 3};
    vector<float> output = scatter_factor(context, input);

    // Print scatter_factor output
    cout << "scatter_factor output: " << endl;
    for (float o : output) {
        cout << o << " ";
    }
    cout << endl;

    // Test c_gamma
    std::vector<float> res = {1.0f, 1.0f};
    std::vector<int> shape = {3, 3};
    vector<float> gamma_output = c_gamma(context, res, shape);

    // Print c_gamma output
    cout << "c_gamma output:" << endl;

    for (int i = 0; i < shape[0]; i++) {
        cout << "[";
        for (int j = 0; j < shape[1]; j++) {
            cout << gamma_output[i * shape[1] + j];
            if (j < shape[1] - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);

    return 0;
}