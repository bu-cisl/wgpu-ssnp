#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
#include "tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h"
#include "c_gamma/c_gamma.h"  
#include "webgpu_utils.h"
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

int main() {
    // default init params for ssnp model
    vector<float> res = {0.1,0.1,0.1};
    float na = 0.65;
    int angles_size = 32;
    bool intensity = true;

    // angles_size vectors of c_ba values
    vector<vector<double>> angles(angles_size, vector<double>(2, 0.0));

    // input matrix
    vector<vector<vector<float>>> n(3, vector<vector<float>>(16, vector<float>(16, 1.0f)));

    // ssnp forward function
    vector<int> shape = {int(n[0].size()), int(n[0][0].size())};
    
    for(vector<double> c_ba : angles) {
        continue;
    }

    cout << na << " " << intensity << endl;
}