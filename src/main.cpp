#include "scatter_factor.h"
#include <vector>
#include <iostream>

using namespace std;

int main() {
    // scatter factor output for now
    vector<float> input = {1,2,3};
    vector<float> output = scatter_factor(input);
    cout << "scatter factor output: " << endl;
    for (float o: output) cout << o << " ";
    cout << endl;
    return 0;
}