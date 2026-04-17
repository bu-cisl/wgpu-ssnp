#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include <cmath>
#include <stdexcept>

// Check if a number is a power of 2
inline bool isPowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Get log base 2 of a power-of-2 number
inline int log2Int(int n) {
    if (!isPowerOf2(n)) {
        throw std::invalid_argument("log2Int requires a power of 2");
    }
    int log = 0;
    int temp = n;
    while (temp > 1) {
        temp >>= 1;
        log++;
    }
    return log;
}

// Validate FFT input dimensions
inline bool isValidFFTDimensions(int rows, int cols) {
    return isPowerOf2(rows) && isPowerOf2(cols);
}

#endif // FFT_UTILS_H
