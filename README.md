# WebGPU SSNP Model
## Overview
This project implements a **Split-Step Non-Paraxial (SSNP)** forward model for wave propagation. Originally developed in PyTorch ([here](https://github.com/mitch-gilmore/Deep-Wave/blob/main/deepwave/ssnp.py)), the full forward pass has been reimplemented in **C++** using **WebGPU compute shaders**, and runs in the browser via **Emscripten**.

## Features
1. **Cross-platform GPU acceleration** - Runs on any modern browser with WebGPU support
2. **Device agnostic** - Compatible with various GPU and compute backends, not tied to a specific platform or vendor
3. **Web Integration** - Native browser support for interactive visualization
4. **Performance** - Achieves near-native performance via WGSL compute shaders
