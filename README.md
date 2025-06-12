# WebGPU SSNP Model
## Overview
This project implements a **Split-Step Non-Paraxial (SSNP)** forward model for wave propagation. Originally developed in PyTorch ([here](https://github.com/mitch-gilmore/Deep-Wave/blob/main/deepwave/ssnp.py)), the full forward pass has been reimplemented in **C++** using **WebGPU compute shaders**, and runs in the browser via **Emscripten**.
