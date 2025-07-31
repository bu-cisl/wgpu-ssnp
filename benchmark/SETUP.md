## Setup for Benchmarking Experiments

### Manually rebuild the new C++ executable within this directory for experiments
```
cmake -B build -S .
cmake --build build
```

### Setup conda env for PyCUDA SSNP
```
module load miniconda cuda/12.2
conda create -n ssnp python=3.10
conda activate ssnp
pip install "git+https://github.com/bu-cisl/SSNP-IDT#subdirectory=ssnp_pkg"
```

### MAKE SURE GPU ACCESSIBLE AT THIS POINT
Ensure `ssnp` conda environment is activated always

### (Optional) Test Functionality of PyCUDA model
```
git clone https://github.com/bu-cisl/SSNP-IDT.git
cd SSNP-IDT
python forward_model.py
```

### Run simple comparison test
Go to root directory of the entire wgpu-ssnp repo
```
python benchmark/benchmark.py
```
Forward times for the WGPU and PyCUDA models should be printed

Early results reveal that for small volumes (ex. 16x16x16), WGPU is faster:
```
WebGPU forward time:   111.00 ms
PyCUDA forward time:   231.07 ms
```

But for much larger volumes (ex. 256x256x256), PyCUDA is significantly faster:
```
WebGPU forward time:   2571.00 ms
PyCUDA forward time:   231.89 ms
```

This makes sense since WGPU's novelty here lies in its ability to be GPU agnostic, browser-compatible, and also run on CPUs if necessary.