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

### MAKE SURE A GPU IS AVAILABLE AT THIS POINT
Ensure `ssnp` conda environment is activated always

### (Optional) Test Functionality of PyCUDA model
```
git clone https://github.com/bu-cisl/SSNP-IDT.git
cd SSNP-IDT
python forward_model.py
```

### Generate comparison results
Go to root directory of the entire wgpu-ssnp repo and run:
```
python benchmark/ssnp/benchmark.py
```

Repo's README.md summarizes results