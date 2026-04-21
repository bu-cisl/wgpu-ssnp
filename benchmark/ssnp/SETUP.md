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
pip install -r requirements.txt
```

### MAKE SURE A GPU IS AVAILABLE AT THIS POINT
Ensure `ssnp` conda environment is activated always

### Generate comparison results
Go to root directory of wgpu-idt repo and run:
```
python benchmark/ssnp/benchmark.py
```

If `benchmark/ssnp/build/benchmark` does not exist yet, the Python script will configure and build it automatically.

Repo's README.md summarizes results
