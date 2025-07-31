import subprocess
import numpy as np
from time import perf_counter as time

# SSNP-IDT imports
from pycuda import gpuarray
import ssnp
from ssnp import BeamArray

def run_webgpu(D, H, W):
    """Run the WebGPU benchmark subprocess."""
    result = subprocess.run(["./benchmark/build/benchmark", str(D), str(H), str(W)],
                            capture_output=True, text=True)
    
    if result.returncode != 0:
        print("WebGPU Benchmark Error:", result.stderr)
        raise RuntimeError("WebGPU benchmark execution failed.")
    
    return int(result.stdout.strip())  # milliseconds

def run_pycuda(D, H, W):
    """Run the PyCUDA forward pass timing."""
    # Volume in CPU → GPU
    n_cpu = np.zeros((D, H, W), dtype=np.float64)
    n_gpu = gpuarray.to_gpu(n_cpu)

    # Beam setup - default params same as wgpu + angle [0,0]
    ssnp.config.res = (0.1, 0.1, 0.1)
    ssnp.config.na = 0.65
    ssnp.config.n0 = 1.33
    u_plane = gpuarray.to_gpu(np.ones((H, W), dtype=np.complex128))
    beam = BeamArray(u_plane, total_ops=D)

    # Normal incidence
    beam.forward = u_plane
    beam.backward = 0

    t0 = time()
    beam.merge_prop()
    _ = beam.forward.get()  # force evaluation
    elapsed = (time() - t0) * 1000  # ms

    return elapsed

if __name__ == "__main__":
    D, H, W = 16, 16, 16

    print(f"Benchmarking forward pass on volume {D}x{H}x{W}...\n")

    webgpu_ms = run_webgpu(D, H, W)
    print(f"WebGPU forward time:   {webgpu_ms:.2f} ms")

    pycuda_ms = run_pycuda(D, H, W)
    print(f"PyCUDA forward time:   {pycuda_ms:.2f} ms")
