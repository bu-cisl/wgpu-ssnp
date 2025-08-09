import subprocess
import numpy as np
import matplotlib.pyplot as plt

# SSNP-IDT imports
from pycuda import gpuarray
from pycuda import driver as cuda
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
    beam = BeamArray(u_plane)

    # Normal incidence
    beam.forward = u_plane
    beam.backward = 0

    # CUDA events for timing
    start = cuda.Event()
    end = cuda.Event()
    
    start.record()
    beam.merge_prop()
    beam.ssnp(1, n_gpu)
    beam.ssnp(-len(n_gpu) / 2)
    beam.backward = None
    beam.binary_pupil(1.0001 * 0.65)
    beam.forward.get()
    end.record()
    end.synchronize()
    
    elapsed = start.time_till(end)  # ms

    return elapsed

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Graph 1: Spatial size (H×W) vs time - D fixed at 64
    print("Benchmarking spatial size scaling...")
    sizes = [64, 128, 256, 512, 1024, 2048]
    D_fixed = 64
    
    webgpu_spatial = []
    pycuda_spatial = []
    
    for size in sizes:
        print(f"Testing {D_fixed}x{size}x{size}...")
        webgpu_ms = run_webgpu(D_fixed, size, size)
        pycuda_ms = run_pycuda(D_fixed, size, size)
        webgpu_spatial.append(webgpu_ms)
        pycuda_spatial.append(pycuda_ms)
        print(f"  WebGPU: {webgpu_ms:.2f} ms, PyCUDA: {pycuda_ms:.2f} ms")
    
    # Graph 2: Number of slices (D) vs time - H,W fixed at 128
    print("\nBenchmarking depth scaling...")
    depths = [16, 32, 64, 128, 256, 512]
    HW_fixed = 128
    
    webgpu_depth = []
    pycuda_depth = []
    
    for depth in depths:
        print(f"Testing {depth}x{HW_fixed}x{HW_fixed}...")
        webgpu_ms = run_webgpu(depth, HW_fixed, HW_fixed)
        pycuda_ms = run_pycuda(depth, HW_fixed, HW_fixed)
        webgpu_depth.append(webgpu_ms)
        pycuda_depth.append(pycuda_ms)
        print(f"  WebGPU: {webgpu_ms:.2f} ms, PyCUDA: {pycuda_ms:.2f} ms")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spatial size plot
    ax1.plot(sizes, webgpu_spatial, 'o-', label='WebGPU', linewidth=2)
    ax1.plot(sizes, pycuda_spatial, 's-', label='PyCUDA', linewidth=2)
    ax1.set_xlabel('Spatial Size (H×W)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Spatial Size vs Time (D={D_fixed})')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Depth plot
    ax2.plot(depths, webgpu_depth, 'o-', label='WebGPU', linewidth=2)
    ax2.plot(depths, pycuda_depth, 's-', label='PyCUDA', linewidth=2)
    ax2.set_xlabel('Number of Slices (D)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title(f'Depth vs Time (H×W={HW_fixed}×{HW_fixed})')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark/ssnp/benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBenchmark complete. Results saved to benchmark_results.png")
