import numpy as np
import subprocess
import struct
import torch
import matplotlib.pyplot as plt
from py.ssnp_model import SSNPBeam

SLICES = 200
ROWS = 1024
COLS = 1024
TOL = 1e-4 # rtol
IMAGE_NAME = None # None if no save

def save_tensor_bin(filename, tensor: np.ndarray):
    assert tensor.ndim == 3
    D, H, W = tensor.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("iii", D, H, W))
        f.write(tensor.astype(np.float32).tobytes())

def load_tensor_bin(filename) -> np.ndarray:
    with open(filename, "rb") as f:
        D, H, W = struct.unpack("iii", f.read(12))
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape((D, H, W))

def create_sphere(shape, radius_fraction=0.25, value_inside=0.01, value_outside=0.0):
    z, y, x = np.indices(shape)
    center = [s // 2 for s in shape]
    radius = int(min(shape) * radius_fraction)
    
    # Compute squared distance from center
    distance_squared = ((x - center[2])**2 + 
                        (y - center[1])**2 + 
                        (z - center[0])**2)
    
    # Mask for points inside the sphere
    mask = distance_squared <= radius**2

    # Create volume
    volume = np.full(shape, value_outside, dtype=np.float32)
    volume[mask] = value_inside
    
    return volume

def generate_input(shape=(3, 128, 128)) -> np.ndarray:
    return create_sphere(shape)

def run_cpp_model(input_tensor, input_path="input.bin", output_path="output.bin"):
    save_tensor_bin(input_path, input_tensor)
    result = subprocess.run(["./build/ssnp_cpp", input_path, output_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("C++ Error:", result.stderr, result.stdout)
        raise RuntimeError("C++ execution failed.")
    return load_tensor_bin(output_path)

def run_python_model(input_tensor):
    tensor_input = torch.tensor(input_tensor, dtype=torch.float32)
    model = SSNPBeam(angles=1)
    with torch.no_grad():
        return model(tensor_input).numpy()

def save_output_as_png(output, filename):
    output = np.squeeze(output, axis=0)
    array = np.array(output)
    plt.imshow(array)
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def compare_outputs(py_output, cpp_output, rtol=TOL, atol=1e-4):
    abs_diff = np.abs(py_output - cpp_output)
    denom = np.maximum(np.abs(py_output), np.abs(cpp_output))
    denom[denom == 0] = 1e-12  # avoid division by zero
    rel_diff = abs_diff / denom

    mask = (rel_diff > rtol) & (abs_diff > atol)

    num_diffs = np.count_nonzero(mask)
    total = py_output.size

    print(f"🔎 Points differing beyond rtol={rtol} and atol={atol}: {num_diffs} / {total}")

    if num_diffs > 0:
        max_idx = np.unravel_index(np.argmax(rel_diff * mask), rel_diff.shape)
        py_val = py_output[max_idx]
        cpp_val = cpp_output[max_idx]
        worst_diff = rel_diff[max_idx]
        worst_abs_diff = abs_diff[max_idx]

        print("❌ Worst offender:")
        print(f"  Python value  : {py_val}")
        print(f"  C++ value     : {cpp_val}")
        print(f"  Rel. diff     : {worst_diff:.6e}")
        print(f"  Abs. diff     : {worst_abs_diff:.6e}")
    else:
        print("✅ All outputs match within specified tolerances.")

if __name__ == "__main__":
    input_tensor = generate_input((SLICES, ROWS, COLS)) 

    print("Running C++ model...")
    cpp_output = run_cpp_model(input_tensor)

    print("Running Python model...")
    py_output = run_python_model(input_tensor)

    if IMAGE_NAME is not None:
        image_folder = f"{ROWS}x{COLS}x{SLICES}"
        output_dir = f"images/{image_folder}/"
        save_output_as_png(cpp_output, f"{output_dir}/cpp_{IMAGE_NAME}.png")
        save_output_as_png(py_output, f"{output_dir}/py_{IMAGE_NAME}.png")
        print(f"Saved images to {output_dir}")

    print("Comparing outputs...")
    compare_outputs(py_output, cpp_output)
