import numpy as np
import subprocess
import struct
import torch
from py.ssnp_model import SSNPBeam

SLICES = 100
ROWS = 512
COLS = 512
ANGLE_COUNT = 1 # note - need to manually change on src/main.cpp line 81 for now
TOL = 1e-4 # rtol

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

def generate_input(shape=(3, 128, 128)) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)

def run_cpp_model(input_tensor, input_path="input.bin", output_path="output.bin"):
    save_tensor_bin(input_path, input_tensor)
    result = subprocess.run(["./build/ssnp_cpp", input_path, output_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("C++ Error:", result.stderr, result.stdout)
        raise RuntimeError("C++ execution failed.")
    return load_tensor_bin(output_path)

def run_python_model(input_tensor):
    tensor_input = torch.tensor(input_tensor, dtype=torch.float32)
    model = SSNPBeam(angles=ANGLE_COUNT)
    with torch.no_grad():
        return model(tensor_input).numpy()

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

    print("Comparing outputs...")
    compare_outputs(py_output, cpp_output)
