import numpy as np
import subprocess
import struct
import torch
from py.ssnp_model import SSNPBeam

SLICES = 5
ROWS = 512
COLS = 512
TOL = 1e-4

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
        print("C++ Error:", result.stderr)
        raise RuntimeError("C++ execution failed.")
    return load_tensor_bin(output_path)

def run_python_model(input_tensor):
    tensor_input = torch.tensor(input_tensor, dtype=torch.float32)
    model = SSNPBeam(angles=10)
    with torch.no_grad():
        return model(tensor_input).numpy()

def compare_outputs(py_output, cpp_output, rtol=TOL):
    denom = np.maximum(np.abs(py_output), np.abs(cpp_output))
    denom[denom == 0] = 1e-12

    rel_diff = np.abs(py_output - cpp_output) / denom
    mask = rel_diff > rtol

    num_diffs = np.count_nonzero(mask)
    total = py_output.size

    print(f"🔎 Points differing beyond rtol={rtol}: {num_diffs} / {total}")

    if num_diffs > 0:
        max_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
        py_val = py_output[max_idx]
        cpp_val = cpp_output[max_idx]
        worst_diff = rel_diff[max_idx]

        print("❌ Worst offender:")
        print(f"  Python value: {py_val}")
        print(f"  C++ value   : {cpp_val}")
        print(f"  Rel. diff   : {worst_diff:.6e}")
    else:
        print("✅ All outputs match within tolerance.")

if __name__ == "__main__":
    input_tensor = generate_input((SLICES, ROWS, COLS)) 

    print("Running C++ model...")
    cpp_output = run_cpp_model(input_tensor)

    print("Running Python model...")
    py_output = run_python_model(input_tensor)

    print("Comparing outputs...")
    compare_outputs(py_output, cpp_output)
