import os
import numpy as np
import subprocess
import struct
import torch
import matplotlib.pyplot as plt
import pyvista as pv
from python.ssnp_model import SSNPBeam
from python.bpm_model import BPMBeam
import pytest

SLICES = 32
ROWS = 128
COLS = 128
TOL = 1e-4 # rtol
IMAGE_NAME = None # None if no save
MODEL = "bpm" # for local testing

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

    distance_squared = ((x - center[2])**2 +
                        (y - center[1])**2 +
                        (z - center[0])**2)

    mask = distance_squared <= radius**2

    volume = np.full(shape, value_outside, dtype=np.float32)
    volume[mask] = value_inside
    return volume

def generate_input(shape=(3, 128, 128)) -> np.ndarray:
    return create_sphere(shape)

def run_cpp_model(model, input_path="input.bin", output_path="output.bin"):
    result = subprocess.run(["./build/optics_sim", model, input_path, output_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("C++ Error:", result.stderr, result.stdout)
        raise RuntimeError("C++ execution failed.")
    return load_tensor_bin(output_path)

def run_python_model(input_tensor, model_name: str):
    tensor_input = torch.tensor(input_tensor, dtype=torch.float32)
    if model_name == "ssnp":
        model = SSNPBeam(angles=1)
    elif model_name == "bpm":
        model = BPMBeam(angles=1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    with torch.no_grad():
        return model(tensor_input).numpy()

def save_output_as_png(output, filename):
    output = np.squeeze(output, axis=0)
    array = np.array(output)
    plt.imshow(array)
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def save_input_as_png(input, filename):
    image = input.transpose(1, 2, 0)

    grid = pv.ImageData()
    grid.dimensions = np.array(image.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data["values"] = image.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="values", cmap="viridis")
    plotter.show(screenshot=f"{filename}.png")

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
        return False
    else:
        print("✅ All outputs match within specified tolerances.")
        return True

def run_model_test(model):
    print("Building C++ model...")
    subprocess.run(["cmake", "-B", "build", "-S", "."])
    subprocess.run(["cmake", "--build", "build"])

    print("Generating input...")
    input_tensor = generate_input((SLICES, ROWS, COLS))
    save_tensor_bin("input.bin", input_tensor)

    print(f"Running C++ model ({model})...")
    cpp_output = run_cpp_model(model)

    print(f"Running Python model ({model})...")
    py_output = run_python_model(input_tensor, model)

    if IMAGE_NAME is not None:
        image_folder = f"{ROWS}x{COLS}x{SLICES}"
        output_dir = f"images/{model}_out/{image_folder}/"
        print(f"Saving images to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        save_input_as_png(input_tensor, f"{output_dir}/input")
        save_output_as_png(cpp_output, f"{output_dir}/cpp_{IMAGE_NAME}.png")
        save_output_as_png(py_output, f"{output_dir}/py_{IMAGE_NAME}.png")

    print("Comparing outputs...")
    assert compare_outputs(py_output, cpp_output), "Outputs do not match within tolerance."

    # Cleanup temporary files
    for file in ["input.bin", "output.bin"]:
        if os.path.exists(file):
            os.remove(file)

# For pytest
def test_ssnp():
    run_model_test("ssnp")

def test_bpm():
    run_model_test("bpm")

if __name__ == "__main__":
    run_model_test(MODEL)
