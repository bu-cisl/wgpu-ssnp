from py.ssnp_model import scatter_factor, diffract, c_gamma, binary_pupil, merge_prop, split_prop #, tilt
import subprocess
import numpy as np
import torch

ROWS, COLS = 512, 512

# ------------- Helper Functions -----------------
def compute_summary(arr):
    """
    Compute summary statistics for a numpy array.
    For complex arrays, compute on the magnitudes.
    Returns a tuple: (count, mean, std, min, max)
    """
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    count = arr.size
    mean = np.mean(arr)
    std = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    return count, mean, std, min_val, max_val

def compare_summary(py_sum, cpp_sum, label, rtol=1e-4, atol=1e-2):
    """
    Compare two summary tuples.
    """
    py_count, py_mean, py_std, py_min, py_max = py_sum
    cpp_count, cpp_mean, cpp_std, cpp_min, cpp_max = cpp_sum
    success = True
    if py_count != cpp_count:
        print(f"{label} count mismatch: Python {py_count}, C++ {cpp_count}")
        success = False
    for name, py_val, cpp_val in zip(["mean", "std", "min", "max"],
                                       [py_mean, py_std, py_min, py_max],
                                       [cpp_mean, cpp_std, cpp_min, cpp_max]):
        if not np.isclose(py_val, cpp_val, rtol=rtol, atol=atol):
            print(f"{label} {name} mismatch: Python {py_val}, C++ {cpp_val}")
            success = False
    return success

def parse_cpp_summaries(output_lines):
    """
    Parse lines from the C++ executable.
    Expected format for each line:
      LABEL: count=<n> mean=<mean> std=<std> min=<min> max=<max>
    Returns a dictionary mapping LABEL to summary tuple.
    """
    summaries = {}
    for line in output_lines:
        if ':' not in line:
            continue
        label, rest = line.split(":", 1)
        parts = rest.split()
        summary = {}
        for part in parts:
            if '=' in part:
                key, val = part.split('=')
                summary[key] = float(val)
        summaries[label.strip()] = (int(summary.get("count", 0)),
                                    summary.get("mean", 0.0),
                                    summary.get("std", 0.0),
                                    summary.get("min", 0.0),
                                    summary.get("max", 0.0))
    return summaries

# ------------------- Set Up Test Inputs -------------------
print("Phase: Creating test inputs in PyTorch")
rows, cols = ROWS, COLS
# Create a 1D vector and reshape to a matrix (all tests use this same input)
n_scatter = torch.rand((rows*cols,)) * 10.0
matrix = n_scatter.reshape(rows, cols)
# Scatter_factor uses the 1D vector.
scatter_py = scatter_factor(n_scatter).cpu().numpy()
scatter_summary = compute_summary(scatter_py)

# c_gamma test uses the shape.
c_gamma_py = c_gamma((0.1, 0.1, 0.1), (rows, cols), device='cpu').squeeze(0).cpu().numpy()
c_gamma_summary = compute_summary(c_gamma_py)

# For complex tests, use the same matrix converted to complex (0 imag).
complex_matrix = matrix.to(torch.complex64)

# Diffract test: use same input for uf and ub.
diffract_uf_py, diffract_ub_py = diffract(complex_matrix, complex_matrix, res=(0.1,0.1,0.1), dz=1.0)
diffract_uf_summary = compute_summary(diffract_uf_py.cpu().numpy())
diffract_ub_summary = compute_summary(diffract_ub_py.cpu().numpy())

# Binary pupil test (only shape matters).
bp_py = binary_pupil((rows, cols), na=0.9, res=(0.1,0.1,0.1), device='cpu').cpu().numpy().astype(np.int32)
binary_summary = (bp_py.size, bp_py.mean(), bp_py.sum(), bp_py.min(), bp_py.max()) # intentionally std->sum

# Tilt test.
# angles = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float32)
# tilt_py = tilt((rows, cols), angles, NA=0.5, res=(0.1,0.1,0.1), trunc=False, device='cpu')
# tilt_summary = compute_summary(tilt_py.cpu().numpy())

# Merge_prop test.
merge_uf_py, merge_ub_py = merge_prop(complex_matrix, complex_matrix, res=(0.1,0.1,0.1))
merge_uf_summary = compute_summary(merge_uf_py.cpu().numpy())
merge_ub_summary = compute_summary(merge_ub_py.cpu().numpy())

# Split_prop test.
split_uf_py, split_ub_py = split_prop(complex_matrix, complex_matrix, res=(0.1,0.1,0.1))
split_uf_summary = compute_summary(split_uf_py.cpu().numpy())
split_ub_summary = compute_summary(split_ub_py.cpu().numpy())

# ------------------- Write Input File for C++ -------------------
print("Phase: Writing input matrix file for C++ tests")
input_filename = "input_matrix.txt"
with open(input_filename, "w") as f:
    f.write(f"{rows} {cols}\n")
    matrix_np = n_scatter.reshape(rows, cols).cpu().numpy()
    for r in range(rows):
        row_str = " ".join(f"{val:.8f}" for val in matrix_np[r])
        f.write(row_str + "\n")

# ------------------- Build and Run C++ Executable -------------------
print("Phase: Building and running WGPU C++ tests")
try:
    subprocess.run(["cmake", "-B", "build", "-S", "."], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", "build"], check=True, stdout=subprocess.DEVNULL)
    result = subprocess.run(["./build/ssnp_cpp", input_filename],
                            capture_output=True, text=True, check=True)
    cpp_output_lines = result.stdout.splitlines()
except subprocess.CalledProcessError as e:
    print("Error during build or execution of the WGPU executable.")
    print(e)
    exit(1)

cpp_summaries = parse_cpp_summaries(cpp_output_lines)

# ------------------- Compare Summaries -------------------
print("Phase: Comparing outputs")
tests = [
    ("SCATTER_FACTOR", scatter_summary),
    ("C_GAMMA", c_gamma_summary),
    ("DIFRACT_UF", diffract_uf_summary),
    ("DIFRACT_UB", diffract_ub_summary),
    ("BINARY_PUPIL", binary_summary),
    # ("TILT", tilt_summary),
    ("MERGE_PROP_UF", merge_uf_summary),
    ("MERGE_PROP_UB", merge_ub_summary),
    ("SPLIT_PROP_UF", split_uf_summary),
    ("SPLIT_PROP_UB", split_ub_summary)
]

all_pass = True
for label, py_sum in tests:
    if label not in cpp_summaries:
        print(f"{label}: No C++ output found")
        all_pass = False
        continue
    cpp_sum = cpp_summaries[label]
    if compare_summary(py_sum, cpp_sum, label):
        print(f"{label} outputs match: PASS")
    else:
        print(f"{label} outputs match: FAIL")
        all_pass = False

if all_pass:
    print("RESULT: All function outputs match within tolerance.")
else:
    print("RESULT: Some function outputs did not match within tolerance.")