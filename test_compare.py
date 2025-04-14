from py.ssnp_model import scatter_factor, diffract, c_gamma, binary_pupil, merge_prop, split_prop
import subprocess
import numpy as np
import torch

# Settings
ROWS, COLS = 2048, 2048
tolerance = 1e-1

# ------------------------------------------------------------------------------
# Helper: Compare two numpy arrays element-wise.
# If the C++ (cpp_arr) appears to be missing interleaved imaginary parts (i.e. its length
# is exactly half of the Python array length), then expand the C++ array by inserting zeros
# for the imaginary parts.
# Returns (num_mismatches, total_count) or None if sizes are incompatible.
# ------------------------------------------------------------------------------
def compare_arrays(py_arr, cpp_arr, tol):
    if len(py_arr) != len(cpp_arr):
        # Check if C++ appears to be real-only (half the length)
        if len(cpp_arr) * 2 == len(py_arr):
            new_cpp = np.empty(len(py_arr), dtype=cpp_arr.dtype)
            new_cpp[0::2] = cpp_arr
            new_cpp[1::2] = 0.0
            cpp_arr = new_cpp
        elif len(py_arr) * 2 == len(cpp_arr):
            # Alternatively, if Python result is half length,
            # expand Python result by inserting zeros.
            new_py = np.empty(len(cpp_arr), dtype=py_arr.dtype)
            new_py[0::2] = py_arr
            new_py[1::2] = 0.0
            py_arr = new_py
        else:
            print(f"Size mismatch: Python size {len(py_arr)}, C++ size {len(cpp_arr)}")
            return None
    diff = np.abs(py_arr - cpp_arr)
    mismatches = np.sum(diff > tol)
    return mismatches, len(py_arr)

# ------------------------------------------------------------------------------
# Helper: Parse the output from the C++ executable.
# Expected format:
#   LABEL:
#   <number of elements>
#   <space-separated list of numbers>
#
# Returns a dictionary mapping each label to a numpy array (as raw floats).
# For labels that are meant to be complex, we leave the raw interleaved float array.
# ------------------------------------------------------------------------------
def parse_cpp_arrays(output_lines):
    cpp_data = {}
    i = 0
    while i < len(output_lines):
        line = output_lines[i].strip()
        if line.endswith(":"):
            label = line[:-1].strip()
            i += 1
            if i >= len(output_lines):
                break
            try:
                count = int(output_lines[i].strip())
            except Exception as e:
                print(f"Error parsing count for label {label}: {e}")
                break
            i += 1
            if i >= len(output_lines):
                break
            data_line = output_lines[i].strip()
            parts = data_line.split()
            if len(parts) != count:
                print(f"Warning: For {label}, expected {count} elements, got {len(parts)}")
            arr = np.array([float(x) for x in parts])
            cpp_data[label] = arr
        i += 1
    return cpp_data

# ------------------------------------------------------------------------------
# Helper: For a complex numpy array (obtained from our PyTorch computations),
# convert to an interleaved float array.
# ------------------------------------------------------------------------------
def to_interleaved(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        real_flat = np.real(arr).ravel()
        imag_flat = np.imag(arr).ravel()
        return np.column_stack((real_flat, imag_flat)).flatten()
    else:
        return arr.ravel()

# ------------------------------------------------------------------------------
# Create test inputs in PyTorch and compute function outputs.
# ------------------------------------------------------------------------------
print("Phase: Creating test inputs in PyTorch")
rows, cols = ROWS, COLS

# Create a 1D random vector and reshape to matrix.
n_scatter = torch.rand((rows * cols,)) * 10.0
matrix = n_scatter.reshape(rows, cols)

# Compute outputs.
scatter_py = scatter_factor(n_scatter).cpu().numpy().ravel()
c_gamma_py = c_gamma((0.1, 0.1, 0.1), (rows, cols), device='cpu').squeeze(0).cpu().numpy().ravel()

# For complex tests, convert the matrix to a complex tensor.
complex_matrix = matrix.to(torch.complex64)
diffract_uf_py, diffract_ub_py = diffract(complex_matrix, complex_matrix, res=(0.1, 0.1, 0.1), dz=1.0)
bp_py = binary_pupil((rows, cols), na=0.9, res=(0.1, 0.1, 0.1), device='cpu').cpu().numpy().astype(np.int32).ravel()
merge_uf_py, merge_ub_py = merge_prop(complex_matrix, complex_matrix, res=(0.1, 0.1, 0.1))
split_uf_py, split_ub_py = split_prop(complex_matrix, complex_matrix, res=(0.1, 0.1, 0.1))

# Prepare a dictionary mapping labels to Python results.
# For those intended to be complex, convert to interleaved float arrays.
py_results = {
    "SCATTER_FACTOR": scatter_py,
    "C_GAMMA": to_interleaved(c_gamma_py),  # Even if c_gamma is real, we convert (won't change size)
    "DIFRACT_UF": to_interleaved(diffract_uf_py.cpu().numpy()),
    "DIFRACT_UB": to_interleaved(diffract_ub_py.cpu().numpy()),
    "BINARY_PUPIL": bp_py,
    "MERGE_PROP_UF": to_interleaved(merge_uf_py.cpu().numpy()),
    "MERGE_PROP_UB": to_interleaved(merge_ub_py.cpu().numpy()),
    "SPLIT_PROP_UF": to_interleaved(split_uf_py.cpu().numpy()),
    "SPLIT_PROP_UB": to_interleaved(split_ub_py.cpu().numpy())
}

# ------------------------------------------------------------------------------
# Write the input matrix file for the C++ tests.
# ------------------------------------------------------------------------------
print("Phase: Writing input matrix file for C++ tests")
input_filename = "input_matrix.txt"
with open(input_filename, "w") as f:
    f.write(f"{rows} {cols}\n")
    matrix_np = n_scatter.reshape(rows, cols).cpu().numpy()
    for r in range(rows):
        row_str = " ".join(f"{val:.8f}" for val in matrix_np[r])
        f.write(row_str + "\n")

# ------------------------------------------------------------------------------
# Build and run the C++ executable.
# ------------------------------------------------------------------------------
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

cpp_results = parse_cpp_arrays(cpp_output_lines)

# ------------------------------------------------------------------------------
# Compare each function's outputs element-by-element.
# ------------------------------------------------------------------------------
print("Phase: Comparing outputs")
all_pass = True
for label, py_arr in py_results.items():
    if label not in cpp_results:
        print(f"{label}: No C++ output found")
        all_pass = False
        continue
    cpp_arr = cpp_results[label]
    py_flat = py_arr.ravel()
    cpp_flat = cpp_arr.ravel()
    ret = compare_arrays(py_flat, cpp_flat, tolerance)
    if ret is None:
        all_pass = False
    else:
        mismatches, total = ret
        print(f"{label}: {mismatches} out of {total} elements mismatched (tolerance {tolerance})")
        if mismatches > 0:
            all_pass = False

if all_pass:
    print("RESULT: All outputs match within tolerance.")
else:
    print("RESULT: Some outputs did not match within tolerance.")
