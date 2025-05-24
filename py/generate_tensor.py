import numpy as np

# Example tensor (shape D=3, H=3, W=3)
tensor = np.array([
    [
        [1.23, 4.56, 7.89],
        [2.34, 5.67, 8.90],
        [3.45, 6.78, 9.01]
    ],
    [
        [0.12, 3.45, 6.78],
        [9.87, 6.54, 3.21],
        [1.11, 2.22, 3.33]
    ],
    [
        [7.77, 8.88, 9.99],
        [4.44, 5.55, 6.66],
        [0.01, 1.02, 2.03]
    ]
], dtype=np.float32)

D, H, W = tensor.shape

# Save with header (D, H, W as int32) followed by data
with open("tensor.bin", "wb") as f:
    # Write header (3 int32s)
    f.write(np.array([D, H, W], dtype=np.int32).tobytes())
    # Write tensor data
    f.write(tensor.tobytes())

print(f"Saved tensor.bin with D={D}, H={H}, W={W}")