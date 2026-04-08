import struct
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch

from python.ssnp_model import SSNPBeam

SHAPE = (64, 100, 100)
RES = (0.2, 0.2, 0.2)
NA = 0.65
N0 = 1.33

MAX_ITERATIONS = 100
LEARNING_RATE = 5e-1
ABS_TOL = 1e-10
REL_TOL = 1e-6
PRINT_EVERY = 1
VERBOSE = True


def load_tensor_bin(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        depth, height, width = struct.unpack("iii", f.read(12))
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape((depth, height, width))


def build_angles() -> np.ndarray:
    angles = [[0.0, 0.0]]
    angle_count = 15
    radius = 0.2
    for i in range(angle_count):
        theta = 2.0 * np.pi * i / angle_count
        angles.append([radius * np.cos(theta), radius * np.sin(theta)])
    return np.asarray(angles, dtype=np.float32)


def create_target_volume(shape):
    volume = np.zeros(shape, dtype=np.float32)
    depth, height, width = shape
    radius = 0.20 * min(depth, height, width)
    z, y, x = np.indices(shape, dtype=np.float32)
    dz = z - 0.5 * depth
    dy = y - 0.5 * height
    dx = x - 0.5 * width
    mask = dx * dx + dy * dy + dz * dz <= radius * radius
    volume[mask] = 0.02
    return volume


def create_initial_volume(shape):
    return np.zeros(shape, dtype=np.float32)


def make_model(angles):
    model = SSNPBeam(angles=len(angles))
    model.angles = torch.nn.Parameter(torch.tensor(angles, dtype=torch.float32))
    model.res = RES
    model.na = NA
    model.intensity = True
    return model


def forward_stack(volume: np.ndarray, angles: np.ndarray) -> np.ndarray:
    model = make_model(angles)
    with torch.no_grad():
        output = model(torch.tensor(volume, dtype=torch.float32)).detach().cpu().numpy()
    return output.astype(np.float32)


def measurement_mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def save_projection_png(filename: str, volume: np.ndarray):
    projection = forward_stack(volume, np.array([[0.0, 0.0]], dtype=np.float32))[0]
    plt.imshow(projection, cmap="magma")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_reconstruction_input(filename: str, measured: np.ndarray, initial: np.ndarray, angles: np.ndarray):
    depth, height, width = initial.shape
    num_angles = measured.shape[0]
    with open(filename, "wb") as f:
        f.write(struct.pack("iiii", depth, height, width, num_angles))
        f.write(struct.pack("fff", *RES))
        f.write(struct.pack("f", NA))
        f.write(struct.pack("f", N0))
        f.write(struct.pack("i", MAX_ITERATIONS))
        f.write(struct.pack("f", LEARNING_RATE))
        f.write(struct.pack("f", ABS_TOL))
        f.write(struct.pack("f", REL_TOL))
        f.write(struct.pack("i", PRINT_EVERY))
        f.write(struct.pack("I", 1 if VERBOSE else 0))
        f.write(angles.astype(np.float32).tobytes())
        f.write(measured.astype(np.float32).tobytes())
        f.write(initial.astype(np.float32).tobytes())


if __name__ == "__main__":
    ANGLES = build_angles()
    target = create_target_volume(SHAPE)
    initial = create_initial_volume(SHAPE)

    measured = forward_stack(target, ANGLES)
    initial_prediction = forward_stack(initial, ANGLES)

    save_reconstruction_input("reconstruct_input.bin", measured, initial, ANGLES)

    result = subprocess.run(
        ["./build/optics_sim", "ssnp_reconstruct", "reconstruct_input.bin", "reconstruct_output.bin"],
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("C++ reconstruction failed.")

    reconstructed = load_tensor_bin("reconstruct_output.bin")
    reconstructed_prediction = forward_stack(reconstructed, ANGLES)

    save_projection_png("original.png", target)
    save_projection_png("reconstruction.png", reconstructed)

    print(
        "Reconstruction measurement MSE:",
        f"{measurement_mse(initial_prediction, measured):.10e} -> {measurement_mse(reconstructed_prediction, measured):.10e}",
    )
