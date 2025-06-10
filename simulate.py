import numpy as np
import torch
import struct
import matplotlib.pyplot as plt
import tifffile
from py.ssnp_model import SSNPBeam

SHAPE = (100, 256, 256)
ANGLE = [-0.49,0.33]
RES = (.1,.1,.1)
NA = .65
INTENSITY = True
TYPE = "tiff" # tiff or bin

def save_tensor(tensor: np.ndarray, type="bin"):
    assert tensor.ndim == 3
    D, H, W = tensor.shape
    if type=="bin":
        with open("input.bin", "wb") as f:
            f.write(struct.pack("iii", D, H, W))
            f.write(tensor.astype(np.float32).tobytes())
    if type=="tiff":
        tifffile.imwrite("input.tiff", tensor.astype(np.float32))

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

if __name__ == "__main__":
    # Initialize model
    model = SSNPBeam()
    model.angles = torch.nn.Parameter(torch.tensor(ANGLE, dtype=torch.float32))
    model.res = RES
    model.na = NA
    model.intensity = INTENSITY

    # Initialize input/run model
    input = torch.tensor(create_sphere(SHAPE), dtype=torch.float32)
    output = model(input).detach().numpy()

    # Save output
    output = np.squeeze(output, axis=0)
    plt.imshow(output)
    plt.colorbar()
    plt.savefig('sample.png')
    plt.close()

    # Save input tensor for site testing
    save_tensor(input.detach().numpy(), type=TYPE)
    print("Expected output saved to 'sample.png'")
    print("Input file to upload to site saved as 'input.tiff'")