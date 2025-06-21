import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from py.ssnp_model import SSNPBeam

ANGLE = [-0.49, 0.33]
RES = (0.1, 0.1, 0.1)
NA = 0.65
INTENSITY = True
TIFF_PATH = "pycuda.tiff"

def main():
    # Load TIFF file
    print(f"Loading {TIFF_PATH}")
    input_tensor = tifffile.imread(TIFF_PATH)
    
    print(input_tensor.dtype, input_tensor.max())
    # Convert to float32
    if input_tensor.dtype != np.float32:
        # change this:
        input_tensor = 0.01 * input_tensor.astype(np.float32) / 0xFFFF
    
    print(f"Shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"Value range: {input_tensor.min():.4f} to {input_tensor.max():.4f}")

    model = SSNPBeam()
    model.angles = torch.nn.Parameter(torch.tensor(ANGLE, dtype=torch.float32))
    model.res = RES
    model.na = NA
    model.intensity = INTENSITY

    # Convert to tensor and process
    input_tensor = torch.tensor(input_tensor)
    output = model(input_tensor).detach().numpy()
    output = np.squeeze(output)  

    plt.imshow(output)
    plt.colorbar()
    plt.savefig('output.png')
    plt.show()
    
    print("Output saved to 'output.png'")

if __name__ == "__main__":
    main()