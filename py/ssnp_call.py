import torch
from ssnp_model import SSNPBeam

if __name__ == "__main__":
    # Create a dummy RI volume (depth=3, 16x16 pixels)
    n = torch.ones((3, 16, 16), dtype=torch.float32)

    # Initialize the model
    model = SSNPBeam()

    # Run the model
    output = model(n)

    # Print the output shape and a small patch of values
    print("Output shape:", output.shape)
    print("Output sample:", output[0:3, 0:3])