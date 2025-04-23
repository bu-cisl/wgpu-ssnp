import torch
from ssnp_model import SNNPBeam

if __name__ == "__main__":
    # Create a dummy RI volume (depth=3, 16x16 pixels)
    n = torch.ones((3, 16, 16), dtype=torch.float32)
    
    # Create dummy tilt angles
    angles = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float32)

    # Initialize the model
    model = SNNPBeam()

    # Run the model
    output = model(n, angles)

    # Print the output shape and a small patch of values
    print("Output shape:", output.shape)
    print("Output sample:", output[0:3, 0:3])