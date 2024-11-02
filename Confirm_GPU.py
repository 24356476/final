import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current device ID and the name of the GPU
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU instead.")
