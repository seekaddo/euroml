# Import torch
import torch

# Print the PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
if torch.cuda.is_available():
    # Print the CUDA device name and version
    print("CUDA device:", torch.cuda.get_device_name())
    print("CUDA version:", torch.version.cuda)
    print("CUDA mem:", torch.cuda.get_device_properties(0).total_memory)
    print("CUDA :", torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3)
else:
    # Print a message that CUDA is not available
    print("CUDA is not available")
