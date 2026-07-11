import torch

# check if MPS is available
if torch.backends.mps.is_available():
    # create MPS device
    mps_device = torch.device("mps")
    # create a tensor on MPS device
    mps_tensor = torch.ones((3, 3), device=mps_device)
    print("MPS tensor:", mps_tensor)
else:
    print("MPS is not available on this system.")
