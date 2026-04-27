import torch
print("MPS доступен:", torch.backends.mps.is_available())
print("Устройство:", "mps" if torch.backends.mps.is_available() else "cpu")
