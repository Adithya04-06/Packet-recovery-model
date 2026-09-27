import torch
import time

print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn((2048, 2048), device="cuda")
y = torch.randn((2048, 2048), device="cuda")

torch.cuda.synchronize()

print("Starting 30-second GPU test...")

start = time.time()
iterations = 0

while time.time() - start < 30:
    z = x @ y
    torch.cuda.synchronize()
    iterations += 1

print("Test completed successfully.")
print("Iterations:", iterations)