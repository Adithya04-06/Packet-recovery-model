import torch

def verify_gpu():
    print("=== PyTorch GPU Verification ===")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA Device Count: {device_count}")
        print(f"CUDA Version (compiled with PyTorch): {torch.version.cuda}")
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = round(props.total_memory / (1024**3), 2)
            print(f"Device {i}: {name} ({memory_gb} GB VRAM)")
            
        print("\nSUCCESS: PyTorch is ready to use the GPU!")
    else:
        print("\nWARNING: PyTorch cannot detect a CUDA-capable GPU.")
        print("You are likely running the CPU-only version of PyTorch.")

if __name__ == "__main__":
    verify_gpu()
