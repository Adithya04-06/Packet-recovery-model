# GPU Setup Instructions

## Current Environment Status
- **OS**: Windows
- **Python**: 3.11.9
- **GPU**: NVIDIA GeForce RTX 3050 (6 GB VRAM)
- **Driver Version**: 581.86 (Supports up to CUDA 13.0)
- **Current PyTorch**: 2.14.0+cpu (CPU-only version)
- **CUDA Available in PyTorch**: False

## The Problem
The current virtual environment has the CPU-only version of PyTorch installed. While your system has a capable RTX 3050 GPU and proper NVIDIA drivers, PyTorch cannot interface with it without the CUDA-compiled wheels.

## 1. Install CUDA-Enabled PyTorch
**Wait until the current CPU training process (PID 4668) finishes completely before running this!**

Once it is safe, run the following command in your virtual environment. Since your driver supports up to CUDA 13.0, you can safely use the PyTorch CUDA 12.4 or 12.1 builds (CUDA drivers are strictly backwards compatible):

```powershell
# Activate your venv if not already active
.\venv\Scripts\activate

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 2. Verify GPU Activation
After installation, verify that PyTorch sees your GPU by running the script we created:
```powershell
python -m recovery.check_gpu
```
It should print `CUDA is available: True` and show your RTX 3050.

## 3. Configuration & Batch Size Tuning
Your RTX 3050 has **6 GB of VRAM**. T5-small is relatively lightweight, but you can still run out of memory (OOM) if the batch size is too large. 

1. Open `config/settings.py`.
2. Keep `BATCH_SIZE = 16` initially. 
3. Start training using `python -m recovery.train`.
4. Monitor VRAM usage in another terminal by running `nvidia-smi`.
5. If VRAM usage is well below 6GB (e.g., ~3GB), you can stop the training, increase to `BATCH_SIZE = 24` or `32`, and restart. 
6. If you encounter a `CUDA OutOfMemoryError`, reduce the batch size back down.

## 4. Run Training
Because the codebase (`recovery/train.py` and `recovery/model.py`) is already written to automatically detect and use CUDA, you do not need to change any Python code. Just launch training normally:
```powershell
python -m recovery.train
```
