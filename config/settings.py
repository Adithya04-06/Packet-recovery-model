import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cryptography Settings
# 32-byte key for AES-256
AES_KEY = b'sixteen byte key for AES-GCM 123'  # 32 bytes
if len(AES_KEY) != 32:
    AES_KEY = AES_KEY.ljust(32, b'0')[:32]

# Packetization Settings
PACKET_PAYLOAD_SIZE = 16  # bytes (small to simulate low bandwidth / high packetization)
PACKET_HEADER_SIZE = 12   # sequence num, length, etc.
CRC_SIZE = 4

# ML Model Settings
MODEL_NAME = "t5-small"  # For English prototype. Later "google/mt5-small" for multilingual.
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
MISSING_TOKEN = "<extra_id_0>"  # T5's default sentinel token for masked language modeling style
# We will conceptually use "<MISSING>" in our UI, but translate it to T5's format internally.

# Training Settings
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 8
VALIDATION_SPLIT = 0.1

# Directories and Versioning
MODEL_VERSION = "v2"  # Change to "v1" or "v2"

DATA_DIR = os.path.join(BASE_DIR, "data", MODEL_VERSION)
MODELS_DIR = os.path.join(BASE_DIR, "models", MODEL_VERSION)
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")
GPU_MODEL_DIR = os.path.join(MODELS_DIR, "gpu_model")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(GPU_MODEL_DIR, exist_ok=True)
