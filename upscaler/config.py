# upscaler/upscaler/config.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "upscaler" / "models"
BINARIES_DIR = PROJECT_ROOT / "upscaler" / "binaries"

if sys.platform.startswith("win"):
    REALESRGAN_BINARY_NAME = "realesrgan-ncnn-vulkan.exe"
elif sys.platform == "darwin":
    REALESRGAN_BINARY_NAME = "realesrgan-ncnn-vulkan"  # macOS without .exe
else:
    REALESRGAN_BINARY_NAME = "realesrgan-ncnn-vulkan"  # Linux

DEFAULT_REALESRGAN_BIN = BINARIES_DIR / REALESRGAN_BINARY_NAME

REALESRGAN_MODELS = {
    "general-x4": "realesrgan-x4plus",                  # x4
    "general-x2": "realesrgan-x2plus",                  # x2
    "anime-x4": "realesrgan-anime-x4",                  # anime x4
}
