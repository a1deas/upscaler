# upscaler/upscaler/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "upscaler" / "models"
BINARIES_DIR = PROJECT_ROOT / "upscaler" / "binaries"

DEFAULT_REALESRGAN_BIN = BINARIES_DIR / "realesrgan-ncnn-vulkan"

REALESRGAN_MODELS = {
    "general-x4": "realesrgan-x4plus",
    "general-x2": "realesrgan-x2plus",
    "anime-x4": "realesrgan-anime-x4",
}