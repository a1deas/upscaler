# upscaler/upscaler/downloads.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import sys
import shutil
import tempfile
import urllib.request
import zipfile

from rich.console import Console

from .config import MODELS_DIR, BINARIES_DIR, DEFAULT_REALESRGAN_BIN

console = Console()

if sys.platform.startswith("win"):
    PLATFORM = "windows"
elif sys.platform == "darwin":
    PLATFORM = "macos"
else:
    PLATFORM = "linux"

BINARY_URLS = {
    "linux": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    ),
    "windows": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
    ),
    "macos": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
    ),
}

BINARY_URL = os.environ.get("CUTSMITH_REALESRGAN_NCNN_URL") or BINARY_URLS[PLATFORM]


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BINARIES_DIR.mkdir(parents=True, exist_ok=True)


def _download_and_unpack_ncnn(url: str) -> None:
    ensure_dirs()

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        zip_path = tmpdir / "realesrgan.zip"

        console.log(f"[cyan]Downloading RealESRGAN NCNN from {url}[/cyan]")
        urllib.request.urlretrieve(url, zip_path)

        console.log("[cyan]Unpacking archive...[/cyan]")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        bin_src: Path | None = None
        model_files: list[Path] = []

        for p in tmpdir.rglob("*"):
            if p.is_file() and p.name.startswith("realesrgan-ncnn-vulkan"):
                bin_src = p
            elif p.is_file() and p.suffix in (".bin", ".param"):
                model_files.append(p)

        if bin_src is None or not model_files:
            raise RuntimeError("Archive missing binary or model files")

        BINARIES_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        shutil.copy2(bin_src, DEFAULT_REALESRGAN_BIN)
        # On Windows chmod is mostly a no-op and can fail depending on ACLs.
        try:
            DEFAULT_REALESRGAN_BIN.chmod(0o755)
        except Exception:
            pass

        for mf in model_files:
            dst = MODELS_DIR / mf.name
            shutil.copy2(mf, dst)

        console.print("[green]RealESRGAN NCNN downloaded and unpacked[/green]")


def ensure_realesrgan_binary(auto_download: bool = False) -> Path:
    ensure_dirs()
    bin_path = DEFAULT_REALESRGAN_BIN

    if bin_path.exists():
        return bin_path

    if not auto_download:
        console.print(
            f"[yellow]Realesrgan-ncnn-vulkan not found: {bin_path}. "
            "Copy it manually or use --auto-download.[/yellow]"
        )
        return bin_path

    _download_and_unpack_ncnn(BINARY_URL)

    if not bin_path.exists():
        raise RuntimeError("Download finished, but binary still not found.")

    return bin_path


def ensure_model(model_name: str, auto_download: bool = False) -> Optional[Path]:
    ensure_dirs()

    param = MODELS_DIR / f"{model_name}.param"
    bin_ = MODELS_DIR / f"{model_name}.bin"

    if param.exists() and bin_.exists():
        return MODELS_DIR

    if auto_download:
        console.print("[cyan]Model not found, trying to download NCNN package...[/cyan]")
        ensure_realesrgan_binary(auto_download=True)
        if param.exists() and bin_.exists():
            return MODELS_DIR

    console.print(
        f"[yellow]Model {model_name} not found in {MODELS_DIR}. "
        "Make sure archive was unpacked correctly.[/yellow]"
    )
    return None
