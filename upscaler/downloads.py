# upscaler/upscaler/downloads.py
from pathlib import Path
from typing import Optional

import tempfile, zipfile, shutil, urllib.request

from rich.console import Console

from .config import MODELS_DIR, BINARIES_DIR, DEFAULT_REALESRGAN_BIN

console = Console()

BINARY_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
)

def ensure_dirs() -> None: 
    MODELS_DIR.mkdir(parents = True, exist_ok = True)
    BINARIES_DIR.mkdir(parents = True, exist_ok = True)

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

        bin_src: Optional[Path] = None
        model_files: list[Path] = []

        # Searching for *.bin/*.param
        for p in tmpdir.rglob("*"):
            if p.is_file() and p.name == "realesrgan-ncnn-vulkan":
                bin_src = p
            elif p.is_file() and p.suffix in (".bin", ".param"):
                model_files.append(p)

        if bin_src is None or not model_files:
            raise RuntimeError("Archive missing binary or model files")

        BINARIES_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Binaries
        shutil.copy2(bin_src, DEFAULT_REALESRGAN_BIN)
        DEFAULT_REALESRGAN_BIN.chmod(0o755)

        # Models
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
            f"[yellow] Realesrgan-ncnn-vulkan not found: {bin_path}. "
            "Copy it or --auto-download.[/yellow]"
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
