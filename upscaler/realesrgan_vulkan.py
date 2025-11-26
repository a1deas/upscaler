# upscaler/upscaler/realesrgan_vulkan.py
import subprocess
from pathlib import Path

from rich.console import Console

from .downloads import ensure_realesrgan_binary, ensure_model

console = Console()

def run_realesrgan(
        input_path: Path,
        output_path: Path,
        scale: int, 
        model_name: str, 
        auto_download: bool = False,
) -> None: 
    bin_path = ensure_realesrgan_binary(auto_download = auto_download)
    model_dir = ensure_model(model_name, auto_download = auto_download)

    if not bin_path.exists() or model_dir is None: 
        raise RuntimeError("RealESRGAN binary or model missing")
    
    cmd = [
        str(bin_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", str(scale),
        "-n", model_name,
        "-m", str(model_dir), 
]


    console.log(f"[blue] RealESRGAN: {' '.join(cmd)}[/blue]")
    proc = subprocess.run(cmd, capture_output = True, text = True)
    if proc.returncode != 0: 
        console.print(f"[red]{proc.stderr}[/red]")
        raise RuntimeError("RealESRGAN failed")