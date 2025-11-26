# upscaler/upscaler/ffmpeg_utils.py
import subprocess
from pathlib import Path

from rich.console import Console

console = Console

def upscale_image_bicubic(input_path: Path, output_path: Path, scale: int) -> None: 
    w = f"iw*{scale}"
    h = f"ih*{scale}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"scale = {w}:{h}:flags=bicubic",
        str(output_path),
    ]
    console.log(f"[blue] FFmpeg bicubic image[/blue]")
    proc = subprocess.run(cmd, capture_output = True, text = True)
    if proc.returncode != 0: 
        console.print(f"[red] {proc.stderr}[/red]")
        raise RuntimeError("ffmpeg bicubic image failed")