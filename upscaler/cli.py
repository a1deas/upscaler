# upscaler/upscaler/cli.py
from pathlib import Path
from typing import Optional

import typer    
from rich.console import Console

from .api import upscale_image, upscale_video

console = Console()

app = typer.Typer(
    help = "Simple CLI for image/video upscaling"
)

@app.command()
def run(
    input_path: Path = typer.Argument(
        ...,
        help = "Path to input file(image or video)",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help = "Path to output file (auto-name if not set)",
    ),
    mode: str = typer.Option(
        "video",
        "--mode",
        "-m",
        help = "Mode: image / video",
    ),
    backend: str = typer.Option(
        "realesrgan",
        "--backend",
        "-b",
        help = "Backend: bicubic / realesrgan",
    ),
    scale: int = typer.Option(
        2,
        "--scale",
        "-s",
        help = "Scale factor (2/3/4)",
    ),
    model: str = typer.Option(
        "realesrgan-x4plus",
        "--model",
        help = "RealESRGAN model name",
    ),
    auto_download: bool = typer.Option(
        False,
        "--auto-download",
        help = "Try to download binary/models if missing",
    ),
):
    if not input_path.exists():
        raise typer.BadParameter(f"Input file does not exist: {input_path}")
    
    if output_path is None: 
        suffix = ".mp4" if mode == "video" else ".png"
        output_path = input_path.with_name(input_path.stem + f"_x{scale}{suffix}")

    console.log(f"[bold cyan] Upscaler[/] running on [yellow]{input_path}")
    console.log(f"mode = {mode}, backend = {backend}, scale = {scale}")

    if mode == "image":
        upscale_image(
            input_path = input_path,
            output_path = output_path, 
            scale = scale,
            backend = backend,  # type: ignore[arg-type]
            model = model,
            auto_download = auto_download,
        )
    else: 
        upscale_video(
            input_path=input_path,
            output_path=output_path,
            scale=scale,
            backend=backend,  # type: ignore[arg-type]
            model=model,
            auto_download=auto_download,
        )

    console.log(f"[green]Done[/]: {output_path}")
