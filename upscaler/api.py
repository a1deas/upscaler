# upscaler/upscaler/api.py
from pathlib import Path
from typing import Literal
from rich.console import Console

import shutil, subprocess, tempfile

from .ffmpeg_utils import upscale_image_bicubic, upscale_video_bicubic
from .realesrgan_vulkan import run_realesrgan

console = Console()
Backend = Literal["bicubic", "realesrgan", "torch"]

def upscale_image(
        input_path: Path | str,
        output_path: Path | str, 
        scale: int = 2,
        backend: Backend = "realesrgan",
        model: str = "realesrgan-x4plus",
        auto_download: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if backend == "bicubic":
        console.print("[yellow]Bicubic upscaling is placeholder for now.[/yellow]")
        return output_path
    
    elif backend == "realesrgan": 
        run_realesrgan(
            input_path, 
            output_path, 
            scale, 
            model_name = model, 
            auto_download = auto_download
        )
    elif backend == "torch":
        # Lazy import: keep vulkan backend usable without torch installed.
        try:
            from .realesrgan_torch import run_realesrgan_torch
        except Exception as e:
            raise RuntimeError(
                f"PyTorch backend requested but torch/torchvision is not available ({e}). "
                f"Use --backend realesrgan for the Vulkan/NCNN backend."
            ) from e
        run_realesrgan_torch(
            input_paths = [input_path],
            output_paths = [output_path],
            scale = scale,
            model_name = model,
            device = "cuda",
            fp16 = True,
            batch_size = 1 
        )
    return output_path

def upscale_video(
        input_path: Path | str,
        output_path: Path | str, 
        scale: int = 2,
        backend: Backend = "torch", 
        model: str = "realesrgan-x4plus",
        fps: int | None = None,
        auto_download: bool = False,
        torch_batch_size: int = 4, 
        torch_fp16: bool = True,
) -> Path: 
    input_path = Path(input_path)
    output_path = Path(output_path)

    if backend == "bicubic":
        console.print("[yellow]Bicubic video upscaling is placeholder for now.[/yellow]")
        return output_path
    
    tmp_dir = Path(tempfile.mkdtemp(prefix = "upscaler_"))
    frames_in = tmp_dir / "in"
    frames_out = tmp_dir / "out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    try:
        pattern_in = frames_in / "frame_%06d.png"
        cmd_extract = [
             "ffmpeg", "-y", "-i", str(input_path),
        ]
        if fps is not None:
             cmd_extract += ["-vf", f"fps={fps}"]
        cmd_extract.append(str(pattern_in))
        
        console.log("[cyan] extracting frames... [/cyan]")
        proc = subprocess.run(cmd_extract, capture_output=True, text=True)
        if proc.returncode != 0:
             raise RuntimeError(f"ffmpeg extract failed:\n{proc.stderr}")

        all_frames_in = sorted(frames_in.glob("frame_*.png"))
        all_frames_out = [frames_out / f.name for f in all_frames_in]

        if backend == "torch":
            console.log(f"[bold green] Launching TORCH-backend (CUDA/FP16) [/bold green]")
            # Lazy import: keep vulkan backend usable without torch installed.
            try:
                from .realesrgan_torch import run_realesrgan_torch
                run_realesrgan_torch(
                    input_paths=all_frames_in,
                    output_paths=all_frames_out,
                    scale=scale,
                    model_name=model,
                    device="cuda", 
                    fp16=torch_fp16,
                    batch_size=torch_batch_size,
                )
            except Exception as e:
                console.print(
                    f"[yellow][upscaler] Torch backend failed ({e}). "
                    f"Falling back to Vulkan/NCNN backend.[/yellow]"
                )
                backend = "realesrgan"
        else:
            console.log(f"[bold yellow] Launching VULKAN-backend (NCNN/Vulkan) [/bold yellow]")
            # Fast path: Real-ESRGAN NCNN can process a whole folder of frames in one process.
            # This is dramatically faster than spawning one process per frame.
            try:
                run_realesrgan(
                    input_path=frames_in,
                    output_path=frames_out,
                    scale=scale,
                    model_name=model,
                    auto_download=auto_download,
                )
            except Exception as e:
                console.print(
                    f"[yellow][upscaler] Folder-mode Vulkan failed ({e}). "
                    f"Falling back to per-frame mode.[/yellow]"
                )
                for frame in all_frames_in:
                    out_frame = frames_out / frame.name
                    run_realesrgan(
                        input_path=frame,
                        output_path=out_frame,
                        scale=scale,
                        model_name=model,
                        auto_download=auto_download,
                    )

        pattern_out = frames_out / "frame_%06d.png"
        cmd_assemble = [
            "ffmpeg", "-y", 
            "-framerate", str(fps or 30), 
            "-i", str(pattern_out), 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", 
            str(output_path),
        ]
        console.log("[cyan] Assemling video... [/cyan]")
        proc2 = subprocess.run(cmd_assemble, capture_output=True, text=True)
        if proc2.returncode != 0:
             raise RuntimeError(f"ffmpeg assemble failed:\n{proc2.stderr}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_path