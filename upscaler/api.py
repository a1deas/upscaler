# upscaler/upscaler/api.py
from pathlib import Path
from typing import Literal

import shutil, subprocess, tempfile

from .ffmpeg_utils import upscale_image_bicubic, upscale_video_bicubic
from .realesrgan_vulkan import run_realesrgan

Backend = Literal["bicubic", "realesrgan"]

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
        upscale_image_bicubic(input_path, output_path, scale)
    else: 
        run_realesrgan(
            input_path, 
            output_path, 
            scale, 
            model_name = model, 
            auto_download = auto_download
        )

    return output_path

def upscale_video(
        input_path: Path | str,
        output_path: Path | str, 
        scale: int = 2,
        backend: Backend = "realesrgan",
        model: str = "realesrgan-x4plus",
        fps: int | None = None,
        auto_download: bool = False,
) -> Path: 
    input_path = Path(input_path)
    output_path = Path(output_path)

    if backend == "bicubic":
        upscale_video_bicubic(input_path, output_path, scale)
        return output_path
    
    # RealESRGAN
    tmp_dir = Path(tempfile.mkdtemp(prefix = "upscaler_"))
    frames_in = tmp_dir / "in"
    frames_out = tmp_dir / "out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    try:
        pattern_in = frames_in / "frame_%06d.png"
        cmd_extract = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
        ]
        if fps is not None:
            cmd_extract += ["-vf", f"fps={fps}"]
        cmd_extract.append(str(pattern_in))

        proc = subprocess.run(cmd_extract, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg extract failed:\n{proc.stderr}")

        for frame in sorted(frames_in.glob("frame_*.png")):
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
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps or 30),
            "-i",
            str(pattern_out),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        proc2 = subprocess.run(cmd_assemble, capture_output=True, text=True)
        if proc2.returncode != 0:
            raise RuntimeError(f"ffmpeg assemble failed:\n{proc2.stderr}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_path
