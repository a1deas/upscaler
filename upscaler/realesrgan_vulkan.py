"""
Thin wrapper around the Real-ESRGAN NCNN/Vulkan binary.

Important: "Vulkan" does NOT guarantee "GPU". In WSL/Docker/misconfigured hosts,
the binary can end up using software Vulkan implementations (e.g. llvmpipe /
SwiftShader), which are CPU-backed and extremely slow. We surface that clearly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

from .downloads import ensure_realesrgan_binary, ensure_model

console = Console()


def _env_int(name: str) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except Exception:
        return None


def _is_software_vulkan(output: str) -> bool:
    """
    Best-effort detection of CPU-backed Vulkan drivers.
    Seen commonly in WSL/Docker/VMs: llvmpipe / SwiftShader.
    """
    s = (output or "").lower()
    return ("llvmpipe" in s) or ("swiftshader" in s)


def run_realesrgan(
        input_path: Path,
        output_path: Path,
        scale: int, 
        model_name: str, 
        auto_download: bool = False,
        gpu_id: Optional[int] = None,
        verbose: bool = False,
        force_gpu: bool = False,
) -> None: 
    bin_path = ensure_realesrgan_binary(auto_download = auto_download)
    model_dir = ensure_model(model_name, auto_download = auto_download)

    if not bin_path.exists() or model_dir is None: 
        raise RuntimeError("RealESRGAN binary or model missing")

    # Support both single-file and folder modes.
    # - If output_path looks like a file (has suffix), create parent folder.
    # - Otherwise treat it as an output directory and create it.
    if output_path.suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    if gpu_id is None:
        # Allow configuring GPU selection without changing call sites.
        gpu_id = _env_int("CUTSMITH_UPSCALER_GPU_ID")
        if gpu_id is None:
            gpu_id = _env_int("REALESRGAN_GPU_ID")

    # Enable verbose output if requested or env var is set.
    verbose = verbose or os.environ.get("CUTSMITH_UPSCALER_VERBOSE") in ("1", "true", "yes", "on")
    force_gpu = force_gpu or os.environ.get("CUTSMITH_FORCE_GPU") in ("1", "true", "yes", "on")

    cmd: list[str] = [
        str(bin_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", str(scale),
        "-n", model_name,
        "-m", str(model_dir), 
]

    if gpu_id is not None:
        # Real-ESRGAN: -g can be 0,1,2... or -1 (CPU). Default is "auto".
        cmd += ["-g", str(gpu_id)]

    if verbose:
        cmd += ["-v"]


    console.log(f"[blue] RealESRGAN: {' '.join(cmd)}[/blue]")
    proc = subprocess.run(cmd, capture_output = True, text = True)

    combined = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    if verbose and combined.strip():
        console.print(combined.rstrip())

    if _is_software_vulkan(combined):
        msg = (
            "RealESRGAN Vulkan is using a software Vulkan device (llvmpipe/SwiftShader) "
            "which runs on CPU. This usually means you're running inside WSL/Docker/VM "
            "without real GPU/Vulkan passthrough.\n\n"
            "Fix options:\n"
            "- Run the native Windows binary (realesrgan-ncnn-vulkan.exe) from Windows Python.\n"
            "- If using WSL2, install GPU drivers + WSLg so Vulkan enumerates your real GPU.\n"
            "- If using Docker, mount /dev/dri (Intel/AMD) or use NVIDIA container toolkit.\n"
        )
        if force_gpu:
            raise RuntimeError(msg)
        console.print(f"[yellow]{msg}[/yellow]")

    if proc.returncode != 0:
        raise RuntimeError(f"RealESRGAN failed:\n{combined or '(no output)'}")