# upscaler/upscaler/realesrgan_torch.py
# WARNING: torch backend is SLOW, for expriments/short clips only
from pathlib import Path
from typing import List
import sys, types

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import normalize
from torchvision.utils import save_image

import torchvision.transforms.functional as F_tv

shim_module = types.ModuleType("torchvision.transforms.functional_tensor")
shim_module.rgb_to_grayscale = F_tv.rgb_to_grayscale # type: ignore[arg-type]
sys.modules.setdefault("torchvision.transforms.functional_tensor", shim_module)

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from rich.console import Console

console = Console()

MODEL_URLS = {
    "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}


def load_realesrgan_model(model_name: str, scale: int, device: str) -> RRDBNet:
    model_url = MODEL_URLS.get(model_name)
    if not model_url:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = load_file_from_url(
        url=model_url,
        model_dir=str(Path.home() / ".cache" / "realesrgan" / "weights"),
        progress=True,
    )

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    ).to(device)

    loadnet = torch.load(model_path, map_location=device)
    if "params_ema" in loadnet:
        loadnet = loadnet["params_ema"]
    elif "params" in loadnet:
        loadnet = loadnet["params"]

    model.load_state_dict(loadnet, strict=True)
    model.eval()

    if len(loadnet) > 0:
        first_key = next(iter(loadnet.keys()))
        if first_key.startswith("module."):
            loadnet = {k.replace("module.", "", 1): v for k, v in loadnet.items()}

    model.load_state_dict(loadnet, strict=True)
    model.eval()

    return model


def run_realesrgan_torch(
    input_paths: List[Path],
    output_paths: List[Path],
    scale: int,
    model_name: str,
    device: str = "cuda",
    fp16: bool = True,
    batch_size: int = 4,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow][upscaler] CUDA not found. Switching to CPU (this will be slow).[/yellow]")
        device = "cpu"

    console.log(f"[cyan][upscaler] Using backend=torch, model={model_name}, scale={scale}, device={device}[/cyan]")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_realesrgan_model(model_name, scale, device)
    if fp16 and device == "cuda":
        model = model.half()

    total_frames = len(input_paths)
    if total_frames == 0:
        console.log("[yellow][upscaler] No frames to process.[/yellow]")
        return

    for i in range(0, total_frames, batch_size):
        batch_in = input_paths[i : i + batch_size]
        batch_out = output_paths[i : i + batch_size]

        tensor_list = []
        for p in batch_in:
            img = read_image(str(p)).to(device)
            img = img.float() / 255.0          
            img = img.unsqueeze(0)               

            normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

            if fp16 and device == "cuda":
                img = img.half()

            tensor_list.append(img)

        if not tensor_list:
            continue

        input_tensor = torch.cat(tensor_list, dim=0) 

        console.log(
            f"[blue][upscaler] Processing batch {i//batch_size + 1} "
            f"({len(batch_in)} frames) on {device}[/blue]"
        )

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor.mul_(0.5).add_(0.5).clamp_(0.0, 1.0)

        for j, out_t in enumerate(output_tensor):
            out_path = batch_out[j]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(out_t, str(out_path))  
