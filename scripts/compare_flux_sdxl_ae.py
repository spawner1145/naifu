"""
Input shape: (1, 3, 1024, 1024)
Flux AE -> flux_recon.png, MAE=0.011757, MSE=0.000492
SDXL FluxVAE -> sdxl_recon.png, MAE=0.011757, MSE=0.000492
Flux vs SDXL recon diff: MAE=0.000000, MSE=0.000000
Saved outputs to: d:\code\naifu1\sdxl\outputs_ae_compare
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SDXL_ROOT = REPO_ROOT / "sdxl"
FLUX_ROOT = REPO_ROOT / "flux"

# Put repo root on sys.path so `flux` and `modules` can be imported as packages.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDXL_ROOT))

from flux.common.autoencoder import AutoEncoder  # type: ignore  # noqa: E402
from flux.common.model_utils import configs, load_file  # type: ignore  # noqa: E402
from modules.flux_vae import FluxVAE  # type: ignore  # noqa: E402


def _load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    h, w = x.shape[-2:]
    if h % 8 != 0 or w % 8 != 0:
        raise ValueError(f"Image size must be divisible by 8; got {h}x{w}. 请直接提供原图或用 --use-sample 生成 1024x1024 测试图。")

    return x * 2 - 1  # map to [-1, 1]


def _save_tensor_as_image(x: torch.Tensor, path: Path) -> None:
    x = x.clamp(-1, 1)
    x = ((x + 1) * 0.5).cpu().squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray((x * 255).astype(np.uint8)).save(path)


def _make_sample_image(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tmp_input_1024.png"
    h = w = 1024
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = np.stack([xx, yy, 0.5 * np.ones_like(xx)], axis=-1)
    checker = (((np.indices((h, w)).sum(axis=0) // 32) % 2) * 0.3).astype(np.float32)
    base[..., 0] = np.clip(base[..., 0] + checker, 0, 1)

    c0, c1 = h // 4, 3 * h // 4
    base[c0:c1, c0:c1, 1] = 1.0
    base[c0:c1, c0:c1, 2] = np.clip(base[c0:c1, c0:c1, 2] + 0.3, 0, 1)

    img = Image.fromarray((base * 255).astype(np.uint8))
    img.save(path)
    return path


def _build_flux_ae(ae_path: Path, device: torch.device) -> tuple[AutoEncoder, dict]:
    params = configs["flux-dev"].ae_params
    ae = AutoEncoder(params)
    sd = load_file(str(ae_path))
    missing, unexpected = ae.load_state_dict(sd, strict=False)
    if missing:
        print(f"[flux] missing keys: {missing}")
    if unexpected:
        print(f"[flux] unexpected keys: {unexpected}")
    ae.reg.sample = False  # use mean instead of sampling noise
    ae.to(device).eval()
    return ae, asdict(params)


def _build_sdxl_ae(ae_params: dict, ae_path: Path, device: torch.device) -> FluxVAE:
    vae = FluxVAE(ae_params, ae_path=str(ae_path))
    vae.ae.reg.sample = False
    vae.to(device).eval()
    return vae


def _reconstruct_flux(ae: AutoEncoder, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z = ae.encode(x)
        return ae.decode(z)


def _reconstruct_sdxl(vae: FluxVAE, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        posterior = vae.encode(x)
        z = posterior.sample()
        return vae.decode_from_unet(z)


def _metrics(rec: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    diff = rec - target
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    return mae, mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Flux vs SDXL AE reconstructions")
    parser.add_argument("--image", type=Path, required=False, help="Path to the input image")
    parser.add_argument(
        "--ae-path",
        type=Path,
        default=REPO_ROOT / "ae.safetensors",
        help="Path to ae weights (safetensors or .pt)",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="使用内置 1024x1024 合成图进行对比（分辨率与 VAE 配置保持一致，不做缩放）",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SDXL_ROOT / "outputs_ae_compare",
        help="Where to save reconstructed images",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_sample:
        img_path = _make_sample_image(args.output_dir)
        print(f"已生成内置测试图: {img_path}")
    elif args.image is not None:
        img_path = args.image
    else:
        raise SystemExit("请提供 --image 或使用 --use-sample 生成 1024x1024 测试图。")

    x = _load_image(img_path).to(device)

    flux_ae, ae_params = _build_flux_ae(args.ae_path, device)
    sdxl_vae = _build_sdxl_ae(ae_params, args.ae_path, device)

    flux_rec = _reconstruct_flux(flux_ae, x)
    sdxl_rec = _reconstruct_sdxl(sdxl_vae, x)

    flux_mae, flux_mse = _metrics(flux_rec, x)
    sdxl_mae, sdxl_mse = _metrics(sdxl_rec, x)
    cross_mae, cross_mse = _metrics(flux_rec, sdxl_rec)

    flux_out = args.output_dir / "flux_recon.png"
    sdxl_out = args.output_dir / "sdxl_recon.png"
    _save_tensor_as_image(flux_rec, flux_out)
    _save_tensor_as_image(sdxl_rec, sdxl_out)

    print(f"Using device: {device}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Flux AE -> {flux_out.name}, MAE={flux_mae:.6f}, MSE={flux_mse:.6f}")
    print(f"SDXL FluxVAE -> {sdxl_out.name}, MAE={sdxl_mae:.6f}, MSE={sdxl_mse:.6f}")
    print(f"Flux vs SDXL recon diff: MAE={cross_mae:.6f}, MSE={cross_mse:.6f}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
