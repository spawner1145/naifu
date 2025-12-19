#!/usr/bin/env python3
"""
Check VAE presence in checkpoint and run a simple decode probe using FluxVAE.

Usage:
  PYTHONPATH=. python sdxl_ae/scripts/check_vae.py --ckpt /path/to/ckpt.safetensors --flux_vae /path/to/ae.safetensors --device cuda

This script:
 - Inspects a checkpoint (safetensors or torch) for keys prefixed with `first_stage_model.`
 - Loads FluxVAE (if provided) and decodes a random probe, printing std/mean and dtype.
"""
import argparse
import os
import sys
import torch

try:
    from safetensors.torch import safe_open
except Exception:
    safe_open = None

def inspect_checkpoint(path: str):
    """Return dict-like info: {'type': 'safetensors'|'torch', 'keys': list, 'metadata': dict} """
    info = {"type": None, "keys": [], "metadata": {}}
    if path.endswith((".safetensors", ".sft")):
        if safe_open is None:
            print("safetensors not available (install safetensors to inspect).")
            return info
        info["type"] = "safetensors"
        with safe_open(path, framework="pt") as f:
            info["keys"] = list(f.keys())
            try:
                info["metadata"] = f.metadata() or {}
            except Exception:
                info["metadata"] = {}
        return info

    # fallback to torch
    info["type"] = "torch"
    try:
        sd = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint via torch.load: {e}")
        return info
    # unwrap common container formats
    if isinstance(sd, dict) and "state_dict" in sd:
        state = sd["state_dict"]
    else:
        state = sd if isinstance(sd, dict) else {}
    info["keys"] = list(state.keys())
    # try to extract metadata-like fields
    for k in ("global_step", "current_epoch"):
        if k in sd:
            info["metadata"][k] = sd[k]
    return info

def test_flux_vae(ae_path: str, latent_channels: int, scale_factor: float, device: str):
    """Instantiate FluxVAE from sdxl_ae and run decode probe.
    Returns tuple (ok: bool, message: str).
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from sdxl_ae.modules.flux_vae import FluxVAE
    except Exception as e:
        return False, f"Failed to import FluxVAE: {e}"

    # minimal default params (will be accepted by FluxVAE)
    ae_params = {
        "resolution": 256,
        "in_channels": 3,
        "ch": 128,
        "out_ch": 3,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "z_channels": latent_channels,
        "scale_factor": scale_factor,
        "shift_factor": 0.0,
    }

    try:
        vae = FluxVAE(ae_params, ae_path, target_channels=latent_channels, target_scale_factor=scale_factor)
        vae.to(device)
        vae.eval()
        with torch.no_grad():
            probe = torch.randn(1, latent_channels, 8, 8, device=device)
            out = vae.decode_from_unet(probe)
            out = out.float()
            return True, f"decode dtype={out.dtype}, mean={out.mean().item():.6g}, std={out.std().item():.6g}"
    except Exception as e:
        return False, f"FluxVAE decode failed: {e}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint (safetensors or torch)")
    p.add_argument("--flux_vae", type=str, default=None, help="Path to Flux VAE weights (safetensors/ckpt)")
    p.add_argument("--latent_channels", type=int, default=16)
    p.add_argument("--scale_factor", type=float, default=0.3611)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    if args.ckpt is None and args.flux_vae is None:
        print("Provide --ckpt and/or --flux_vae to run checks.")
        return

    if args.ckpt:
        print(f"Inspecting checkpoint: {args.ckpt}")
        info = inspect_checkpoint(args.ckpt)
        print(f"  type: {info.get('type')}")
        n_keys = len(info.get("keys", []))
        print(f"  total keys: {n_keys}")
        # count first_stage_model.* keys
        first_keys = [k for k in info.get("keys", []) if k.startswith("first_stage_model.")]
        print(f"  first_stage_model.* keys: {len(first_keys)}")
        if first_keys:
            sample = first_keys[:5]
            print("  sample first_stage keys:\n    - " + "\n    - ".join(sample))
        if info.get("metadata"):
            print("  metadata:")
            for k, v in info["metadata"].items():
                print(f"    {k}: {v}")

    if args.flux_vae:
        print(f"Testing FluxVAE decode from: {args.flux_vae} on device {args.device}")
        ok, msg = test_flux_vae(args.flux_vae, args.latent_channels, args.scale_factor, args.device)
        print("  ", msg)

    # Quick heuristic advice
    if args.ckpt and not args.flux_vae:
        print("Note: if you use advanced.use_flux_vae in sdxl_ae, the checkpoint's first_stage_model may be ignored in favor of --flux_vae.\nCheck the sdxl_model logs for lines 'Filtered first_stage_model.* from checkpoint'.")

if __name__ == "__main__":
    main()
