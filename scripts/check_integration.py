#!/usr/bin/env python3
"""
Integration check: inspect a SDXL checkpoint and Flux VAE, report VAE decode stats
and UNet parameter statistics to help diagnose "noise samples" issues.

Usage:
  python sdxl_ae/scripts/check_integration.py --ckpt cknoobep13.safetensors --flux_vae ae.safetensors --device cpu
"""
import sys, os
# ensure repo root and sdxl_ae package dir are importable so `common` imports resolve
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sdxl_ae_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, sdxl_ae_pkg)
import argparse
import torch
from pathlib import Path

def inspect_ckpt(path: str):
    from sdxl_ae.common.utils import load_torch_file
    info = {}
    try:
        sd = load_torch_file(path, device="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
        return None
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    info['n_keys'] = len(keys)
    info['first_stage_keys'] = [k for k in keys if k.startswith('first_stage_model.')]
    info['model_keys'] = [k for k in keys if k.startswith('model.')]
    info['cond_keys'] = [k for k in keys if k.startswith('conditioner.')]
    info['sample_keys'] = keys[:10]
    return info, sd

def test_flux_vae(ae_path: str, latent_channels: int, scale_factor: float, device: str, ckpt_sd: dict | None = None, target_z: int = 16):
    try:
        from sdxl_ae.modules.flux_vae import FluxVAE
    except Exception as e:
        return False, f"Import FluxVAE error: {e}"
    try:
        ae_params = {
            'resolution': 256,
            'in_channels': 3,
            'ch': 128,
            'out_ch': 3,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'z_channels': target_z,
            'scale_factor': scale_factor,
            'shift_factor': 0.0,
        }
        # instantiate target FluxVAE WITHOUT loading weights so we can obtain target shapes
        vae = FluxVAE(ae_params, ae_path=None, target_channels=target_z, target_scale_factor=scale_factor)
        vae.to(device)
        vae.eval()

        # if a checkpoint state dict provided and it contains first_stage_model weights,
        # try to reuse/expand those weights into the FluxVAE target using sdxl_ae merge logic
        if ckpt_sd is not None:
            try:
                from sdxl_ae.modules.sdxl_model import StableDiffusionModel
            except Exception:
                from sdxl_ae.modules.sdxl_model import StableDiffusionModel

            # prepare target and src dicts with matching key namespaces
            target_sd = {f'first_stage_model.{k}': v for k, v in vae.state_dict().items()}
            src_fs = {k: v for k, v in ckpt_sd.items() if k.startswith('first_stage_model.')}

            if src_fs:
                merged = StableDiffusionModel._merge_state_dict(None, target_sd, src_fs)
                # strip prefix and load into internal ae module
                load_sd = {k.replace('first_stage_model.', ''): v for k, v in merged.items()}
                missing, unexpected = vae.ae.load_state_dict(load_sd, strict=False)
                if missing:
                    print(f"  After merge, missing VAE keys: {missing}")
                if unexpected:
                    print(f"  After merge, unexpected VAE keys: {unexpected}")
        else:
            # if no checkpoint provided, attempt to let FluxVAE load its own file
            if ae_path is not None:
                # re-init with loading
                vae = FluxVAE(ae_params, ae_path, target_channels=target_z, target_scale_factor=scale_factor)
                vae.to(device)
                vae.eval()

        with torch.no_grad():
            probe = torch.randn(1, target_z, 8, 8, device=device)
            out = vae.decode_from_unet(probe)
            out = out.float()
            return True, {'dtype': str(out.dtype), 'mean': float(out.mean().item()), 'std': float(out.std().item())}
    except Exception as e:
        return False, f"VAE decode failed: {e}"

def unet_param_stats(sd: dict):
    # collect params that look like unet (model.*)
    model_keys = [k for k in sd.keys() if k.startswith('model.')]
    if not model_keys:
        return {'found': False, 'msg': 'no model.* keys in checkpoint'}
    stats = {}
    cnt=0
    tot_params=0
    for k in model_keys:
        v = sd[k]
        if not isinstance(v, torch.Tensor):
            continue
        cnt += 1
        tot_params += v.numel()
        # compute simple stats for first few tensors
        if cnt <= 10:
            stats[k] = {'shape': tuple(v.shape), 'mean': float(v.float().mean().item()), 'std': float(v.float().std().item()), 'max_abs': float(v.abs().max().item())}
    return {'found': True, 'n_model_keys': len(model_keys), 'sample_stats': stats, 'total_params': tot_params}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='cknoobep13.safetensors')
    p.add_argument('--flux_vae', type=str, default='ae.safetensors')
    p.add_argument('--latent_channels', type=int, default=4)
    p.add_argument('--scale_factor', type=float, default=0.3611)
    p.add_argument('--target_z', type=int, default=16, help='desired Flux VAE latent channels (e.g. 16)')
    p.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found.")
    else:
        print(f"Inspecting checkpoint: {ckpt_path}")
        info_sd = inspect_ckpt(str(ckpt_path))
        if info_sd is None:
            print("Failed to inspect checkpoint.")
        else:
            info, sd = info_sd
            print(f"  total keys: {info['n_keys']}")
            print(f"  first_stage_model.* keys: {len(info['first_stage_keys'])}")
            if info['first_stage_keys']:
                print('    sample:', info['first_stage_keys'][:5])
            print(f"  model.* keys: {len(info['model_keys'])}")
            if info['model_keys']:
                print('    sample model keys:', info['model_keys'][:5])
            print(f"  conditioner.* keys: {len(info['cond_keys'])}")
            print('  sample keys:', info['sample_keys'])
            # unet param stats
            ups = unet_param_stats(sd)
            print('UNet param stats:', ups)

    # test flux vae
    if args.flux_vae and Path(args.flux_vae).exists():
        print(f"Testing FluxVAE at {args.flux_vae} on device {args.device}")
        ckpt_sd = locals().get('sd', None)
        ok, res = test_flux_vae(args.flux_vae, args.latent_channels, args.scale_factor, args.device, ckpt_sd=ckpt_sd, target_z=args.target_z)
        if ok:
            print(f"  VAE decode ok: dtype={res['dtype']} mean={res['mean']:.6g} std={res['std']:.6g}")
        else:
            print(f"  VAE decode failed: {res}")
    else:
        print(f"Flux VAE file {args.flux_vae} not found; skipping VAE decode test.")

    print('\nIntegration check completed.')

if __name__ == '__main__':
    main()
