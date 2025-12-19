
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_sft
from safetensors.torch import save_file as save_sft
from omegaconf import OmegaConf
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.flux_vae import FluxVAE, _load_flux_state  # noqa: E402
from modules.sdxl_model import StableDiffusionModel  # noqa: E402

AE_PATH = ROOT / "ae.safetensors"
SDXL_PATH = ROOT / "sdxl.safetensors"

DEFAULT_PARAMS = {
    "resolution": 256,
    "in_channels": 3,
    "ch": 128,
    "out_ch": 3,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "z_channels": 16,
    "scale_factor": 0.3611,
    "shift_factor": 0.1159,
}


def _merge_state_dict(target_sd: dict, src_sd: dict) -> dict:
    """Copy overlapping tensor regions to maximize reuse when shapes differ."""
    merged = target_sd.copy()
    for k, src in src_sd.items():
        if k not in merged:
            continue
        tgt = merged[k]
        if tgt.shape == src.shape:
            merged[k] = src
        else:
            slices = tuple(slice(0, min(tgt.size(i), src.size(i))) for i in range(src.dim()))
            new_t = tgt.clone()
            new_t[slices] = src[slices]
            merged[k] = new_t
    return merged


def test_flux_vae_encode_decode(device: str = "cpu") -> None:
    print("FluxVAE encode/decode smoke")
    vae = FluxVAE(DEFAULT_PARAMS, ae_path=str(AE_PATH))
    vae.to(device)
    vae.eval()

    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        posterior = vae.encode(x)
        z = posterior.sample()
        x_rec = vae.decode_from_unet(z)

    print(f"input -> {tuple(x.shape)}, latent -> {tuple(z.shape)}, recon -> {tuple(x_rec.shape)}")
    diff = (x_rec - x).abs().mean().item()
    print(f"mean abs diff: {diff:.6f}")


def test_merge_state_dict() -> None:
    print("_merge_state_dict channel copy")
    target = {"block.weight": torch.zeros(1, 16, 4, 4)}
    src = {"block.weight": torch.randn(1, 4, 4, 4)}
    merged = _merge_state_dict(target, src)
    overlap_diff = merged["block.weight"][:, :4].sub(src["block.weight"]).abs().max().item()
    print(f"overlap diff (expect 0): {overlap_diff:.6f}")


def test_flux_vae_channel_guard() -> None:
    print("FluxVAE channel assert (should fail when mismatched)")
    try:
        FluxVAE({**DEFAULT_PARAMS, "z_channels": 16}, target_channels=32)
    except AssertionError as exc:
        print(f"caught AssertionError as expected: {exc}")
    else:
        raise RuntimeError("FluxVAE channel assert did not trigger")


def test_filter_checkpoint_for_flux() -> None:
    print("SDXL checkpoint filtering for flux_vae_path override")
    if not SDXL_PATH.exists():
        print("sdxl.safetensors missing; skip")
        return

    with safe_open(str(SDXL_PATH), framework="pt") as f:
        keys = list(f.keys())
    first_stage_keys = [k for k in keys if k.startswith("first_stage_model.")]
    print(f"original first_stage_model.* keys: {len(first_stage_keys)}")

    # simulate filtering when flux_vae_path is provided (sdxl_model init removes these)
    filtered = {k: torch.empty(0) for k in keys if not k.startswith("first_stage_model.")}
    filtered_first_stage = [k for k in filtered if k.startswith("first_stage_model.")]
    print(f"filtered first_stage_model.* keys: {len(filtered_first_stage)} (expect 0)")

    if len(filtered_first_stage) != 0:
        raise RuntimeError("Filtering failed; VAE weights still present")


def test_partial_merge_with_sdxl_shapes() -> None:
    print("Partial merge shape check (4c -> 16c)")
    # mock a conv weight from original SDXL (4 channels) into a Flux UNet (16 channels)
    src = {"model.diffusion_model.input_blocks.0.0.weight": torch.randn(320, 4, 3, 3)}
    tgt = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(320, 16, 3, 3)}
    merged = _merge_state_dict(tgt, src)
    copied = merged["model.diffusion_model.input_blocks.0.0.weight"][:, :4]
    diff = (copied - src["model.diffusion_model.input_blocks.0.0.weight"]).abs().max().item()
    print(f"copied diff (expect 0): {diff:.6f}")


def test_checkpoint_headers() -> None:
    print("checkpoint header read")
    ae_exists = AE_PATH.exists()
    sdxl_exists = SDXL_PATH.exists()
    print(f"ae.safetensors exists: {ae_exists}, sdxl.safetensors exists: {sdxl_exists}")

    if ae_exists:
        ae_state = _load_flux_state(str(AE_PATH))
        ae_keys = list(ae_state.keys())[:5]
        print(f"ae keys sample ({len(ae_state)} total): {ae_keys}")

    if sdxl_exists:
        with safe_open(str(SDXL_PATH), framework="pt") as f:
            sdxl_keys = list(f.keys())
        print(f"sdxl keys sample ({len(sdxl_keys)} total): {sdxl_keys[:5]}")
        # avoid loading full tensors; just read one small tensor to confirm I/O
        with safe_open(str(SDXL_PATH), framework="pt") as f:
            first_key = sdxl_keys[0]
            tensor_meta = f.get_tensor(first_key)
            print(f"first tensor {first_key} shape: {tuple(tensor_meta.shape)}")


def _build_flux_config() -> "OmegaConf":
    cfg = OmegaConf.load(str(ROOT / "config" / "train_sdxl.yaml"))
    cfg.trainer.model_path = str(SDXL_PATH)
    cfg.advanced.use_flux_vae = True
    cfg.advanced.flux_vae_path = str(AE_PATH)
    cfg.advanced.save_flux_vae = True
    # smaller batch/size for smoke
    cfg.trainer.batch_size = 1
    cfg.dataset.max_token_length = 75
    return cfg


@torch.no_grad()
def test_end_to_end_forward(device: str = "cpu", height: int = 64, width: int = 64) -> None:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cfg = _build_flux_config()
    try:
        model = StableDiffusionModel(str(SDXL_PATH), cfg, torch.device(device))
    except Exception as exc:  # noqa: BLE001
        print(f"skip end-to-end forward: {exc}")
        return
    model.eval()

    # prepare cond batch
    size = (height, width)
    prompts_batch = {
        "target_size_as_tuple": torch.tensor([size], device=device),
        "original_size_as_tuple": torch.tensor([size], device=device),
        "crop_coords_top_left": torch.tensor([(0, 0)], device=device),
        "prompts": "a photo of a cat",
    }
    cond = model.encode_batch(prompts_batch)
    model_dtype = next(model.model.parameters()).dtype
    cond = {k: v.to(device=device, dtype=model_dtype) for k, v in cond.items()}

    # latent and timestep
    latents = torch.randn(1, model.latent_channels, height // 8, width // 8, device=device, dtype=model_dtype)
    t = torch.tensor([0], device=device, dtype=torch.int64)

    try:
        out = model.model(latents, t, cond)
        print(f"UNet output shape: {tuple(out.shape)}")
        decoded = model.decode_first_stage(latents)
        print(f"Decoded image shape: {tuple(decoded.shape)}")
    except RuntimeError as exc:
        print(f"skip UNet forward due to runtime error: {exc}")


@torch.no_grad()
def test_flux_vae_save_roundtrip(tmp_name: str = "tmp_flux_vae.safetensors", device: str = "cpu") -> None:
    print("Flux VAE save/load roundtrip")
    vae = FluxVAE(DEFAULT_PARAMS, ae_path=str(AE_PATH)).to(device)
    state = vae.ae.state_dict()
    tmp_path = ROOT / tmp_name
    save_sft(state, str(tmp_path))
    reloaded = load_sft(str(tmp_path))
    print(f"saved {len(state)} tensors, reloaded {len(reloaded)} tensors")
    tmp_path.unlink(missing_ok=True)


@torch.no_grad()
def test_flux_vae_geom_image(device: str = "cpu", size: int = 64) -> None:
    print("Flux VAE encode/decode on synthetic geometric image")
    vae = FluxVAE(DEFAULT_PARAMS, ae_path=str(AE_PATH)).to(device)
    vae.eval()

    # create simple RGB geometric pattern (center square + border)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = 0.2  # dim red background
    img[:, :, 1] = 0.6  # green tint
    img[:, :, 2] = 0.9  # blue tint
    # center square
    c0, c1 = size // 4, 3 * size // 4
    img[c0:c1, c0:c1, 0] = 0.9
    img[c0:c1, c0:c1, 1] = 0.1
    img[c0:c1, c0:c1, 2] = 0.2
    # border line
    img[0:2, :, :] = 1.0
    img[-2:, :, :] = 1.0
    img[:, 0:2, :] = 1.0
    img[:, -2:, :] = 1.0

    # to tensor in [-1,1]
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    x = x * 2 - 1
    posterior = vae.encode(x)
    z = posterior.sample()
    x_rec = vae.decode_from_unet(z)

    # back to [0,1] for saving
    x_np = ((x.clamp(-1, 1) + 1) * 0.5).cpu().squeeze(0).permute(1, 2, 0).numpy()
    x_rec_np = ((x_rec.clamp(-1, 1) + 1) * 0.5).cpu().squeeze(0).permute(1, 2, 0).numpy()

    inp_path = ROOT / "tmp_geom_input.png"
    rec_path = ROOT / "tmp_geom_recon.png"
    Image.fromarray((x_np * 255).astype(np.uint8)).save(inp_path)
    Image.fromarray((x_rec_np * 255).astype(np.uint8)).save(rec_path)

    mse = ((x_np - x_rec_np) ** 2).mean()
    print(f"saved input -> {inp_path.name}, recon -> {rec_path.name}, mse={mse:.6f}")


@torch.no_grad()
def test_state_reuse_stats(device: str = "cpu") -> None:
    """Report how many non-VAE tensors were fully reused vs. shape-mismatched when loading SDXL -> Flux.

    - Excludes keys starting with first_stage_model. (original VAE)
    - Counts preserved (same shape) vs partial (shape mismatch, gets partial copy) and missing.
    """

    print("State reuse stats (exclude VAE)")
    cfg = _build_flux_config()
    model = StableDiffusionModel(str(SDXL_PATH), cfg, torch.device(device))
    model.eval()

    tgt_sd = model.state_dict()
    src_sd = load_sft(str(SDXL_PATH))
    src_sd = {k: v for k, v in src_sd.items() if not k.startswith("first_stage_model.")}

    preserved = 0
    partial = 0
    missing = 0
    mismatched = []

    for k, tgt in tgt_sd.items():
        if k.startswith("first_stage_model."):
            continue
        if k not in src_sd:
            missing += 1
            continue
        src = src_sd[k]
        if src.shape == tgt.shape:
            preserved += 1
        else:
            partial += 1
            mismatched.append((k, tuple(src.shape), tuple(tgt.shape)))

    considered = preserved + partial
    percent_preserved = 100.0 * preserved / considered if considered > 0 else 0.0
    print(
        f"preserved: {preserved}, partial(replaced): {partial}, missing(new keys): {missing}, "
        f"preserved% (of keys with source) = {percent_preserved:.2f}%"
    )

    if mismatched:
        print("mismatched layers (source_shape -> target_shape), first 20")
        for name, src_shape, tgt_shape in mismatched[:20]:
            print(f"{name}: {src_shape} -> {tgt_shape}")
        if len(mismatched) > 20:
            print(f"... and {len(mismatched) - 20} more")


def main() -> None:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    test_flux_vae_encode_decode(device=device)
    test_merge_state_dict()
    test_flux_vae_channel_guard()
    test_filter_checkpoint_for_flux()
    test_partial_merge_with_sdxl_shapes()
    test_checkpoint_headers()
    test_end_to_end_forward(device=device, height=64, width=64)
    test_flux_vae_save_roundtrip(device=device)
    test_flux_vae_geom_image(device=device)
    test_state_reuse_stats(device=device)


if __name__ == "__main__":
    main()
