import torch
from torch import nn
from safetensors.torch import load_file as load_sft_file

from models.flux_autoencoder import AutoEncoder, AutoEncoderParams


def _load_flux_state(path: str):
    if path is None:
        return None
    if path.endswith((".sft", ".safetensors")):
        return load_sft_file(path)
    state = torch.load(path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    return {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}


class _Posterior:
    """Lightweight wrapper to mimic diffusers VAE posterior API."""

    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    def sample(self) -> torch.Tensor:
        return self.latent


class FluxVAE(nn.Module):
    """Run the Flux autoencoder inside the SDXL training stack (no adapters).

    Assumes UNet in/out channels == Flux VAE z_channels (16 by default).
    """

    def __init__(
        self,
        ae_params: dict | AutoEncoderParams,
        ae_path: str | None = None,
        target_channels: int | None = None,
        target_scale_factor: float | None = None,
    ) -> None:
        super().__init__()
        if isinstance(ae_params, AutoEncoderParams):
            self.ae_params = ae_params
        else:
            self.ae_params = AutoEncoderParams(**ae_params)

        self.target_channels = target_channels or self.ae_params.z_channels
        self.target_scale_factor = target_scale_factor or self.ae_params.scale_factor
        self.ae = AutoEncoder(self.ae_params)
        assert (
            self.target_channels == self.ae_params.z_channels
        ), "UNet channels must match Flux VAE z_channels; adapters are disabled."
        self.is_flux_vae = True

        if ae_path is not None:
            state = _load_flux_state(ae_path)
            if state is not None:
                missing, unexpected = self.ae.load_state_dict(state, strict=False)
                if len(missing) > 0:
                    print(f"Flux VAE missing keys: {missing}")
                if len(unexpected) > 0:
                    print(f"Flux VAE unexpected keys: {unexpected}")

    def encode(self, x: torch.Tensor) -> _Posterior:
        z = self.ae.encode(x)
        return _Posterior(z)

    @torch.no_grad()
    def encode_for_unet(self, x: torch.Tensor, bsz: int) -> torch.Tensor:
        latents = []
        x = x.float()
        scale = self.target_scale_factor / self.ae_params.scale_factor
        for i in range(0, x.shape[0], bsz):
            chunk = x[i : i + bsz]
            z = self.ae.encode(chunk)
            if scale != 1.0:
                z = z * scale
            latents.append(z)
        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def decode_from_unet(self, z_unet: torch.Tensor) -> torch.Tensor:
        z_unet = z_unet.float()
        scale = self.ae_params.scale_factor / self.target_scale_factor
        if scale != 1.0:
            z_unet = z_unet * scale
        return self.ae.decode(z_unet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_from_unet(x)
