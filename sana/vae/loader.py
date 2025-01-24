import torch
import comfy.sd
import comfy.model_patcher
from comfy import model_management
from copy import deepcopy

from .model.dc_ae import DCAE, DCAEConfig, EncoderConfig, DecoderConfig
from .configs import VAE_CONFIG


class SanaVAE(comfy.sd.VAE):
    """interface with ComfyUI"""

    def __init__(self, model_path, model_config: DCAEConfig, dtype=None, device=None):
        self.memory_used_encode = lambda shape, dtype: (
            1767 * shape[2] * shape[3]
        ) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (
            2178 * shape[2] * shape[3] * 64
        ) * model_management.dtype_size(dtype)
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )
        self.working_dtypes = [torch.bfloat16, torch.float32]

        model = DCAE(model_config)
        self.first_stage_model = model.eval()
        sd = comfy.utils.load_torch_file(model_path)
        self.first_stage_model.load_state_dict(sd, strict=False)

        if device is None:
            device = model_management.vae_device()
        self.device = device
        offload_device = model_management.vae_offload_device()
        if dtype is None:
            dtype = model_management.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = model_management.intermediate_device()

        self.patcher = comfy.model_patcher.ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )


def load_sana_vae(model_path, dtype):
    config = deepcopy(VAE_CONFIG)
    config["encoder"] = EncoderConfig(**config["encoder"])
    config["decoder"] = DecoderConfig(**config["decoder"])
    return SanaVAE(model_path, DCAEConfig(**config), dtype)
