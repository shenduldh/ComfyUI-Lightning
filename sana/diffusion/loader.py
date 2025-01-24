import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds
import torch
from comfy import model_management, model_detection
from comfy.latent_formats import LatentFormat

from .model.sana_multi_scale import SanaMS
from .configs import DIFFUSION_CONFIGS


class SanaLatentFormat(LatentFormat):
    scale_factor = 0.41407
    latent_channels = 32


class SanaBase(comfy.supported_models_base.BASE):
    unet_extra_config = {"disable_unet_model_creation": False}
    latent_format = SanaLatentFormat
    sampling_settings = {"shift": 3.0}
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return SanaBaseModel(self, self.model_type(state_dict, prefix), device, SanaMS)

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.FLOW


class SanaBaseModel(comfy.model_base.BaseModel):
    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        cn_hint = kwargs.get("cn_hint", None)
        if cn_hint is not None:
            out["cn_hint"] = comfy.conds.CONDRegular(cn_hint)

        return out


def load_sana_diffusion(model_path: str, model_options: dict = {}):
    state_dict = comfy.utils.load_torch_file(model_path)
    sd = state_dict.get("model", state_dict)

    # get target state dict
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    # convert diffusers
    pass

    # get device
    offload_device = model_management.unet_offload_device()
    load_device = model_management.get_torch_device()

    # create model config for ComfyUI
    model_type = model_options.get("model_type", None)
    assert model_type in DIFFUSION_CONFIGS.keys()
    model_config = SanaBase(DIFFUSION_CONFIGS[model_type])

    # set inference dtype
    dtype = model_options.get("dtype", None)
    if dtype is None:
        parameters = comfy.utils.calculate_parameters(sd)
        weight_dtype = comfy.utils.weight_dtype(sd)
        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if weight_dtype is not None and model_config.scaled_fp8 is None:
            unet_weight_dtype.append(weight_dtype)
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype
        )
    else:
        unet_dtype = dtype
    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get(
        "custom_operations", model_config.custom_operations
    )
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    # create base model for ComfyUI
    model = model_config.get_model(sd)
    model = model.to(offload_device)
    model.load_model_weights(sd)
    model.diffusion_model.eval()
    if unet_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if model_options.get("convert_model_dtype", False):
            model.diffusion_model.to(unet_dtype)

    # patch model
    return comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )
