import torch
import comfy
from comfy.model_patcher import ModelPatcher
from comfy.ldm.flux.model import Flux, DoubleStreamBlock, SingleStreamBlock
import types

from .cache_functions import cache_init
from .utils import (
    toca_Flux_forward_orig,
    toca_DoubleStreamBlock_forward,
    toca_SingleStreamBlock_forward,
)


class ApplyToCa:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning/ToCa"
    TITLE = "Apply ToCa"

    def patch(self, model: ModelPatcher) -> tuple[ModelPatcher,]:
        cloned_model = model.clone()

        dm: Flux = cloned_model.get_model_object("diffusion_model")
        if isinstance(dm, torch._dynamo.OptimizedModule):
            dm: Flux = getattr(dm, "_orig_mod", dm)
        dm.forward_orig = types.MethodType(toca_Flux_forward_orig, dm)

        for block in dm.double_blocks:
            if isinstance(block, torch._dynamo.OptimizedModule):
                block: DoubleStreamBlock = getattr(block, "_orig_mod", block)
            block.forward = types.MethodType(toca_DoubleStreamBlock_forward, block)

        for block in dm.single_blocks:
            if isinstance(block, torch._dynamo.OptimizedModule):
                block: SingleStreamBlock = getattr(block, "_orig_mod", block)
            block.forward = types.MethodType(toca_SingleStreamBlock_forward, block)

        cache_dic, current = None, None

        def model_unet_function_wrapper(model_function, kwargs):
            nonlocal cache_dic, current

            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            current_timestep = timestep[0].item()

            if current_timestep == 1.0:  # init cache
                sigmas = c["transformer_options"]["sample_sigmas"]
                cache_dic, current = cache_init(sigmas)
                current["step"] = 0
                current["num_steps"] = len(sigmas) - 1

            current["t"] = current_timestep
            c["transformer_options"]["cache_dic"] = cache_dic
            c["transformer_options"]["current"] = current
            res = model_function(input, timestep, **c)
            current["step"] += 1
            return res

        cloned_model.set_model_unet_function_wrapper(model_unet_function_wrapper)

        return (cloned_model,)


NODE_CLASS_MAPPINGS = {"ApplyToCa": ApplyToCa}
