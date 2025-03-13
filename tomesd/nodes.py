import types
import torch
from comfy.model_patcher import ModelPatcher
from comfy.ldm.flux.model import Flux, DoubleStreamBlock, SingleStreamBlock

from .utils import (
    DoubleStreamBlock_forward,
    SingleStreamBlock_forward,
    Flux_forward_orig,
)


class ApplyTokenMerging:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning/tomesd"
    TITLE = "Apply Token Merging"

    def patch(self, model: ModelPatcher):
        cloned_model = model.clone()

        dm: Flux = cloned_model.get_model_object("diffusion_model")
        if isinstance(dm, torch._dynamo.OptimizedModule):
            dm: Flux = getattr(dm, "_orig_mod", dm)
        dm.forward_orig = types.MethodType(Flux_forward_orig, dm)

        for idx, block in enumerate(dm.double_blocks):  # 19
            if isinstance(block, torch._dynamo.OptimizedModule):
                block: DoubleStreamBlock = getattr(block, "_orig_mod", block)
            block.forward = types.MethodType(DoubleStreamBlock_forward, block)

        for idx, block in enumerate(dm.single_blocks):  # 38
            if isinstance(block, torch._dynamo.OptimizedModule):
                block: SingleStreamBlock = getattr(block, "_orig_mod", block)
            block.forward = types.MethodType(SingleStreamBlock_forward, block)

        return (cloned_model,)


NODE_CLASS_MAPPINGS = {"ApplyTokenMerging": ApplyTokenMerging}
