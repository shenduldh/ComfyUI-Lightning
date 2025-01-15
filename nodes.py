"""
1. TeaCache
2. skip blocks
3. compile and quantize
4. fast CuDNN attention kernels
"""

import torch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20

from torchao.quantization import float8_weight_only, int8_weight_only, quantize_
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
import types
from utils import custom_forward_orig, has_affordable_memory, is_newer_than_ada_lovelace


class ApplyTeaCacheAndSkipBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip_DoubleStreamBlocks": ("STRING", {"default": "3,6,8,12"}),
                "skip_SingleStreamBlocks": ("STRING", {"default": ""}),
                "rel_l1_thresh": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Lightning"
    TITLE = "Apply TeaCache and Skip Blocks"

    def apply(
        self,
        model: ModelPatcher,
        skip_DoubleStreamBlocks: str,
        skip_SingleStreamBlocks: str,
        rel_l1_thresh: float,
        steps: int,
    ):
        dm: torch.nn.Module = model.get_model_object("diffusion_model")
        if hasattr(dm, "_orig_mod"):
            dm: torch.nn.Module = dm._orig_mod

        dm.__class__.ds_skip_blocks = [
            int(x) for x in skip_DoubleStreamBlocks.split(",") if x
        ]
        dm.__class__.ss_skip_blocks = [
            int(x) for x in skip_SingleStreamBlocks.split(",") if x
        ]
        dm.__class__.cnt = 0
        dm.__class__.rel_l1_thresh = rel_l1_thresh
        dm.__class__.steps = steps
        dm.__class__.forward_orig = types.MethodType(custom_forward_orig, dm)

        return model


class CompileAndQuantizeModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "do_compile": ("BOOLEAN", {"default": True}),
                "dynamic": ("BOOLEAN", {"default": True}),
                "fullgraph": ("BOOLEAN", {"default": True}),
                "backend": (
                    torch.compiler.list_backends(),
                    {"default": "inductor"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "apply"
    CATEGORY = "Lightning"
    TITLE = "Compile and Quantize Model"

    _quantized = False
    _last_compiled_cfg = None

    def apply(
        self,
        model: ModelPatcher,
        vae: VAE,
        do_compile: bool,
        dynamic: bool,
        fullgraph: bool,
        backend: str,
    ):
        dm: torch.nn.Module = model.get_model_object("diffusion_model")
        ae: torch.nn.Module = vae.first_stage_model
        if hasattr(dm, "_orig_mod"):
            dm: torch.nn.Module = getattr(dm, "_orig_mod", dm)
            ae: torch.nn.Module = getattr(ae, "_orig_mod", ae)

        if not self._quantized:
            if ae.parameters().__next__().dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2fnuz,
                torch.int8,
            ):
                pass
            elif is_newer_than_ada_lovelace(torch.device(0)):
                quantize_(ae, float8_weight_only())
            else:
                quantize_(ae, int8_weight_only())
            self._quantized = True

        if do_compile:
            curr_compiled_cfg = str(dynamic) + str(fullgraph) + backend
            if curr_compiled_cfg == self._last_compiled_cfg:
                return model, vae
            self._last_compiled_cfg == curr_compiled_cfg

            compile_mode = (
                "reduce-overhead"
                if has_affordable_memory(torch.device(0))
                else "default"
            )

            dm = dm.to(memory_format=torch.channels_last)
            dm = torch.compile(
                dm,
                mode=compile_mode,
                fullgraph=fullgraph,
                backend=backend,
                dynamic=dynamic,
            )

            ae = ae.to(memory_format=torch.channels_last)
            ae = torch.compile(
                ae,
                mode=compile_mode,
                fullgraph=fullgraph,
                backend=backend,
                dynamic=dynamic,
            )

            model.model.diffusion_model = dm
            vae.first_stage_model = ae

        return model, vae


NODE_CLASS_MAPPINGS = {
    "ApplyTeaCacheAndSkipBlocks": ApplyTeaCacheAndSkipBlocks,
    "CompileAndQuantizeModel": CompileAndQuantizeModel,
}
