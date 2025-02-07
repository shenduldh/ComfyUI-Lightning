import torch
from torchao.quantization import float8_weight_only, int8_weight_only, quantize_
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
import types

from .utils import (
    skip_forward_orig,
    teacache_skip_forward_orig,
    fbcache_skip_forward_orig,
    mbcache_skip_forward_orig,
    Cache,
    has_affordable_memory,
    is_newer_than_ada_lovelace,
)


class ApplyFastCuDNNKernels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "switch": ("BOOLEAN", {"default": True}),
                "float32_matmul_precision": (["high", "medium"], {"default": "high"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning"
    TITLE = "Apply Fast CuDNN Kernels"

    def patch(self, model, switch, float32_matmul_precision):
        if switch:
            torch.set_float32_matmul_precision(float32_matmul_precision)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 20
        else:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.benchmark_limit = 10
        return (model,)


class ApplySageAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "use_SageAttention": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning"
    TITLE = "Apply SageAttention"

    def patch(self, model: ModelPatcher, use_SageAttention: bool):
        try:
            if use_SageAttention:
                from sageattention import sageattn
                from comfy.ldm.modules.attention import attention_sage
                from comfy.ldm.modules import attention
                from comfy.ldm.flux import math

                attention.sageattn = sageattn
                math.optimized_attention = attention_sage
        except:
            pass

        return (model,)


class ApplyTeaCacheAndSkipBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip_DoubleStreamBlocks": ("STRING", {"default": "3,6,8,12"}),
                "skip_SingleStreamBlocks": ("STRING", {"default": ""}),
                "do_teacache": ("BOOLEAN", {"default": True}),
                "rel_l1_threshold": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning"
    TITLE = "Apply TeaCache and Skip Blocks"

    def patch(
        self,
        model: ModelPatcher,
        skip_DoubleStreamBlocks: str,
        skip_SingleStreamBlocks: str,
        do_teacache: bool,
        rel_l1_threshold: float,
    ) -> tuple[ModelPatcher,]:
        cloned_model = model.clone()

        if "transformer_options" not in cloned_model.model_options:
            cloned_model.model_options["transformer_options"] = {}
        cloned_model.model_options["transformer_options"]["ds_skip_blocks"] = [
            int(x) for x in skip_DoubleStreamBlocks.split(",") if x
        ]
        cloned_model.model_options["transformer_options"]["ss_skip_blocks"] = [
            int(x) for x in skip_SingleStreamBlocks.split(",") if x
        ]

        dm: torch.nn.Module = cloned_model.get_model_object("diffusion_model")
        if isinstance(dm, torch._dynamo.OptimizedModule):
            dm: torch.nn.Module = getattr(dm, "_orig_mod", dm)

        if do_teacache:
            cloned_model.model_options["transformer_options"][
                "rel_l1_threshold"
            ] = rel_l1_threshold
            dm.forward_orig = types.MethodType(teacache_skip_forward_orig, dm)
        else:
            dm.forward_orig = types.MethodType(skip_forward_orig, dm)

        return (cloned_model,)


class ApplyFBCacheAndSkipBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip_DoubleStreamBlocks": ("STRING", {"default": "3,6,8,12"}),
                "skip_SingleStreamBlocks": ("STRING", {"default": ""}),
                "fbcache_threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "start": (
                    "FLOAT",
                    {"default": 0.0, "step": 0.01, "max": 1.0, "min": 0.0},
                ),
                "end": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.01, "max": 1.0, "min": 0.0},
                ),
                "max_consecutive_cache_hits": ("INT", {"default": -1, "min": -1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning"
    TITLE = "Apply FBCache and Skip Blocks"

    def patch(
        self,
        model: ModelPatcher,
        skip_DoubleStreamBlocks: str,
        skip_SingleStreamBlocks: str,
        fbcache_threshold: float,
        start: float,
        end: float,
        max_consecutive_cache_hits: int,
    ) -> tuple[ModelPatcher,]:
        cloned_model = model.clone()

        if "transformer_options" not in cloned_model.model_options:
            cloned_model.model_options["transformer_options"] = {}
        cloned_model.model_options["transformer_options"]["ds_skip_blocks"] = [
            int(x) for x in skip_DoubleStreamBlocks.split(",") if x
        ]
        cloned_model.model_options["transformer_options"]["ss_skip_blocks"] = [
            int(x) for x in skip_SingleStreamBlocks.split(",") if x
        ]

        dm: torch.nn.Module = cloned_model.get_model_object("diffusion_model")
        if isinstance(dm, torch._dynamo.OptimizedModule):
            dm: torch.nn.Module = getattr(dm, "_orig_mod", dm)

        cloned_model.model_options["transformer_options"][
            "fbcache_threshold"
        ] = fbcache_threshold
        if max_consecutive_cache_hits >= 0 or start > 0 or end < 1:
            model_sampling = model.get_model_object("model_sampling")
            start_sigma, end_sigma = (
                float(model_sampling.percent_to_sigma(i)) for i in (start, end)
            )
            consecutive_cache_hits = 0

            def validator(use_cached, curr_timestep):
                nonlocal consecutive_cache_hits
                use_cached = (
                    use_cached
                    and end_sigma <= curr_timestep <= start_sigma
                    and consecutive_cache_hits < max_consecutive_cache_hits
                )
                consecutive_cache_hits = consecutive_cache_hits + 1 if use_cached else 0
                return use_cached

        else:
            validator = lambda *args: args[0]
        cloned_model.model_options["transformer_options"]["validator"] = validator

        dm.forward_orig = types.MethodType(fbcache_skip_forward_orig, dm)

        return (cloned_model,)


class ApplyMBCacheAndSkipBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip_DoubleStreamBlocks": ("STRING", {"default": "3,6,8,12"}),
                "skip_SingleStreamBlocks": ("STRING", {"default": ""}),
                "default_cache_threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "dsb_cache_thresholds": ("STRING", {"default": ""}),
                # "ssb_cache_thresholds": ("STRING", {"default": ""}),
                "start": (
                    "FLOAT",
                    {"default": 0.0, "step": 0.01, "max": 1.0, "min": 0.0},
                ),
                "end": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.01, "max": 1.0, "min": 0.0},
                ),
                "max_consecutive_cache_hits": ("INT", {"default": -1, "min": -1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning"
    TITLE = "Apply MBCache and Skip Blocks"

    def patch(
        self,
        model: ModelPatcher,
        skip_DoubleStreamBlocks: str,
        skip_SingleStreamBlocks: str,
        default_cache_threshold: float,
        dsb_cache_thresholds: str,
        # ssb_cache_thresholds: str,
        start: float,
        end: float,
        max_consecutive_cache_hits: int,
    ) -> tuple[ModelPatcher,]:
        cloned_model = model.clone()

        # skip related
        if "transformer_options" not in cloned_model.model_options:
            cloned_model.model_options["transformer_options"] = {}
        cloned_model.model_options["transformer_options"]["ds_skip_blocks"] = [
            int(x) for x in skip_DoubleStreamBlocks.split(",") if x
        ]
        cloned_model.model_options["transformer_options"]["ss_skip_blocks"] = [
            int(x) for x in skip_SingleStreamBlocks.split(",") if x
        ]

        dm: torch.nn.Module = cloned_model.get_model_object("diffusion_model")
        if isinstance(dm, torch._dynamo.OptimizedModule):
            dm: torch.nn.Module = getattr(dm, "_orig_mod", dm)

        # cache related
        if max_consecutive_cache_hits >= 0 or start > 0 or end < 1:
            model_sampling = model.get_model_object("model_sampling")
            start_sigma, end_sigma = (
                float(model_sampling.percent_to_sigma(i)) for i in (start, end)
            )
            consecutive_cache_hits = 0

            def validator(use_cached, curr_timestep):
                nonlocal consecutive_cache_hits
                use_cached = (
                    use_cached
                    and end_sigma <= curr_timestep <= start_sigma
                    and consecutive_cache_hits < max_consecutive_cache_hits
                )
                consecutive_cache_hits = consecutive_cache_hits + 1 if use_cached else 0
                return use_cached

        else:
            validator = lambda *args: args[0]
        cloned_model.model_options["transformer_options"]["validator"] = validator

        thresholds = [default_cache_threshold] * len(dm.double_blocks)
        for i, t in enumerate(dsb_cache_thresholds.split(",")):
            t = t.strip()
            if t:
                thresholds[i] = float(t)
        cloned_model.model_options["transformer_options"][
            "dsb_cache_thresholds"
        ] = thresholds

        # thresholds = [default_cache_threshold] * len(dm.single_blocks)
        # for i, t in enumerate(ssb_cache_thresholds.split(",")):
        #     t = t.strip()
        #     if t:
        #         thresholds[i] = float(t)
        # cloned_model.model_options["transformer_options"][
        #     "ssb_cache_thresholds"
        # ] = thresholds

        cloned_model.model_options["transformer_options"][
            "previous_ds_comparisons"
        ] = Cache()
        # cloned_model.model_options["transformer_options"][
        #     "previous_ss_comparisons"
        # ] = Cache()
        cloned_model.model_options["transformer_options"][
            "previous_residuals"
        ] = Cache()

        dm.forward_orig = types.MethodType(mbcache_skip_forward_orig, dm)

        return (cloned_model,)


class CompileAndQuantizeModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "do_compile": ("BOOLEAN", {"default": True}),
                "dynamic": ("BOOLEAN", {"default": False}),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "backend": (
                    torch.compiler.list_backends(),
                    {"default": "inductor"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "compile_and_quantize"
    CATEGORY = "Lightning"
    TITLE = "Compile and Quantize Model"

    _quantized = False
    _last_compiled_cfg = None

    def compile_and_quantize(
        self,
        model: ModelPatcher,
        vae: VAE,
        do_compile: bool,
        dynamic: bool,
        fullgraph: bool,
        backend: str,
    ) -> tuple[ModelPatcher, VAE]:
        cloned_model = model.clone()
        dm: torch.nn.Module = cloned_model.get_model_object("diffusion_model")
        ae: torch.nn.Module = getattr(vae, "first_stage_model")
        if isinstance(dm, torch._dynamo.OptimizedModule):
            dm: torch.nn.Module = getattr(dm, "_orig_mod", dm)
        if isinstance(ae, torch._dynamo.OptimizedModule):
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
            if dynamic:  # fix rope to prevent dynamic compilation error
                from comfy.ldm.flux import layers
                from .utils import fixed_rope

                layers.rope = fixed_rope

            curr_compiled_cfg = str(dynamic) + str(fullgraph) + backend
            if curr_compiled_cfg == self._last_compiled_cfg:
                return cloned_model, vae
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

            cloned_model.add_object_patch("diffusion_model", dm)
            setattr(vae, "first_stage_model", ae)

        return cloned_model, vae


NODE_CLASS_MAPPINGS = {
    "ApplyTeaCacheAndSkipBlocks": ApplyTeaCacheAndSkipBlocks,
    "ApplyFBCacheAndSkipBlocks": ApplyFBCacheAndSkipBlocks,
    "ApplyMBCacheAndSkipBlocks": ApplyMBCacheAndSkipBlocks,
    "ApplySageAttention": ApplySageAttention,
    "CompileAndQuantizeModel": CompileAndQuantizeModel,
}
