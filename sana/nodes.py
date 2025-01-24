import os
import folder_paths
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from nodes import EmptyLatentImage
from comfy import model_management
from comfy.model_patcher import ModelPatcher
import types

from .diffusion.loader import load_sana_diffusion, DIFFUSION_CONFIGS
from .vae.loader import load_sana_vae
from ..utils import Cache
from .utils import mbcache_forward_orig


class SanaDiffusionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (list(DIFFUSION_CONFIGS.keys()),),
                "weight_dtype": (
                    [
                        "default",
                        "fp32",
                        "fp16",
                        "bf16",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                    ],
                ),
                "convert_model_dtype": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Lightning/Sana"
    TITLE = "Load Sana Diffusion Model"

    def load(
        self,
        model_name: str,
        model_type: str,
        weight_dtype: str,
        convert_model_dtype: bool,
    ):
        model_options = {
            "model_type": model_type,
            "convert_model_dtype": convert_model_dtype,
        }
        model_options["dtype"] = {
            "default": None,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e4m3fn_fast": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }[weight_dtype]
        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["fp8_optimizations"] = True

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        model = load_sana_diffusion(model_path, model_options)
        return (model,)


class SanaCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    os.listdir(os.path.join(folder_paths.models_dir, "text_encoders")),
                ),
                "max_length": (
                    "INT",
                    {"default": 300, "min": 1, "max": 9999, "step": 1},
                ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["auto", "fp32", "fp16", "bf16"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("GEMMA",)
    FUNCTION = "load"
    CATEGORY = "Lightning/Sana"
    TITLE = "Load Sana CLIP"

    def load(self, model_name: str, max_length: int, device, dtype):
        dtype = {
            "auto": model_management.text_encoder_dtype(),
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[dtype]
        model_path = os.path.join(folder_paths.models_dir, "text_encoders", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "right"

        text_encoder = (
            AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
            .get_decoder()
            .to(device)
            .eval()
        )

        return (
            {
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
                "max_length": max_length,
            },
        )


class SanaVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"),),
                "dtype": (["auto", "fp32", "fp16", "bf16"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "Lightning/Sana"
    TITLE = "Load Sana VAE"

    def load(self, model_name, dtype: str):
        dtype = {
            "auto": model_management.vae_dtype(),
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[dtype]
        model_path = folder_paths.get_full_path("vae", model_name)
        vae = load_sana_vae(model_path, dtype)
        return (vae,)


class SanaTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "GEMMA": ("GEMMA",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Lightning/Sana"
    TITLE = "Sana Text Encode"

    def encode(self, text: str, GEMMA):
        tokenizer = GEMMA["tokenizer"]
        text_encoder = GEMMA["text_encoder"]
        max_length = GEMMA["max_length"]

        with torch.no_grad():
            tokens = tokenizer(
                [text],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(text_encoder.device)
            select_index = [0] + list(range(-max_length + 1, 0))
            embeddings = text_encoder(tokens.input_ids, tokens.attention_mask)[0]
            embeddings = embeddings[:, None][:, :, select_index]
            embedding_masks = (
                tokens.attention_mask[:, select_index]
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(1, 1, 1, embeddings.size(-1))
            )
            embeddings *= embedding_masks
            embeddings = torch.cat([embeddings, embedding_masks], dim=1)

        return ([[embeddings, {}]],)


class SanaEmptyLatentImage(EmptyLatentImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 16384, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 16384, "step": 32},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Lightning/Sana"
    TITLE = "Sana Empty Latent Image"

    def generate(self, width: int, height: int, batch_size: int):
        latent = torch.zeros(
            [batch_size, 32, height // 32, width // 32], device=self.device
        )
        return ({"samples": latent},)


class ApplyMBCacheAndSkipBlocksForSana:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip_blocks": ("STRING", {"default": "14,17"}),
                "default_cache_threshold": (
                    "FLOAT",
                    {"default": 0.0002, "min": 0, "max": 1.0, "step": 0.00001},
                ),
                "block_cache_thresholds": ("STRING", {"default": ""}),
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
    CATEGORY = "Lightning/Sana"
    TITLE = "Apply MBCache and Skip Blocks for Sana"

    def patch(
        self,
        model: ModelPatcher,
        skip_blocks: str,
        default_cache_threshold: float,
        block_cache_thresholds: str,
        start: float,
        end: float,
        max_consecutive_cache_hits: int,
    ) -> tuple[ModelPatcher,]:
        cloned_model = model.clone()

        dm: torch.nn.Module = cloned_model.get_model_object("diffusion_model")
        if isinstance(dm, torch._dynamo.OptimizedModule):
            dm: torch.nn.Module = getattr(dm, "_orig_mod", dm)
        dm.extra_options = {}

        # skip related
        dm.extra_options["skip_blocks"] = [int(x) for x in skip_blocks.split(",") if x]

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
        dm.extra_options["validator"] = validator

        thresholds = [default_cache_threshold] * len(dm.blocks)
        for i, t in enumerate(block_cache_thresholds.split(",")):
            t = t.strip()
            if t:
                thresholds[i] = float(t)
        dm.extra_options["block_cache_thresholds"] = thresholds

        dm.extra_options["previous_comparisons"] = Cache()
        dm.extra_options["previous_residuals"] = Cache()

        dm.forward_orig = types.MethodType(mbcache_forward_orig, dm)

        return (cloned_model,)


NODE_CLASS_MAPPINGS = {
    "SanaDiffusionLoader": SanaDiffusionLoader,
    "SanaCLIPLoader": SanaCLIPLoader,
    "SanaEmptyLatentImage": SanaEmptyLatentImage,
    "SanaTextEncode": SanaTextEncode,
    "SanaVAELoader": SanaVAELoader,
    "ApplyMBCacheAndSkipBlocksForSana": ApplyMBCacheAndSkipBlocksForSana,
}
