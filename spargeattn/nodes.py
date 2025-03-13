import sys
import os
from pathlib import Path
from typing import Optional
import types

import torch
import comfy.sd
from comfy.model_patcher import ModelPatcher
from comfy.ldm.flux.model import Flux, DoubleStreamBlock, SingleStreamBlock
import comfy.utils
import folder_paths


class ApplySpargeAttn:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "l1": ("FLOAT", {"default": 0.06, "step": 0.0001}),
                "pv_l1": ("FLOAT", {"default": 0.065, "step": 0.0001}),
                "enable_tuning_mode": ("BOOLEAN", {"default": False}),
                "parallel_tuning": ("BOOLEAN", {"default": False}),
                "tuned_hyperparams": (
                    [None] + folder_paths.get_filename_list("checkpoints"),
                    {"default": None},
                ),
                "skip_DoubleStreamBlocks": ("STRING", {"default": "3,6,8,12"}),
                "skip_SingleStreamBlocks": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Lightning/SpargeAttn"
    TITLE = "Apply SpargeAttn"

    def patch(
        self,
        model: ModelPatcher,
        l1: float,
        pv_l1: float,
        enable_tuning_mode: bool,
        parallel_tuning: bool,
        tuned_hyperparams: Optional[str],
        skip_DoubleStreamBlocks: str,
        skip_SingleStreamBlocks: str,
    ):
        cloned_model = model.clone()

        try:
            from .utils import (
                spargeattn_DoubleStreamBlock_forward,
                spargeattn_SingleStreamBlock_forward,
                load_sparse_attention_state_dict,
                SparseAttentionMeansim,
            )

            dm: Flux = cloned_model.get_model_object("diffusion_model")
            if isinstance(dm, torch._dynamo.OptimizedModule):
                dm: Flux = getattr(dm, "_orig_mod", dm)

            skip_DoubleStreamBlocks = [
                int(i.strip()) for i in skip_DoubleStreamBlocks.split(",") if i.strip()
            ]
            for idx, block in enumerate(dm.double_blocks):  # 19
                if idx in skip_DoubleStreamBlocks:
                    continue

                if isinstance(block, torch._dynamo.OptimizedModule):
                    block: DoubleStreamBlock = getattr(block, "_orig_mod", block)

                if not hasattr(block, "spargeattn"):
                    block.spargeattn = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1)
                    block.forward = types.MethodType(
                        spargeattn_DoubleStreamBlock_forward, block
                    )

                block.spargeattn.enable_tuning_mode = enable_tuning_mode

            skip_SingleStreamBlocks = [
                int(i.strip()) for i in skip_SingleStreamBlocks.split(",") if i.strip()
            ]
            for idx, block in enumerate(dm.single_blocks):  # 38
                if idx in skip_SingleStreamBlocks:
                    continue

                if isinstance(block, torch._dynamo.OptimizedModule):
                    block: SingleStreamBlock = getattr(block, "_orig_mod", block)

                if not hasattr(block, "spargeattn"):
                    block.spargeattn = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1)
                    block.forward = types.MethodType(
                        spargeattn_SingleStreamBlock_forward, block
                    )

                block.spargeattn.enable_tuning_mode = enable_tuning_mode

            if tuned_hyperparams is not None:
                sd_path = folder_paths.get_full_path("checkpoints", tuned_hyperparams)
                sd = comfy.utils.load_torch_file(sd_path, safe_load=True)
                load_sparse_attention_state_dict(dm, sd)

            if parallel_tuning:
                comfyui_root = Path(os.path.abspath(__file__)).resolve().parents[3]
                sys.path.insert(0, str(comfyui_root))
                os.environ["PARALLEL_TUNE"] = "1"
            else:
                os.environ["PARALLEL_TUNE"] = ""

        except Exception as e:
            print(e)

        return (cloned_model,)


class SaveSpargeAttnHyperparams:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": (
                    "STRING",
                    {"default": "spargeattn_hyperparams"},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "Lightning/SpargeAttn"
    OUTPUT_NODE = True
    TITLE = "Save Finetuned SpargeAttn Hyperparams"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    def save(self, model: ModelPatcher, filename_prefix: str):
        try:
            from .utils import extract_sparse_attention_state_dict

            dm: Flux = model.get_model_object("diffusion_model")
            if isinstance(dm, torch._dynamo.OptimizedModule):
                dm: Flux = getattr(dm, "_orig_mod", dm)

            sd = extract_sparse_attention_state_dict(dm)
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            )
            saved_path = f"{filename}_{counter:05}_.safetensors"
            saved_path = os.path.join(full_output_folder, saved_path)
            comfy.utils.save_torch_file(sd, saved_path, metadata=None)

        except Exception as e:
            print(e)

        return {}


NODE_CLASS_MAPPINGS = {
    "ApplySpargeAttn": ApplySpargeAttn,
    "SaveSpargeAttnHyperparams": SaveSpargeAttnHyperparams,
}
