import torch
from torch import Tensor

from .diffusion.model.sana_multi_scale import (
    get_2d_sincos_pos_embed,
    _xformers_available,
)


def are_tensors_similar(tensor1: Tensor, tensor2: Tensor, threshold: float):
    assert tensor1.shape == tensor2.shape
    tensor1 = tensor1.clone().float()
    tensor2 = tensor2.clone().float()
    sim = (tensor1 * tensor2).sum()
    sim /= tensor1.norm(p=2) * tensor2.norm(p=2) + 1e-8
    diff = (-sim + 1.0) / 2.0
    return diff < threshold


def mbcache_forward_orig(self, x, timestep, y, mask=None, data_info=None, **kwargs):
    """
    Forward pass of Sana.
    x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    t: (N,) tensor of diffusion timesteps
    y: (N, 1, 120, C) tensor of class labels
    """
    ###
    skip_blocks = self.extra_options.get("skip_blocks")
    validate_use_cached = self.extra_options.get("validator")
    block_cache_thresholds = self.extra_options.get("block_cache_thresholds")
    previous_comparisons = self.extra_options.get("previous_comparisons")
    previous_residuals = self.extra_options.get("previous_residuals")
    use_cached = False

    bs = x.shape[0]
    x = x.to(self.dtype)
    timestep = timestep.to(self.dtype)
    y = y.to(self.dtype)
    self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
    if self.use_pe:
        x = self.x_embedder(x)
        if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x.shape[1:]:
            self.pos_embed_ms = (
                torch.from_numpy(
                    get_2d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        (self.h, self.w),
                        pe_interpolation=self.pe_interpolation,
                        base_size=self.base_size,
                    )
                )
                .unsqueeze(0)
                .to(x.device)
                .to(self.dtype)
            )
        x += self.pos_embed_ms  # (N, T, D), where T = H * W / patch_size ** 2
    else:
        x = self.x_embedder(x)

    t = self.t_embedder(timestep)  # (N, D)

    t0 = self.t_block(t)
    y = self.y_embedder(y, self.training, mask=mask)  # (N, D)
    if self.y_norm:
        y = self.attention_y_norm(y)

    if mask is not None:
        mask = (
            mask.repeat(y.shape[0] // mask.shape[0], 1)
            if mask.shape[0] != y.shape[0]
            else mask
        )
        mask = mask.squeeze(1).squeeze(1)
        if _xformers_available:
            y = (
                y.squeeze(1)
                .masked_select(mask.unsqueeze(-1) != 0)
                .view(1, -1, x.shape[-1])
            )
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = mask
    elif _xformers_available:
        y_lens = [y.shape[2]] * y.shape[0]
        y = y.squeeze(1).view(1, -1, x.shape[-1])
    else:
        raise ValueError(
            f"Attention type is not available due to _xformers_available={_xformers_available}."
        )

    for i, block in enumerate(self.blocks):
        ### skip blocks
        if i in skip_blocks:
            continue

        x = block(
            x, y, t0, y_lens, (self.h, self.w), **kwargs
        )  # (N, T, D) #support grad checkpoint

        ### MBCache
        cache_key = f"block{i}"
        if previous_comparisons.has(cache_key):
            # torch._dynamo.graph_break()
            use_cached = are_tensors_similar(
                x, previous_comparisons.get(cache_key), block_cache_thresholds[i]
            )
            use_cached = validate_use_cached(use_cached, timestep[0] / 1000)
        previous_comparisons.set(cache_key, x)
        if use_cached:
            break

    if use_cached:
        x += previous_residuals.get(cache_key)
    else:
        for k in previous_comparisons.keys():
            previous_residuals.set(k, x - previous_comparisons.get(k))

    x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
    x = self.unpatchify(x)  # (N, out_channels, H, W)
    return x
