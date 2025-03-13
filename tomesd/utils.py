import math
import torch
from torch import Tensor
from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import timestep_embedding
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1), dim - 1 if dim < 0 else dim, index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: torch.Generator = None,
) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(
                sy * sx,
                size=(hsy, wsx, 1),
                device=generator.device,
                generator=generator,
            ).to(metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(
            hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64
        )
        idx_buffer_view.scatter_(
            dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype)
        )
        idx_buffer_view = (
            idx_buffer_view.view(hsy, wsx, sy, sx)
            .transpose(1, 2)
            .reshape(hsy * sy, wsx * sx)
        )

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[: (hsy * sy), : (wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, r, c),
            src=src,
        )

        return out

    return merge, unmerge


def compute_merge(
    metric,
    orig_size=(512, 512),
    merge_ratio=0.5,
    sx=2,
    sy=2,
):
    metric = metric.clone()

    orig_width, orig_height = orig_size
    orig_size = orig_width * orig_height
    downsample = int(math.ceil(math.sqrt(orig_size // metric.shape[1])))
    w = int(math.ceil(orig_width / downsample))
    h = int(math.ceil(orig_height / downsample))
    r = int(metric.shape[1] * merge_ratio)

    merge_func, unmerge_func = bipartite_soft_matching_random2d(
        metric, w, h, sx, sy, r, generator=torch.Generator(metric.device)
    )
    return merge_func, unmerge_func


def Flux_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError(
                "Didn't get guidance strength for guidance distilled model."
            )
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask},
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                )
                return out

            out = blocks_replace[("single_block", i)](
                {"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask},
                {"original_block": block_wrap},
            )
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


def DoubleStreamBlock_forward(
    self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None
):
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(
        img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
    ).permute(2, 0, 3, 1, 4)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(
        txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
    ).permute(2, 0, 3, 1, 4)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    if self.flipped_img_txt:
        # run actual attention
        q = torch.cat((img_q, txt_q), dim=2)
        k = torch.cat((img_k, txt_k), dim=2)
        v = torch.cat((img_v, txt_v), dim=2)
        attn = attention(q, k, v, pe=pe[:, :, : q.size(2), :, :, :], mask=attn_mask)
        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
    else:
        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe=pe[:, :, : q.size(2), :, :, :], mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    ###
    merge, unmerge = compute_merge(img_attn, (1024, 1024))
    img_attn = unmerge(self.img_attn.proj(merge(img_attn)))

    # calculate the img bloks
    img = img + img_mod1.gate * img_attn
    img = img + img_mod2.gate * self.img_mlp(
        (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
    )

    # calculate the txt bloks
    txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt += txt_mod2.gate * self.txt_mlp(
        (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
    )

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt


def SingleStreamBlock_forward(
    self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None
) -> Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(
        self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
    )

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(
        2, 0, 3, 1, 4
    )
    q, k = self.norm(q, k, v)

    # compute attention
    attn = attention(q, k, v, pe=pe, mask=attn_mask)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x
