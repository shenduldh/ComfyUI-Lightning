import math
import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.layers import timestep_embedding, DoubleStreamBlock

from .cache_functions import cal_type
from .cache_functions import force_init, cache_cutfresh, update_cache


def dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weight += attn_bias

    attn_map = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_map, dropout_p, train=True)
    out = torch.matmul(attn_weight, value)
    out = rearrange(out, "B H L D -> B L (H D)")
    return out, attn_map.mean(dim=1).mean(dim=1)


def attention(
    q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None, **kwargs
) -> Tensor:
    cache_dic = kwargs.get("cache_dic", None)
    current = kwargs.get("current", None)

    q, k = apply_rope(q, k, pe)
    heads = q.shape[1]

    if cache_dic is not None and cache_dic["cache_type"] == "attention":
        x, score = dot_product_attention(q, k, v, attn_mask=mask)
        cache_dic["attn_map"][-1][current["stream"]][current["layer"]]["total"] = score
    else:
        heads = q.shape[1]
        x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)

    return x


def toca_Flux_forward_orig(
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
    cache_dic = transformer_options.get("cache_dic", None)
    current = transformer_options.get("current", None)

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

    cal_type(cache_dic, current)

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        current["layer"] = i

        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    cache_dic=args.get("cache_dic"),
                    current=args.get("current"),
                )
                return out

            out = blocks_replace[("double_block", i)](
                {
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": pe,
                    "attn_mask": attn_mask,
                    "cache_dic": cache_dic,
                    "current": current,
                },
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
                attn_mask=attn_mask,
                cache_dic=cache_dic,
                current=current,
            )

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        current["layer"] = i

        if ("single_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    cache_dic=args.get("cache_dic"),
                    current=args.get("current"),
                )
                return out

            out = blocks_replace[("single_block", i)](
                {
                    "img": img,
                    "vec": vec,
                    "pe": pe,
                    "attn_mask": attn_mask,
                    "cache_dic": cache_dic,
                    "current": current,
                },
                {"original_block": block_wrap},
            )
            img = out["img"]
        else:
            img = block(
                img,
                vec=vec,
                pe=pe,
                attn_mask=attn_mask,
                cache_dic=cache_dic,
                current=current,
            )

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


def toca_DoubleStreamBlock_forward(
    self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, **kwargs
):
    cache_dic = kwargs.get("cache_dic", None)
    current = kwargs.get("current", None)

    if cache_dic is None:
        img, txt = DoubleStreamBlock.forward(self, img, txt, vec, pe, attn_mask)
    else:
        current["stream"] = "double_stream"

        if current["type"] == "full":
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

            # prepare image for attention
            img_modulated = self.img_norm1(img)
            img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = img_qkv.view(
                img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)

            ######
            if cache_dic["cache_type"] == "k-norm":
                img_k_norm = img_k.norm(dim=-1, p=2).mean(dim=1)
                cache_dic["k-norm"][-1][current["stream"]][current["layer"]][
                    "img_mlp"
                ] = img_k_norm
            elif cache_dic["cache_type"] == "v-norm":
                img_v_norm = img_v.norm(dim=-1, p=2).mean(dim=1)
                cache_dic["v-norm"][-1][current["stream"]][current["layer"]][
                    "img_mlp"
                ] = img_v_norm

            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

            # prepare txt for attention
            txt_modulated = self.txt_norm1(txt)
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = txt_qkv.view(
                txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)

            ######
            if cache_dic["cache_type"] == "k-norm":
                txt_k_norm = txt_k.norm(dim=-1, p=2).mean(dim=1)
                cache_dic["k-norm"][-1][current["stream"]][current["layer"]][
                    "txt_mlp"
                ] = txt_k_norm
            elif cache_dic["cache_type"] == "v-norm":
                txt_v_norm = txt_v.norm(dim=-1, p=2).mean(dim=1)
                cache_dic["v-norm"][-1][current["stream"]][current["layer"]][
                    "txt_mlp"
                ] = txt_v_norm

            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

            if self.flipped_img_txt:
                # run actual attention
                attn = attention(
                    torch.cat((img_q, txt_q), dim=2),
                    torch.cat((img_k, txt_k), dim=2),
                    torch.cat((img_v, txt_v), dim=2),
                    pe=pe,
                    mask=attn_mask,
                    cache_dic=cache_dic,
                    current=current,
                )

                img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
            else:
                # run actual attention
                attn = attention(
                    torch.cat((txt_q, img_q), dim=2),
                    torch.cat((txt_k, img_k), dim=2),
                    torch.cat((txt_v, img_v), dim=2),
                    pe=pe,
                    mask=attn_mask,
                    cache_dic=cache_dic,
                    current=current,
                )

                txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

            ######
            cache_dic["cache"][-1]["double_stream"][current["layer"]]["attn"] = attn
            cache_dic["txt_shape"] = txt.shape[1]

            ######
            if cache_dic["cache_type"] == "attention":
                cache_dic["attn_map"][-1][current["stream"]][current["layer"]][
                    "txt_mlp"
                ] = cache_dic["attn_map"][-1][current["stream"]][current["layer"]][
                    "total"
                ][
                    :, : txt.shape[1]
                ]
                cache_dic["attn_map"][-1][current["stream"]][current["layer"]][
                    "img_mlp"
                ] = cache_dic["attn_map"][-1][current["stream"]][current["layer"]][
                    "total"
                ][
                    :, txt.shape[1] :
                ]

            ######
            current["module"] = "img_mlp"
            force_init(cache_dic=cache_dic, current=current, tokens=img)

            # calculate the img bloks
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)
            img_mlp_out = self.img_mlp(
                (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
            )
            img = img + img_mod2.gate * img_mlp_out

            ######
            cache_dic["cache"][-1]["double_stream"][current["layer"]][
                "img_mlp"
            ] = img_mlp_out
            current["module"] = "txt_mlp"
            force_init(cache_dic=cache_dic, current=current, tokens=txt)

            # calculate the txt bloks
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
            txt_mlp_out = self.txt_mlp(
                (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
            )
            txt = txt + txt_mod2.gate * txt_mlp_out

            ######
            cache_dic["cache"][-1]["double_stream"][current["layer"]][
                "txt_mlp"
            ] = txt_mlp_out

            if txt.dtype == torch.float16:
                txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        elif current["type"] == "ToCa":
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

            # run actual attention
            attn = cache_dic["cache"][-1]["double_stream"][current["layer"]]["attn"]
            if self.flipped_img_txt:
                img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
            else:
                txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

            # calculate the img bloks
            current["module"] = "img_mlp"
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)
            fresh_indices, fresh_tokens_img = cache_cutfresh(
                cache_dic=cache_dic, tokens=img, current=current
            )
            fresh_tokens_img = self.img_mlp(
                (1 + img_mod2.scale) * self.img_norm2(fresh_tokens_img) + img_mod2.shift
            )
            update_cache(
                fresh_indices=fresh_indices,
                fresh_tokens=fresh_tokens_img,
                cache_dic=cache_dic,
                current=current,
            )
            img = (
                img
                + img_mod2.gate
                * cache_dic["cache"][-1]["double_stream"][current["layer"]]["img_mlp"]
            )

            # calculate the txt bloks
            current["module"] = "txt_mlp"
            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
            fresh_indices, fresh_tokens_txt = cache_cutfresh(
                cache_dic=cache_dic, tokens=txt, current=current
            )
            fresh_tokens_txt = self.txt_mlp(
                (1 + txt_mod2.scale) * self.txt_norm2(fresh_tokens_txt) + txt_mod2.shift
            )
            update_cache(
                fresh_indices=fresh_indices,
                fresh_tokens=fresh_tokens_txt,
                cache_dic=cache_dic,
                current=current,
            )
            txt = (
                txt
                + txt_mod2.gate
                * cache_dic["cache"][-1]["double_stream"][current["layer"]]["txt_mlp"]
            )

            if txt.dtype == torch.float16:
                txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        elif current["type"] == "FORA":
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
            img = (
                img
                + img_mod2.gate
                * cache_dic["cache"][-1]["double_stream"][current["layer"]]["img_mlp"]
            )
            txt = (
                txt
                + txt_mod2.gate
                * cache_dic["cache"][-1]["double_stream"][current["layer"]]["txt_mlp"]
            )

            if txt.dtype == torch.float16:
                txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        elif current["type"] == "aggressive":
            current["module"] = "skipped"
        else:
            raise ValueError("Unknown cache type.")

    return img, txt


def toca_SingleStreamBlock_forward(
    self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, **kwargs
) -> Tensor:
    cache_dic = kwargs.get("cache_dic", None)
    current = kwargs.get("current", None)

    mod, _ = self.modulation(vec)

    if cache_dic is None:
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

    else:
        current["stream"] = "single_stream"

        if current["type"] == "full":
            x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(
                self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
            )

            cache_dic["cache"][-1]["single_stream"][current["layer"]]["mlp"] = mlp
            current["module"] = "attn"

            q, k, v = qkv.view(
                qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)

            if cache_dic["cache_type"] == "k-norm":
                cache_dic["k-norm"][-1][current["stream"]][current["layer"]][
                    "total"
                ] = k.norm(dim=-1, p=2).mean(dim=1)
            elif cache_dic["cache_type"] == "v-norm":
                cache_dic["v-norm"][-1][current["stream"]][current["layer"]][
                    "total"
                ] = v.norm(dim=-1, p=2).mean(dim=1)

            q, k = self.norm(q, k, v)

            # compute attention
            attn = attention(
                q, k, v, pe=pe, mask=attn_mask, cache_dic=cache_dic, current=current
            )

            force_init(cache_dic=cache_dic, current=current, tokens=attn)
            cache_dic["cache"][-1]["single_stream"][current["layer"]]["attn"] = attn
            current["module"] = "mlp"

            # compute activation in mlp stream, cat again and run second linear layer
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

            force_init(cache_dic=cache_dic, current=current, tokens=output)
            current["module"] = "total"
            cache_dic["cache"][-1]["single_stream"][current["layer"]]["total"] = output

        elif current["type"] == "ToCa":
            current["module"] = "mlp"
            fresh_indices, fresh_tokens_mlp = cache_cutfresh(
                cache_dic=cache_dic, tokens=x, current=current
            )
            x_mod = (1 + mod.scale) * self.pre_norm(fresh_tokens_mlp) + mod.shift

            mlp_fresh = F.linear(
                x_mod,
                self.linear1.weight[self.hidden_size * 3 :, :].to(x_mod.dtype),
                (
                    self.linear1.bias[self.hidden_size * 3 :].to(x_mod.dtype)
                    if self.linear1.bias is not None
                    else None
                ),
            )
            update_cache(
                fresh_indices=fresh_indices,
                fresh_tokens=mlp_fresh,
                cache_dic=cache_dic,
                current=current,
            )

            # compute attention
            fake_fresh_attn = torch.gather(
                input=cache_dic["cache"][-1]["single_stream"][current["layer"]]["attn"],
                dim=1,
                index=fresh_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    cache_dic["cache"][-1]["single_stream"][current["layer"]][
                        "attn"
                    ].shape[-1],
                ),
            )

            # compute activation in mlp stream, cat again and run second linear layer
            current["module"] = "total"
            fresh_tokens_output = self.linear2(
                torch.cat((fake_fresh_attn, self.mlp_act(mlp_fresh)), 2)
            )
            update_cache(
                fresh_indices=fresh_indices,
                fresh_tokens=fresh_tokens_output,
                cache_dic=cache_dic,
                current=current,
            )
            output = cache_dic["cache"][-1]["single_stream"][current["layer"]]["total"]

        elif current["type"] == "FORA":
            output = cache_dic["cache"][-1]["single_stream"][current["layer"]]["total"]

        elif current["type"] == "aggressive":
            current["module"] = "skipped"
            if current["layer"] == 37:
                x = cache_dic["cache"][-1]["aggressive_output"]
            return x
        else:
            raise ValueError("Unknown cache type.")

        if current["layer"] == 37:
            cache_dic["cache"][-1]["aggressive_output"] = x

    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x
