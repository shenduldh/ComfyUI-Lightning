import torch
from torch import Tensor, nn
from comfy.ldm.flux.math import apply_rope

from spas_sage_attn.autotune import (
    SparseAttentionMeansim,
    extract_sparse_attention_state_dict,
)


def load_sparse_attention_state_dict(model, saved_state_dict, verbose=False):
    device = next(model.parameters()).device

    for k, v in model.named_modules():
        if isinstance(
            v, SparseAttentionMeansim
        ):  # find each SparseAttentionMeansim instance
            if verbose:
                print(
                    k, "is an instance of SparseAttentionMeansim, but it is empty now."
                )
            for sk, sv in saved_state_dict.items():
                if k in sk:
                    if verbose:
                        print(f"{sk} is a substate_dict of {k}, we will load it.")

                    sub_name = sk.split(k)[1][1:]
                    sv = sv.to(device=device)
                    setattr(v, sub_name, nn.Parameter(sv, requires_grad=False))
    return model


def attention_sparge(
    spargeattn: SparseAttentionMeansim,
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head),
            (q, k, v),
        )
        tensor_layout = "NHD"

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = spargeattn(
        q,
        k,
        v,
        mask=mask,
        is_causal=False,
        tune_mode=spargeattn.enable_tuning_mode,
        return_sparsity=False,
    )

    if tensor_layout == "HND":
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        out = out.reshape(b, -1, heads * dim_head)
    return out


def attention(
    spargeattn: SparseAttentionMeansim,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pe: Tensor,
    mask=None,
) -> Tensor:
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = attention_sparge(spargeattn, q, k, v, heads, skip_reshape=True, mask=mask)
    return x


def spargeattn_DoubleStreamBlock_forward(
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
        attn = attention(
            self.spargeattn,
            torch.cat((img_q, txt_q), dim=2),
            torch.cat((img_k, txt_k), dim=2),
            torch.cat((img_v, txt_v), dim=2),
            pe=pe,
            mask=attn_mask,
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
    else:
        # run actual attention
        attn = attention(
            self.spargeattn,
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe,
            mask=attn_mask,
        )

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    # calculate the img bloks
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
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


def spargeattn_SingleStreamBlock_forward(
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
    attn = attention(self.spargeattn, q, k, v, pe=pe, mask=attn_mask)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x
