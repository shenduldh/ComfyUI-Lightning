import torch
from torch import Tensor
from comfy.ldm.flux.layers import timestep_embedding


def has_affordable_memory(device: torch.device) -> bool:
    free_memory, _ = torch.cuda.mem_get_info(device)
    free_memory_gb = free_memory / (1024**3)
    return free_memory_gb > 24


def is_newer_than_ada_lovelace(device: torch.device) -> int:
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    return cc_major * 10 + cc_minor >= 89


def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result


def skip_forward_orig(
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
    ds_skip_blocks = transformer_options.get("ds_skip_blocks", [])
    ss_skip_blocks = transformer_options.get("ss_skip_blocks", [])
    
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
        #### skip blocks
        if i in ds_skip_blocks:
            continue

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
            #### skip blocks
            if i in ss_skip_blocks:
                continue

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


def teacache_skip_forward_orig(
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
    rel_l1_thresh = transformer_options.get("rel_l1_thresh", 0.4)
    ds_skip_blocks = transformer_options.get("ds_skip_blocks", [])
    ss_skip_blocks = transformer_options.get("ss_skip_blocks", [])

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

    ### teacache
    inp = img.clone()
    vec_ = vec.clone()
    img_mod1, _ = self.double_blocks[0].img_mod(vec_)
    modulated_inp = self.double_blocks[0].img_norm1(inp)
    modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift
    ca_idx = 0

    if not hasattr(self, "accumulated_rel_l1_distance"):
        should_calc = True
        self.accumulated_rel_l1_distance = 0
    else:
        try:
            coefficients = [
                4.98651651e02,
                -2.83781631e02,
                5.58554382e01,
                -3.82021401e00,
                2.64230861e-01,
            ]
            self.accumulated_rel_l1_distance += poly1d(
                coefficients,
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                ),
            )
            if self.accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        except:
            should_calc = True
            self.accumulated_rel_l1_distance = 0

    self.previous_modulated_input = modulated_inp

    if not should_calc:
        img += self.previous_residual
    else:
        orig_img = img.clone()
        for i, block in enumerate(self.double_blocks):
            ### skip blocks
            if i in ds_skip_blocks:
                continue

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
                    {
                        "img": img,
                        "txt": txt,
                        "vec": vec,
                        "pe": pe,
                        "attn_mask": attn_mask,
                    },
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

            ### PuLID attention
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    ### calculate influence of all pulid nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any(
                            (node_data["sigma_start"] >= timesteps)
                            & (timesteps >= node_data["sigma_end"])
                        ):
                            img = img + node_data["weight"] * self.pulid_ca[ca_idx](
                                node_data["embedding"], img
                            )
                    ca_idx += 1

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                ### skip blocks
                if i in ss_skip_blocks:
                    continue

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

            ### PuLID attention
            if getattr(self, "pulid_data", {}):
                real_img, txt = img[:, txt.shape[1] :, ...], img[:, : txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    ### calculate influence of all nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any(
                            (node_data["sigma_start"] >= timesteps)
                            & (timesteps >= node_data["sigma_end"])
                        ):
                            real_img = real_img + node_data["weight"] * self.pulid_ca[
                                ca_idx
                            ](node_data["embedding"], real_img)
                    ca_idx += 1
                img = torch.cat((txt, real_img), 1)

        img = img[:, txt.shape[1] :, ...]
        self.previous_residual = img - orig_img

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img
