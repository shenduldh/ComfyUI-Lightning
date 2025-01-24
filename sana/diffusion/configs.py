DIFFUSION_CONFIGS = {
    "SanaMS_1600M_P1_D20": {
        # "input_size": 32,
        # "class_dropout_prob": 0.1,
        # "drop_path": 0.0,
        # "caption_channels": 2304,
        # "pe_interpolation": 1.0,
        # "config": None,
        # "qk_norm": False,
        # "norm_eps": 1e-5,
        # "y_norm_scale_factor": 1.0,
        # "patch_embed_kernel": None,
        # "cross_norm": False,
        "patch_size": 1,
        "in_channels": 32,
        "hidden_size": 2240,
        "depth": 20,
        "num_heads": 20,
        "mlp_ratio": 2.5,
        "learn_sigma": False,
        "pred_sigma": False,
        "model_max_length": 300,
        "y_norm": True,
        "attn_type": "linear",
        "ffn_type": "glumbconv",
        "use_pe": False,
        "mlp_acts": ("silu", "silu", None),
        "linear_head_dim": 32,
        "fp32_attention": True,
    },
    "SanaMS_600M_P1_D28": {
        # "input_size": 32,
        # "class_dropout_prob": 0.1,
        # "drop_path": 0.0,
        # "caption_channels": 2304,
        # "pe_interpolation": 1.0,
        # "config": None,
        # "qk_norm": False,
        # "norm_eps": 1e-5,
        # "y_norm_scale_factor": 1.0,
        # "patch_embed_kernel": None,
        # "cross_norm": False,
        "patch_size": 1,
        "in_channels": 32,
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
        "mlp_ratio": 2.5,
        "learn_sigma": False,
        "pred_sigma": False,
        "model_max_length": 300,
        "y_norm": True,
        "attn_type": "linear",
        "ffn_type": "glumbconv",
        "use_pe": False,
        "mlp_acts": ("silu", "silu", None),
        "linear_head_dim": 32,
        "fp32_attention": True,
    },
}
