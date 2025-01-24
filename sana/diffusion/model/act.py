from typing import Optional
import copy

import torch.nn as nn

__all__ = ["build_act", "get_act_name"]

# register activation function here
#   name: module, kwargs with default values
REGISTERED_ACT_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "relu": (nn.ReLU, {"inplace": True}),
    "relu6": (nn.ReLU6, {"inplace": True}),
    "hswish": (nn.Hardswish, {"inplace": True}),
    "hsigmoid": (nn.Hardsigmoid, {"inplace": True}),
    "swish": (nn.SiLU, {"inplace": True}),
    "silu": (nn.SiLU, {"inplace": True}),
    "tanh": (nn.Tanh, {}),
    "sigmoid": (nn.Sigmoid, {}),
    "gelu": (nn.GELU, {"approximate": "tanh"}),
    "mish": (nn.Mish, {"inplace": True}),
    "identity": (nn.Identity, {}),
}


def build_act(name: Optional[str], **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls, default_args = copy.deepcopy(REGISTERED_ACT_DICT[name])
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return act_cls(**default_args)
    elif name is None or name.lower() == "none":
        return None
    else:
        raise ValueError(f"do not support: {name}")


def get_act_name(act: Optional[nn.Module]) -> Optional[str]:
    if act is None:
        return None
    module2name = {}
    for key, config in REGISTERED_ACT_DICT.items():
        module2name[config[0].__name__] = key
    return module2name.get(type(act).__name__, "unknown")
