# ⚡ComfyUI-Lightning

## Introduction

This repository integrates all the tricks I know to speed up Flux inference:

1. Use `TeaCache` or `FBCache` or `MBCache`;
2. Skip some unnessasery blocks;
3. Compile and quantize model;
4. Use fast CuDNN attention kernels;
5. Use SageAttention;
6. Fix `AttributeError: 'SymInt' object has no attribute 'size'` to speed up recompilation after resolution changing.

`MBCache` extends `FBCache` and is applied to cache multiple blocks. The codes are modified from [SageAttention](https://github.com/thu-ml/SageAttention), [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache), [comfyui-flux-accelerator](https://github.com/discus0434/comfyui-flux-accelerator) and [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed). More details see above given repositories.

## Updates

- \[2025/1/24\] Now support [Sana](https://github.com/NVlabs/Sana). Get your 1024*1024 images within 2s. All the codes are modified from [Sana](https://github.com/NVlabs/Sana).

## Usage

### For Flux

You can use `XXCache`, `SageAttention`, and `torch.compile` with the following examples:

<img src="./assets/FBCache.png" alt="FBCache" width="80%"/>
<img src="./assets/TeaCache.png" alt="TeaCache" width="80%"/>
<img src="./assets/MBCache.png" alt="MBCache" width="80%"/>

More specific:

1. Download Flux diffusion model and VAE image decoder from [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) or [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell). Put the `flux1-dev.safetensors` or `flux1-schnell.safetensors` file into `models/diffusion_models` and the `ae.safetensors` file into `models/vae`;

3. Download Flux text encoder from [flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) and put all the `.safetensors` files into `models/clip`;

4. Run the example [workflow](./examples/flux_example_workflow.json).

### For Sana

<img src="./assets/sana_generation_results.png" alt="Sana Generation Results" width="80%"/>

1. Download Sana diffusion model from [Model Zoo](https://github.com/NVlabs/Sana/blob/main/asset/docs/model_zoo.md) and put the `.pth` file into `models/diffusion_models`;

2. Download Gemma text encoder from [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it), [unsloth/gemma-2b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-2b-it-bnb-4bit) or [Efficient-Large-Model/gemma-2-2b-it](https://huggingface.co/Efficient-Large-Model/gemma-2-2b-it) and put the whole folder into `models/text_encoders`;

3. Download DCAE image decoder from [mit-han-lab/dc-ae-f32c32-sana-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0) and put the `.safetensors` file into `models/vae`;

4. Run the example [workflow](./examples/sana_example_workflow.json).
