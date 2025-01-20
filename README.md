# ComfyUI-Lightning

The codes are modified from [SageAttention](https://github.com/thu-ml/SageAttention), [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache), [comfyui-flux-accelerator](https://github.com/discus0434/comfyui-flux-accelerator) and [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed).

I use follow tricks to accelerate FLUX inference speed:

1. Add `TeaCache` or `FBCache` or `MBCache`;
2. Skip some unnessasery blocks;
3. Compile and quantize model;
4. Use fast CuDNN attention kernels;
5. Use SageAttention;
6. Fix `AttributeError: 'SymInt' object has no attribute 'size'` to speed up recompilation after a resolution change.

`MBCache` extends `FBCache` and is applied to cache multiple blocks. More details see above given repositories.

This repository integrates all the tricks I know to speed up Flux inference. If you know more, please comment and I will try to complete.

## Usage

![fbcache_and_skip_blocks](./assets/FBCache.png)

![teacache_and_skip_blocks](./assets/TeaCache.png)

![teacache_and_skip_blocks](./assets/MBCache.png)
