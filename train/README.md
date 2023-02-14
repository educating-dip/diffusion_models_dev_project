Training a score-based diffusion model: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=XCR6m0HjWGVV

TODO:
1. The Unet currently implement in train.model is a rather small UNet (only 3 scales...). For the 512x512 walnut images we thus would get 512->256->128. I think we need at least 2 more scales. (The current ellipses are 128x128)
2. The SDE in train.sde is dx = \sigma^t dw, is this optimal? 
3. Add Exponential Moving Average (m=0.999 from Yang Song) (there is this library: https://github.com/lucidrains/ema-pytorch). But I use the implementation from Song's Github https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/ema.py#L10 (which is essentially modified from ema-torch) - Done, but not tested (maybe only start EMA after 100-200 steps? We dont want the initial random weights in the average...)
4. Currently we sample the time step t~U[0,1] uniform in [0,1]. Maybe it would help to "cheat" and sample more time steps near zero? So that the last few denoising steps are really good? 