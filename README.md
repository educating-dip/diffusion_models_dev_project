# [Steerable Conditional Diffusion for Out-of-Distribution Adaptation in Imaging Inverse Problems](https://arxiv.org/abs/2308.14409)

by [Riccardo Barbano](https://scholar.google.com/citations?user=6jYGiC0AAAAJ&hl=en)\*, [Alexander Denker](https://www.uni-bremen.de/techmath/team/doktorandinnen/alexander-denker)\*, Hyungjin Chung\*, Tae Hoon Roh, Simon Arrdige, Peter Maass, Bangti Jin, Jong Chul Ye. 

This repository contains the experiments for the Computed Tomography experiments in the _main_ branch.

TODO:
1. remove the OpenAI arch. and only leave the dds-U-Net one (@rb876)
2. remove the two channel division (this will allow only to load the ellipses arch) (@adenker)
3. re-write the ellipses config for ddpm (@adenker)
4. simplify CG type and configs (@rb876)
5. remove the corrector and the come-closer-diffuse-faster approach (@rb876)
6. remeove ellipses dataset and lodopab datasets that are deprecated (@adenker)
7. remove the hard coded path (@adenker)
8. remove the Mayo dataset 
8. adjust the ellipses and wrap it into a SimulatedDataset class