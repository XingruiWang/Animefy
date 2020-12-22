import os

import numpy as np
import torch

import stylegan2
from stylegan2 import utils



def synthesis(G_file, latent_file):
    device = torch.device(0)
    G = stylegan2.models.load(G_file).G_synthesis 
    latent = np.load(latent_file)
    G.to(device)
    latent = torch.tensor(latent[np.newaxis, ...]).to(device)

    out = G(latent)

    out = utils.tensor_to_PIL(out, pixel_min=-1, pixel_max=1)[0]
    return out

if __name__ == '__main__':

    # out = synthesis('checkpoints/stylegan2_512x512_with_pretrain_new_2/10000_2020-12-22_03-42-54/Gs.pth', 'projects/latent/image0000-step1000.npy')
    out = synthesis('G_out.pth', 'projects/latent/image0000-step1000.npy')
    # G = stylegan2.models.load('checkpoints/stylegan2_512x512_with_pretrain/pretrain/Gs.pth').G_synthesis # 
    out.save('out.png')
