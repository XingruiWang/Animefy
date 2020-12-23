import os

import numpy as np
import torch

import stylegan2
from stylegan2 import utils



def synthesis(G_file, latent_file):
    device = torch.device('cpu')
    G = stylegan2.models.load(G_file).G_synthesis 
    latent = np.load(latent_file, allow_pickle=True)
    G.to(device)
    latent = torch.tensor(latent[np.newaxis, ...]).to(device)

    out = G(latent)

    out = utils.tensor_to_PIL(out, pixel_min=-1, pixel_max=1)[0]
    return out

if __name__ == '__main__':

    # out = synthesis('checkpoints/stylegan2_512x512_with_pretrain_new_2/10000_2020-12-22_03-42-54/Gs.pth', 'projects/latent/image0000-step1000.npy')
    out = synthesis('G_out.pth', 'projects/latent/image0000-step0011.npy')
    # out = synthesis('checkpoints/stylegan2_512x512_with_pretrain/pretrain/Gs.pth', '/home/wxr/stylegan2_pytorch_backup/projects/latent/zj.npy')
    # G = stylegan2.models.load('checkpoints/stylegan2_512x512_with_pretrain/pretrain/Gs.pth').G_synthesis # 
    out.save('out.png')
    # for s in ['0041','0081','0121','0161','0200']:
    #     out = synthesis('G_out.pth', 'projects/latent/image0000-step%s.npy'%(s))
    #     out.save('out%s.png'%(s))

