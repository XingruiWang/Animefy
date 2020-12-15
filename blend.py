import cv2 as cv
import torch
import stylegan2

def extract_conv_names(model):
    model = list(model.keys())
    conv_name = []
    resolutions = [4*2**x for x in range(9)]
    level_names =  [["Conv0_up", "Const"],["Conv1", "ToRGB"]]

def blend_models(model_1, model_2, resolution, level, blend_with=None):
    resolution = f"{resolution}x{resolution}"
    G_1 = stylegan2.models.load(model_1)
    G_2 = stylegan2.models.load(model_2)
    model_1_state_dict = G_1.state_dict()
    model_2_state_dict = G_2.state_dict()
    assert(model_1_state_dict.keys()==model_2_state_dict.keys())
    # model_1_conv = extract_conv_names(model_1_state_dict)
    print(G_1.to_data_layers)




def main():
    blend_models("checkpoints/stylegan2_512x512_with_pretrain/1000_2020-12-14_02-30-35/G.pth", "checkpoints/stylegan2_512x512_with_pretrain/pretrain/G.pth", 64, None)
if __name__ == '__main__':
    main()