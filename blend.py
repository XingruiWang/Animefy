import cv2 as cv
import torch
import stylegan2


def extract_conv_names(model):
    model = list(model.keys())
    conv_name = []
    resolutions = [4*2**x for x in range(9)]
    level_names = [["Conv0_up", "Const"], ["Conv1", "ToRGB"]]


def blend_models(model_1, model_2, resolution, level, blend_width=None):
    resolutions = [4 * 2 ** i for i in range(7)]
    mid = resolutions.index(resolution)

    G_1 = stylegan2.models.load(model_1)
    G_2 = stylegan2.models.load(model_2)
    model_1_state_dict = G_1.state_dict()
    model_2_state_dict = G_2.state_dict()
    assert(model_1_state_dict.keys() == model_2_state_dict.keys())
    G_out = G_1.clone()

    layers = []
    ys = []
    for k, v in model_1_state_dict.items():
        if k.startswith('G_synthesis.conv_blocks.'):
            pos = int(k[len('G_synthesis.conv_blocks.')])
            x = pos - mid
            if blend_width:
                exponent = -x / blend_width
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1 if x > 0 else 0

            layers.append(k)
            ys.append(y)
        elif k.startswith('G_synthesis.to_data_layers.'):
            pos = int(k[len('G_synthesis.to_data_layers.')])
            x = pos - mid
            if blend_width:
                exponent = -x / blend_width
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1 if x > 0 else 0
            layers.append(k)
            ys.append(y)
    out_state = G_out.state_dict()
    for y, layer in zip(ys, layers):
        out_state[layer] = y * model_2_state_dict[layer] + \
            (1 - y) * model_1_state_dict[layer]
    G_out.load_state_dict(out_state)
    return G_out


def blend_models_2(model_1, model_2, resolution, level, blend_width=None):
    # resolution = f"{resolution}x{resolution}"
    resolutions = [4 * 2 ** i for i in range(7)]
    mid = [resolutions.index(r) for r in resolution]

    G_1 = stylegan2.models.load(model_1)
    G_2 = stylegan2.models.load(model_2)
    model_1_state_dict = G_1.state_dict()
    model_2_state_dict = G_2.state_dict()
    assert(model_1_state_dict.keys() == model_2_state_dict.keys())
    G_out = G_1.clone()

    layers = []
    ys = []
    for k, v in model_1_state_dict.items():
        if k.startswith('G_synthesis.conv_blocks.'):
            pos = int(k[len('G_synthesis.conv_blocks.')])
            y = 0 if pos in mid else 1
            layers.append(k)
            ys.append(y)
        elif k.startswith('G_synthesis.to_data_layers.'):
            pos = int(k[len('G_synthesis.to_data_layers.')])
            y = 0 if pos in mid else 1
            layers.append(k)
            ys.append(y)
    # print(ys, layers)
    out_state = G_out.state_dict()
    for y, layer in zip(ys, layers):
        out_state[layer] = y * model_2_state_dict[layer] + \
            (1 - y) * model_1_state_dict[layer]
    G_out.load_state_dict(out_state)
    return G_out


def main():
    G_out = blend_models("checkpoints/stylegan2_512x512_with_pretrain/pretrain/Gs.pth",
                         "checkpoints/stylegan2_512x512_with_pretrain_new_2/20000_2020-12-23_13-17-51/Gs.pth",
                         8,
                         None)
    G_out.save('G_blend.pth')


if __name__ == '__main__':
    main()
