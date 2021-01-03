# Generate Your Own Animate Character

<img src="https://raw.githubusercontent.com/XingruiWang/GenerateYourOwnAnimateCharacter/master/example/example.png" alt="example" style="width: 100%;" />

### About

The whole project is based on StyleGAN2\[([Official code](https://github.com/NVlabs/stylegan2)\]\[[paper](https://arxiv.org/abs/1912.04958)\]\[[video](https://youtu.be/c-NJtV9Jvp0)\]and layer swapping technique proposed by [Justin Pinkney](https://www.justinpinkney.com/). And also thanks for the highly reproduceable Pytorch reimplementing styleGAN2 project by [Tetratrio](https://github.com/Tetratrio/stylegan2_pytorch) .

### Installation

- clone the repository by running

```{shell}
git clone https://github.com/XingruiWang/GenerateYourOwnAnimateCharacter.git
```

- dependencies

  We refer to the dependencies of [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), all dependencies are provided in `environment/anime.yaml`

### Notebook

We are working on preparing a Colab notebook to present our result.

### Train model

- If you want to reproduce the project,  run `run.sh`

```shell
sh run.sh
```

- Remember to change configure file `settings_with_pretrain.yaml`.Set your data path in `data_dir`. Set the checkpoint in `g_file` and `d_file`. 

- We trained our model on 4 GPUs and obtained an acceptable result after 2000 iterations (3-4 hours). Our final model was trained for 18000 iterations (around 2 days).
- If you want to transfer the project to other dataset, like project human faces to cat faces, we recommend you to keep the pretrain model as StyleGAN2 FFHQ pretrain model and finetuning on your custom dataset (by change `data_dir`).
- Since our propose is to project human face to anime character, we freeze the `G_mapping` layer after loading pretrain model to keep the ability of model to extract the human faces features. If you just want to generate high quality image, you don't need to do that (set `param.requires_grad = True` in line 767 of `run_training.py`).

### Model blending

- The trained model is not enough to generate custom animate faces. Intrigued by  [Justin Pinkney](https://www.justinpinkney.com/), we blended the human faces generating model and our animate faces generating model in order to keep the low resolution information of real world human face (gesture, head position and angle) and high resolution information (big eyes, small nose, hair style and etc. ) of animate faces. 

### Pretrained model

| Name     |  Description    |
| ---- | ---- |
|  [StyleGAN2 on FFHQ dataset (config-f 512x512)](https://github.com/justinpinkney/awesome-pretrained-stylegan2#faces-FFHQ-config-f-512x512)    | The pretrain model trained on FFHQ human dataset to generate human faces. |
| [StyleGAN2 on animate characters]() | Our final result to generate animate faces and also to project real-world human faces, namely `G_out.pth`. |
| [StyleGAN2 on animate characters(blended)]() | Our final result to generate animate faces and also to project real-world human faces, namely `G_blend.pth`. |

### Custom your own Animate Character

- Download the pretrain model `G_blend.pth`

- put your own picture in `./projects/real`

- run `resize.py`

```
cd ./projects
python resize.py
cd ..
```

- project your faces to latent layer

```
sh latent.sh
```

- latent feature have been stored in `./projects/latent`
- run `python generate.py`

