# Animefy yourself!

🙃 I realize that maybe the model needs to be trained for longer time, I will update this when I get time later ...

<img src="https://raw.githubusercontent.com/XingruiWang/Animefy/master/example/example.png" alt="example" style="width: 100%;" />

### About

A "selfie2anime" project based on StyleGAN2. You can generate your own animate faces base on real-world selfie. The whole "selfie2anime" project is based on StyleGAN2\[[Official code](https://github.com/NVlabs/stylegan2)\]\[[paper](https://arxiv.org/abs/1912.04958)\]\[[video](https://youtu.be/c-NJtV9Jvp0)\]and layer swapping technique proposed by [Justin Pinkney](https://www.justinpinkney.com/). And also thanks for the highly reproduceable Pytorch reimplementing styleGAN2 project by [Tetratrio](https://github.com/Tetratrio/stylegan2_pytorch) .

```
Animefy
│  README.md -------------------------------- Description of the projects.
│  latent.sh -------------------------------- The script to find latent feature of a given image.
│  run.sh ----------------------------------- The script to train the model.
│  synthesis.sh ----------------------------- The script to generate animate image without condition.
│  align_images.py -------------------------- Align the face of given images, since the given selfie image might not in the same scale.
│  blend.py --------------------------------- Blend the model after well trained.
│  generate.py ------------------------------ Generate animate image based on the latent code.
│  run_convert_from_tf.py ------------------- Convert pretrained model file in tensorflow to pytorch.
│  run_generator.py ------------------------- Generate animte images without condition (i.e. latent code), called by `synthesis.sh`.
│  run_metrics.py --------------------------- Caculate the metric of trained model.
│  run_projector.py ------------------------- Find latent feature of a given image, called by `latent.sh`.
│  run_training.py -------------------------- Train the model, called by `run.sh`.
│  requirements.txt ------------------------- Environment required, can be set up using `pip install -r requirements.txt`.
│  settings_with_pretrain.yaml -------------- Configuration when training the model.
│
├─environment
│      anime.yaml
│
├─example
│      example.png ------------------------- Example image of the process of finding latent code and generate corresponding anime image by interating.
│
├─Notebook
│      Animefy-yourself.ipynb -------------- Description notebook of the project.
│
├─projects
│  │  resize.py ---------------------------- Might be useless now, resize the selfie image to 512 × 512.
│  │
│  └─latent
│          image0000-target.png
│
└─stylegan2 -------------------------------- Core files.
    │  loss_fns.py ------------------------- Loss function.
    │  models.py --------------------------- StyleGAN2 model file
    │  modules.py -------------------------- Dependence module of StyleGAN2.
    │  project.py -------------------------- Find the latent code.
    │  train.py ---------------------------- Define class `Trainer` to train the model.
    │  utils.py ---------------------------- Utility file of model and training.
    │  __init__.py ------------------------- Using for importing.
    │
    ├─external_models
    │      inception.py -------------------- Inception module.
    │      lpips.py ------------------------ Caculate the similarity between images.
    │      __init__.py
    │
    └─metrics ------------------------------ Metric file.
            fid.py ------------------------- Fréchet Inception Distance。
            ppl.py ------------------------- Perplexity.
            __init__.py -------------------- Using for importing.

```

### Dateset

The animate faces datast we used is [here](http://www.seeprettyface.com/mydataset_page3.html#anime)，this dataset is processed from [DANBOORU2018](https://www.gwern.net/Danbooru2020#danbooru2018). The dataset contains 140000 animate faces.（[Baidu Drive](https://pan.baidu.com/share/init?surl=8pHjzcOWhVF2u6LKOlT3yg)(code：JIMD)）

### Installation

- clone the repository by running

```{shell}
git clone https://github.com/XingruiWang/Animefy.git 
```

- dependencies

  We refer to the dependencies of [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), all dependencies are provided in `environment/anime.yaml`

### Train model

- If you want to reproduce the project,  run `run.sh`

```shell
sh run.sh
```

- Remember to change configure file `settings_with_pretrain.yaml`.Set your data path in `data_dir`. Set the checkpoint in `g_file` and `d_file`. 

- We trained our model on 4 GPUs and obtained an acceptable result after 2000 iterations (3-4 hours). Our final model was trained for 18000 iterations (around 2 days).
- If you want to transfer the project to other dataset, like project human faces to cat faces, we recommend you to keep the pretrain model as StyleGAN2 FFHQ pretrain model and finetuning on your custom dataset (by change `data_dir`).
- Since our proposal is to project human face to anime character, we **freeze** the `G_mapping` layer after loading pretrain model, to keep the ability of model of extracting the human faces features. If you just want to generate high quality animate image, you don't need to do that (set `param.requires_grad = True` in line 767 of `run_training.py`).

### Model blending

- The trained model is not enough to generate custom animate faces. Inspired by  [Justin Pinkney](https://www.justinpinkney.com/), we blended the human faces generating model and our animate faces generating model in order to keep the low resolution information of real world human face (gesture, head position and angle) and high resolution information (big eyes, small nose, hair style and etc. ) of animate faces. 

- I reimplement the [blend_models.py](https://github.com/justinpinkney/stylegan2/blob/master/blend_models.py) by pytorch ([here](https://github.com/XingruiWang/Animefy/blob/master/blend.py))

```python
from blend import blend_models

G_out = blend_models("G_pretrain.pth", 
                     "G_out.pth", 
                     8, None)
G_out.save('G_blend.pth')
```

### Pretrained model

| Name     |  Description    |
| ---- | ---- |
|  [StyleGAN2 on FFHQ dataset (config-f 512x512)](https://github.com/justinpinkney/awesome-pretrained-stylegan2#faces-FFHQ-config-f-512x512)    | The pretrain model trained on FFHQ human dataset to generate human faces. |
| [StyleGAN2 on animate characters](https://drive.google.com/file/d/1J6sJaRZJg4dAoSw03fyanWV2oEscOeSk/view?usp=sharing) | Our final result to generate animate faces and also to project real-world human faces, namely `G_out.pth`. |
| [StyleGAN2 on animate characters(blended)](https://drive.google.com/file/d/1kzjSchNGZ1b9Q_eX0fCZ1ACd2NgS-SS6/view?usp=sharing) | Our final result to generate animate faces and also to project real-world human faces, namely `G_blend.pth`. |


### Custom your own Animate Character

:point_right: **You can generate your own character on the colab Notebook [here](https://colab.research.google.com/github/XingruiWang/Animefy/blob/master/Notebook/Animefy-yourself-new.ipynb) (Recommanded)**

Or running in **terminal**: 

- Download the pretrain model `G_blend.pth`

- put your own pictures in `./projects/real` （must be 512x512）

- project your faces to latent layer

```bash
sh latent.sh
```

- latent feature have been stored in `./projects/latent`

- run `python generate.py`


