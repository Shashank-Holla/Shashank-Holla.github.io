---
layout: page
title: Stable Diffusion with Self-Guided Attention and ControlNet
description: Self-guided attention and ControlNet guidance capabilities on pre-trained Stable Diffusion.
img: assets/img/MR_p9.png
importance: 1
category: computer vision
# related_publications: einstein1956investigations, einstein1950meaning
---

## 1. Overview

This is Stable Diffusion built on pre-trained Stable Diffusion v1.5 weights with Self-Attention Guidelines (SAG) to enhance generated image's stability. It also uses ControlNet, a neural network model, to support additional input to control the image generation. Additionally, the model can add artistic features to the generated image by utilizing trained style weights.

This model is built on Hugging Face modules. It utilizes Tokenizer, Text Encoder, Variational Auto Encoder and Unet model from it.

1. Tokenizer - creates tokens with padding to match required length.
2. Text Encoder - Get token embedding from tokens and the positional embedding. It is then combined and fed to a transformer model to get the output embedding
3. UNet - Takes in noisy latents and predicts the noise residual of the latent shape.
4. Variational Autoencoder - Takes in the latents and decodes it into the image space.

## 2. Features

### 2.1. Self Attention Guidelines

Self attention guidelines helps stable diffusion to improve generated image. It uses the intermediate self-attention maps to adversially blur and guides the model. Parameter `sag_scale` controls the SAG influence on the model.

### 2.2. ControlNet support

ControlNet conditions the diffusion model to learn specific user input conditions (like edges, depth). This helps it generate images which are related to the desired spatial context. `canny` and `openpose` controlnets are supported in this application. Conditional input image such as edge map, keypoints are also provided along with the controlnet model for inference.
`controlnet_cond_scale` parameter controls the scale to which the generated image are faithful to the conditional image.

### 2.3. Style

The application is trained on a novel art via Textual Inversion. In our case, images stylistically related to pop-art are trained in order to associate it with `<pop-art>` word within the text encoder embedding. Training images and the weights for style training are available here [\<pop-art\>](https://huggingface.co/sd-concepts-library/pop-art).

To use the style, add <pop-art> in the prompt. While running the model, enable `style_flag` to use the style.

<div class="row">
<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/stablediffusion/popart-3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/stablediffusion/popart-4.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/stablediffusion/popart-5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/stablediffusion/popart-7.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Above are some of the style images on which textual inversion is trained on.
</div>

## 3. Deploy and Run

Stable Diffusion can be run in the following two ways-

### 3.1. Clone Repository and execute

Clone repository and change directory-

```
git clone https://github.com/Shashank-Holla/diffusion-controlnet-sag.git

cd diffusion-controlnet-sag/
```

Install dependencies-

```
pip install -r requirements.txt
```

Run model

```
!python main.py --prompt "Margot Robbie as wonderwoman in style" --seed 3 --batch_size 1 --controlNet_image ./control_images/controlimage_1.jpg --controlNet_type canny --style_flag T --sag_scale 0.75 --controlnet_cond_scale 1.0
```

### 3.2. Install CLI application and run

This repository is also available as CLI application. Build files are available in `dist` folder in this repository. Control Image and style weights path must be absolute. Valid Control Image is required if controlnet model is provided.

Clone repository and change directory-

```
git clone https://github.com/Shashank-Holla/diffusion-controlnet-sag.git

cd diffusion-controlnet-sag/
```

Install distribution-

```
!pip install dist/diffusion-0.0.7-py3-none-any.whl
```

Run application `generate`. Provide input as prompted-

```
/usr/local/bin/generate
```

## 4. Results

Shared here are few run results by changing the various parameters.

### 4.1. By changing SAG scale and adding artistic style

These run results are by varying SAG scale and adding artistic style.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.html path="assets/img/6.jpg" title="example image"
    class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.html path="assets/img/11.jpg" title="example image"
    class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %}
