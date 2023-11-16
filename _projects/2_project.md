---
layout: page
title: AIMANet- Detecting Heart Rate from Facial Videos
description: Fully convolutional neural network to detect heart rate
img: assets/img/aimanet/model_branch.jpg
importance: 2
category: computer vision
giscus_comments: true
---

## 1. Overview

This project is a fully convolutional neural network to detect heart rate from facial videos. This is a state-of-the-art model reimplemented in PyTorch and trained and evaluated on the UBFC-Phys dataset. The model is also trained and evaluated on data with/without background to examine the impact on the model's performance.

The heart pumps blood throughout the body causing changes in the amount of blood under the skin. This affects the amount of light absorbed and reflected by the skin. This is used to estimate the Blood Volume Pulse (BVP) signal, which can be used to infer the heart rate.
Previous research has focused on using photoplethysmography (PPG) signals to measure blood volume changes in the skin using light.
Remote photoplethysmography (rPPG) is a method for measuring PPG signals from a distance using a video camera, allowing for PPG measurements without direct skin contact.

## 2. Dataset description

The UBFC-Phys dataset was collected from 56 participants. This includes video recordings of participants undergoing three tasks along with their BVP signal. The tasks are a rest task (T1), a speech task (T2), and an arithmetic task (T3).

The participants completed the three-step experience. We used a subset of the dataset containing data from 26 participants, containing videos with a duration of over 230 minutes. The videos were recorded at 35 frames per second (fps) and the BVP signal was recorded at 64Hz.

<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/aimanet/dataset_description.png" title="dataset_prep" class="img-fluid rounded z-depth-1" %}
    </div>

### 3. Model Architecture

The primary objective of our architecture is to learn the mapping to translate the spatial information in a RGB image to latent encoding corresponding to the pulse signals. The architecture is also required to learn features that account for noise factors like head movement and changes in lighting. This is achieved by incorporating a temporal shift convolution attention mechanism.

For this purpose, the architecture uses two branches to learn facial and motion features with a spatial attention module. The first branch is used to learn the temporal motion information while the second branch learns the spatial facial information. One of the ways to learn the temporal motion information is with 3D convolutions. But, 3D convolution introduces high computation cost which makes it infeasible for real time computation and inference.

To this end, the architecture leverages Time Shift Attention based Convolution (TS-CAN) to remove the need for 3D convolution operations while still allowing for spatial-temporal modeling. TS-CAN has two main components: the temporal shift module (TSM) and an attention module. TSM splits the input data into three chunks and shifts the first and second chunks along the temporal axis, while the third chunk remains unchanged. This allows information to be exchanged among neighboring frames without adding any additional parameters to the network. The architecture uses TSM in the motion branch. The appearance branch takes in the mean applied input frame and applies Attention.

<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/aimanet/model_branch.jpg" title="dataset_prep" class="img-fluid rounded z-depth-1" %}
    </div>

### 4. Model Hyperparameters

- Optimizer - Adadelta optimizer
- Loss function - MSE Loss
- Epochs - 8

### 5. Results

The 2 experiments yielded interesting results where the Mean Squared Error (MSE) is almost the same for both the cases. This could be due to the controlled nature of the dataset, where everyone was recorded with the same static background under equal lighting conditions and distance. Furthermore, running more experiments on the complete dataset and for more epochs could yield more precise results.

| MSE (with background) | MSE (without background) |
| :-------------------: | :----------------------: |
|        0.3457         |          0.3455          |
