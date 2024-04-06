# Variational Autoencoder (VAE)

## Overview

This repository contains an implementation of a Variational Autoencoder (VAE), a type of generative model, 
using Python and PyTorch. The VAE is a probabilistic model that learns to encode and decode high-dimensional data, 
such as images, by mapping them to a latent space where the data's structure is preserved. 
This project is aimed at exploring the capabilities of VAEs for image generation and reconstruction tasks.
This Repo is still a work in progress, as time goes on I plan on modify the model further in oreder to produce better
and higher fidelity images.

## Features

Implementation of a Variational Autoencoder architecture.

Training pipeline for training the VAE on a dataset of images.

Generation of new images from latent space samples.

Reconstruction of input images from the latent space.

## Requirements

Python 3.x

PyTorch

NumPy

Matplotlib (for visualization)

## Usage

Clone the repository:

Copy code
```bash
git clone https://github.com/K3dA2/VAE.git
```

## Sample generated images.
Here are some sample Images generated by smapling the latent space of the current model
![e92c3285-9053-4dbf-961c-56cd9cb56943](https://github.com/K3dA2/VAE/assets/112480809/3cd4eaa1-cdba-40a5-9766-83d4e9ee4a9d)
![2d239f67-75f8-4cdb-a7ec-a2a5c05a57ea](https://github.com/K3dA2/VAE/assets/112480809/4ea3ab98-15a2-4b86-a7de-8081fdc6acd6)
![1f89c3d9-1ff9-4629-9af8-aba89d72d7b3](https://github.com/K3dA2/VAE/assets/112480809/cfe548d4-76df-4659-9337-b4e4a9eae71d)

## Training Graphs
Here are the plots showing how the model's loss flattens with each epoch count 
![waifu-vae-loss](https://github.com/K3dA2/VAE/assets/112480809/09ee7e65-54e0-4753-8a2c-459229d191ef)
![waifu-vae-ema](https://github.com/K3dA2/VAE/assets/112480809/9a45df56-c979-489b-88f2-9683f3995c8b)


### Future Work

Experiment with alternative architectures and loss functions weights to improve the VAE's performance.

Explore different hyperparameter settings to enhance training stability and convergence.

Incorporate advanced techniques such as Conditional VAEs or VAE-GANs for more sophisticated generation tasks.

### Contributions

Contributions to this project are welcome! Feel free to fork the repository, make improvements, and submit pull requests.