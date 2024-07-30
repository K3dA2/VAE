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
Here are some sample Images reconstructions of the current model (still in training)

![0e689b38-e58b-4acf-b988-94f0bc76b274](https://github.com/user-attachments/assets/4543a8d7-456a-4415-a9c5-a705e132239e)

![12d07dab-d5a8-4e46-abc6-6b613d4c6a1d](https://github.com/user-attachments/assets/b5009a49-a968-4d22-a63c-c5189918d15a)

![4764370c-1598-4107-bc8d-3a3fa1d74fa1](https://github.com/user-attachments/assets/3f497455-58c2-4744-b25e-356196b65ef7)


### Future Work

Experiment with alternative architectures and loss functions weights to improve the VAE's performance.

Explore different hyperparameter settings to enhance training stability and convergence.

Incorporate advanced techniques such as Conditional VAEs or VAE-GANs for more sophisticated generation tasks.

### Contributions

Contributions to this project are welcome! Feel free to fork the repository, make improvements, and submit pull requests.
