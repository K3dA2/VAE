import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import datetime
import random
import torch.nn.utils as utils
from latent_img_vae import VAE,Encoder,Decoder
from utils import get_data_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def kl_loss(mu,l_sigma):
    kl_loss = -0.5 * torch.sum(1 + l_sigma - mu**2 - torch.exp(l_sigma))
    return kl_loss

def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, valid_loader,
                  max_grad_norm=1.0, epoch_start=0,
                kl_weight= 0.0001):
    
    model.train()
    for epoch in range(epoch_start, n_epochs):
        loss_train = 0.0
        recon_loss_train = 0.0
        kl_loss_train = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        for imgs, _ in progress_bar:
            imgs = imgs.to(device)

            mu, logvar, outputs = model(imgs)
            
            mse_loss = loss_fn(outputs, imgs)
            kl = kl_loss(mu, logvar)
            loss = mse_loss + (kl * kl_weight)
            
            loss_train += loss.item()
            recon_loss_train += mse_loss.item()
            kl_loss_train += kl.item()

            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item(), recon_loss=mse_loss.item(), kl_loss=kl.item())

        avg_train_loss = loss_train / len(data_loader)
        avg_recon_loss = recon_loss_train / len(data_loader)
        avg_kl_loss = kl_loss_train / len(data_loader)
        
        with open("vae_training_loss.txt", "a") as file:
            file.write(f"{avg_train_loss},{avg_recon_loss},{avg_kl_loss}\n")
        
        with open("vae_training_recon_loss.txt", "a") as file:
            file.write(f"]{avg_recon_loss}\n")
        
        print(f'{datetime.datetime.now()} Epoch {epoch}, Training loss {avg_train_loss}, Recon loss {avg_recon_loss}, KL loss {avg_kl_loss}')
        
        if epoch % 5 == 0:
            model.inferenceR()
            model.reconstruct(next(iter(valid_loader))[0].to(device))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)




if __name__ == "__main__":
    path = '/Users/ayanfe/Documents/Datasets/Waifus/Train'
    val_path = '/Users/ayanfe/Documents/Datasets/Waifus/Val'
    model_path = 'Weights/waifu-vae-resnet.pth'
    epoch = 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    model = VAE(Encoder,Decoder,z_dim=200)
    #model = VariationalAutoencoder(Encoder,Decoder,z_dim=200,device=device)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=3e-4)

    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(epoch)
    

    print("Total parameters: ",count_parameters(model))
    loss_fn = nn.MSELoss().to(device)  #  <4>
    data_loader = get_data_loader(path, batch_size=64,num_samples=30_000)
    val_loader = get_data_loader(val_path, batch_size=1,num_samples=10_000)
    
    model.inferenceR()
    model.reconstruct(next(iter(val_loader))[0].to(device))
    
    
    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        valid_loader=val_loader,
        epoch_start=epoch
    )