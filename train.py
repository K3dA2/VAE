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
#from models import VAE,Encoder,Decoder
#from vae import VariationalAutoencoder,Encoder,Decoder
from latent_img_vae import VAE,Encoder,Decoder

def get_data(path):
    image_extensions = ['.jpg']
    image_names = []
    for filename in os.listdir(path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_names.append(filename)
    return image_names

def reshape_img(img,size = (64,64)):
    data = cv2.resize(img,size)
    data = np.transpose(data,(2,0,1))
    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def kl_loss(mu,l_sigma):
    kl_loss = -0.5 * torch.sum(1 + l_sigma - mu**2 - torch.exp(l_sigma))
    return kl_loss

def training_loop(n_epochs, optimizer, model, loss_fn, device,
                  l_weight = 100, kl_weight = 0.1, 
                  epoch_start = 0, batch_size = 64, 
                  data_length = 4000,max_grad_norm=1.0):
    data_idx = list(range(0,data_length))
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(epoch_start, n_epochs + 1):
            random.shuffle(data_idx)
            loss_train = 0.0
            # Use tqdm function for the progress bar
            with tqdm(range(0, (data_length//batch_size)), desc=f'Epoch {epoch}', unit=' batch') as pbar:
                st = 0
                for x in pbar:
                    # Training loop code
                    sp = st + batch_size
                    if sp > data_length:
                        st = 0
                        sp = st + batch_size
                    img_arr = []
                    for i in range(st,sp):
                        img = plt.imread(path + '/' + image_names[data_idx[i]])
                        img = reshape_img(img)
                        img = np.expand_dims(img, 0)
                        img_arr.append(img)
                    st+= batch_size
                    imgs = np.squeeze(np.array(img_arr))

                    if torch.is_tensor(imgs):
                        imgs = imgs.type(torch.float32).to(device)
                    else:
                        imgs = torch.from_numpy(imgs).type(torch.float32).to(device)
                    
                    #Normalize images
                    imgs = imgs/255

                    mu,sigma,outputs = model(imgs)
                    
                    mse_loss = loss_fn(outputs, imgs)
                    kl = kl_loss(mu,sigma)
                    loss = (l_weight* mse_loss) + (kl*kl_weight)
                    
                    loss_train += loss.mean().item()  # Accumulate loss values
                    pbar.set_postfix(loss=loss.item(),mse=mse_loss.item(),kl_loss=kl.item())

                    loss.backward()

                    # Clip gradients
                    utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()  # Reset gradients

                    
                
            avg_loss_epoch = loss_train / (data_length//batch_size)
            with open("waifu-VAE1-4000-loss.txt", "a") as file:
                file.write(f"{avg_loss_epoch}\n")
            
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                avg_loss_epoch))
            #torch.save(model.state_dict(), path1)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
            #inference(model, device)
            if epoch % 5 == 0:
                model.inferenceR()
            
                


if __name__ == "__main__":
    path = '/Users/ayanfe/Documents/Datasets/animefaces256cleaner'
    model_path = '/Users/ayanfe/Documents/Code/VAE/Weights/waifu-vae-resnet.pth'
    image_names = get_data(path)
    print("Image Length: ",len(image_names))

    device = torch.device("mps")
    model = VAE(Encoder,Decoder,z_dim=200)
    #model = VariationalAutoencoder(Encoder,Decoder,z_dim=200,device=device)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=5e-4)
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']

    print("Total parameters: ",count_parameters(model))
    loss_fn = nn.MSELoss()  #  <4>

    training_loop(  # <5>
        n_epochs = 1000,
        optimizer = optimizer,
        model = model,
        loss_fn = loss_fn,
        device = device,
        batch_size = 16,
        epoch_start = 1,
        data_length = 10_000
    )
