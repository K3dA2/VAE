import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils import ResNet
import matplotlib.pyplot as plt
import uuid
import os
import cv2


class Encoder(nn.Module):
    def __init__(self, z_dim = 64) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3,32,kernel_size=3,stride=2)
        self.conv1 = nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=2)
        self.dropout = nn.Dropout2d(0.25)
        self.mu = nn.Linear(512,z_dim)
        self.sigma = nn.Linear(512,z_dim)
        self.batch_norm = nn.BatchNorm2d(64)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # Flatten the output
        batch_size = x.size(0)
        shape = x.shape
        flattened_size = x.view(batch_size, -1).size(1)  # Calculate the flattened size dynamically
        x = x.view(batch_size, -1)
        
        x = nn.Linear(flattened_size,512).to(x.device)(x)

        mu = self.mu(x)
        l_sigma = self.sigma(x)

        z = mu + torch.exp(l_sigma/2)*torch.rand_like(l_sigma)

        return mu,l_sigma,z,flattened_size,shape


class Decoder(nn.Module):
    def __init__(self,z_dim=64,data_shape=(64,64)) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.ln = nn.Linear(z_dim,576)
        self.conv = nn.ConvTranspose2d(64,128,kernel_size=3,stride=2)
        self.conv1 = nn.ConvTranspose2d(128,256,kernel_size=3,stride=2)
        self.conv2 = nn.ConvTranspose2d(256,32,kernel_size=3,stride=2)
        self.conv3 = nn.ConvTranspose2d(32,3,kernel_size=3,stride=2,output_padding=1)
        self.batch_norm = nn.BatchNorm2d(128)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(0.2)
        self.data_shape = data_shape

    def forward(self,z):
        
        z = self.ln(z)
        z = z.view(-1,64,int(self.data_shape[0]//16 - 1),int(self.data_shape[0]//16 - 1))
        
        z = self.conv(z)
        z = self.batch_norm(z)
        z = F.leaky_relu(z)
        z = self.dropout(z)
        
        z = self.conv1(z)
        z = self.batch_norm1(z)
        z = F.leaky_relu(z)
        z = self.dropout(z)

        z = self.conv2(z)
        z = self.batch_norm2(z)
        z = F.leaky_relu(z)
        z = self.dropout(z)

        z = self.conv3(z)
        z = F.sigmoid(z)

        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder,decoder,
                 z_dim = 64,device = 'mps') -> None:
        super().__init__()
        self.encoder = encoder(z_dim)
        self.decoder = decoder(z_dim)
        self.device = device
        self.z_dim = z_dim
    
    def forward(self,x):

        mu,l_sigma,z,_,shape = self.encoder.forward(x)

        out = self.decoder.forward(z)

        return mu,l_sigma,out
    
    def inferenceR(self,should_save = True):
        z_var = torch.rand(1,self.z_dim).to(self.device)
        self.decoder.eval()
        pred = self.decoder.forward(z_var)
        if should_save:
            plt.imshow(np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0)))
            plt.axis('off')  # If you want to hide the axes
            # Generate a random filename
            random_filename = str(uuid.uuid4()) + '.png'

            # Specify the directory where you want to save the image
            save_directory = 'Images/'

            # Create the full path including the directory and filename
            full_path = os.path.join(save_directory, random_filename)
            # Save the image with the random filename
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.imshow(np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0)))
            plt.show()
        self.decoder.train()

    def reconstruct(self, data):
        self.decoder.eval()
        self.encoder.eval()
        
        _,_,l_img,_,_ = self.encoder(data)  # Pass data through the decoder
        img_rec = self.decoder(l_img)  # Pass the output of the decoder to the encoder
        
        plt.imshow(np.transpose(img_rec[-1].cpu().detach().numpy(), (1, 2, 0)))
        plt.axis('off')  # Hide the axes

        # Generate a random filename and specify the directory to save the image
        random_filename = str(uuid.uuid4()) + '.png'
        save_directory = 'Reconstructed/'
        full_path = os.path.join(save_directory, random_filename)
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0)  # Save the image
        self.decoder.train()
        self.encoder.train()





class TestEncoder(unittest.TestCase):
    def test_forward(self):
        '''
        model = Encoder()
        input_tensor = torch.randn(1, 3, 64, 64)
        mu,sigma,z = model.forward(input_tensor)
        self.assertEqual(mu.shape,(1,64))
        
        model = Decoder((1*64*7*7))
        input_tensor = torch.randn(1, 64)
        x_size = (1, 64, 7, 7)
        out = model.forward(input_tensor,x_size)
        self.assertEqual(out.shape,(1,3,64,64))
        '''
        model = VariationalAutoencoder(Encoder,Decoder)
        input_tensor = torch.randn(1,3,64,64)
        x_size = (1, 64, 7, 7)
        _,_,out = model.forward(input_tensor)
        self.assertEqual(out.shape,(1,3,64,64))


if __name__ == "__main__":
    unittest.main()