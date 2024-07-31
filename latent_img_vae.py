import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils import ResNet,StridedResNet,ResNetTranspose
import matplotlib.pyplot as plt
import uuid
import os
import cv2


class Encoder(nn.Module):
    def __init__(self,z_dim) -> None:
        super().__init__()
        self.res = StridedResNet(3,64)
        self.res1 = StridedResNet(64,128)
        self.res2 = StridedResNet(128,512)
        self.mu = nn.Conv2d(512,4,kernel_size=3,padding=1)
        self.l_sigma = nn.Conv2d(512,4,kernel_size=3,padding=1)
        
    def forward(self,x):
        x = self.res(x)
        
        x = self.res1(x)
        
        x = self.res2(x)
        

        mu = self.mu(x)
        l_sigma = self.l_sigma(x)

        z = mu + torch.exp(l_sigma/2) * torch.rand_like(l_sigma)

        return mu,l_sigma,z


class Decoder(nn.Module):
    def __init__(self, z_dim=64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.fres = ResNetTranspose(4,32)
        self.res = ResNetTranspose(32,64)
        self.res1 = ResNet(64,128)
        self.res2 = ResNetTranspose(128,256)
        self.res3 = ResNet(256,64)
        self.conv = nn.Conv2d(64,3,kernel_size=3, padding=1)

    def forward(self,z):
        z = self.fres(z)
        
        z = self.res(z)
        
        z = self.res1(z)
        
        z = self.res2(z)
        
        z = self.res3(z)
        

        z = self.conv(z)

        return z

class VAE(nn.Module):
    def __init__(self, encoder,decoder,z_dim = 64,device = 'mps') -> None:
        super().__init__()
        self.encoder = encoder(z_dim)
        self.decoder = decoder(z_dim)
        self.device = device
        self.z_dim = z_dim
    
    def forward(self,x):
        mu,l_sigma,z = self.encoder.forward(x)

        out = self.decoder.forward(z)

        return mu,l_sigma,out
    
    def inferenceR(self,should_save = True):
        z_var = torch.rand(1,4,4,4).to(self.device)
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
        
        # Pass data through the encoder and decoder
        _, _, l_img = self.encoder(data)
        img_rec = self.decoder(l_img)
        
        # Convert tensors to numpy arrays and transpose the dimensions
        img_orig = np.transpose(data[-1].cpu().detach().numpy(), (1, 2, 0))
        img_rec = np.transpose(img_rec[-1].cpu().detach().numpy(), (1, 2, 0))
        
        # Concatenate the original and reconstructed images horizontally
        img_concat = np.concatenate((img_orig, img_rec), axis=1)
        
        # Display the concatenated image
        plt.imshow(img_concat)
        plt.axis('off')  # Hide the axes
        
        # Generate a random filename and specify the directory to save the image
        random_filename = str(uuid.uuid4()) + '.png'
        save_directory = 'Reconstructed/'
        os.makedirs(save_directory, exist_ok=True)
        full_path = os.path.join(save_directory, random_filename)
        
        # Save the image
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
        
        # Restore the model to training mode
        self.decoder.train()
        self.encoder.train()
        


class TestEncoder(unittest.TestCase):
    def test_forward(self):
        '''
        model = Encoder()
        input_tensor = torch.randn(1, 3, 64, 64)
        mu,sigma,z = model.forward(input_tensor)
        self.assertEqual(mu.shape,(1,64))
        '''
        model = Decoder()
        input_tensor = torch.randn(1, 4,4,4)
        out = model.forward(input_tensor)
        self.assertEqual(out.shape,(1,3,32,32))
        

if __name__ == "__main__":
    unittest.main()