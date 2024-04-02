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


class Encoder(nn.Module):
    def __init__(self, z_dim = 64) -> None:
        super().__init__()
        self.res = ResNet(3,64,useMaxPool=True)
        self.res1 = ResNet(64,128,useMaxPool=True)
        self.res2 = ResNet(128,512,useMaxPool=True)
        #self.ln = nn.Linear(None,512)
        self.mu = nn.Linear(512,z_dim)
        self.sigma = nn.Linear(512,z_dim)
        
    def forward(self,x):
        x = self.res(x)
        x = self.res1(x)
        x = self.res2(x)

        # Flatten the output
        batch_size = x.size(0)
        flattened_size = x.view(batch_size, -1).size(1)  # Calculate the flattened size dynamically
        x = x.view(batch_size, -1)
        
        x = nn.Linear(flattened_size,512).to(x.device)(x)

        mu = self.mu(x)
        l_sigma = self.sigma(x)

        z = mu + torch.exp(l_sigma/2)*torch.rand_like(l_sigma)

        return mu,l_sigma,z


class Decoder(nn.Module):
    def __init__(self, z_dim=64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.ln = nn.Linear(z_dim,256)
        self.res = ResNet(1,64,upscale=True)
        self.res1 = ResNet(64,128,upscale=True)
        self.res2 = ResNet(128,16)
        self.conv = nn.Conv2d(16,3,kernel_size=3, padding=1)

    def forward(self,z):
        batch_size = z.size(0)
        z = self.ln(z)
        z = z.view(batch_size,1,16,16)
        
        z = self.res(z)
        
        z = self.res1(z)
        z = self.res2(z)

        z = F.relu(self.conv(z))

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




class TestEncoder(unittest.TestCase):
    def test_forward(self):
        '''
        model = Encoder()
        input_tensor = torch.randn(1, 3, 64, 64)
        mu,sigma,z = model.forward(input_tensor)
        self.assertEqual(mu.shape,(1,64))
        '''
        model = Decoder()
        input_tensor = torch.randn(1, 64)
        out = model.forward(input_tensor)
        self.assertEqual(out.shape,(1,3,64,64))

if __name__ == "__main__":
    unittest.main()