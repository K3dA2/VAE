import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest

'''
This ResNet class is intended to be used as the smallest unit of the block class
'''
class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32, useMaxPool = False, upscale = False):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.useMaxPool = useMaxPool
        self.upscale = upscale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        # Apply LayerNorm after conv1
        out = F.leaky_relu(self.conv1(x))
        out = self.dropout(out)
        out = F.layer_norm(out, out.size()[1:])
        
        # Apply LayerNorm after conv2
        out = F.leaky_relu(self.conv2(out))
        out = self.dropout(out)
        out = F.layer_norm(out, out.size()[1:])
        
        out1 = self.skip_conv(x)
        out = F.leaky_relu(self.conv3(out))
        
        if self.useMaxPool:
            #skip = out + out1
            out = F.max_pool2d(out + out1, 2)
            return out
        elif self.upscale:
            out = F.upsample(out + out1, scale_factor=2)
        else:
            out = F.leaky_relu(out + out1)
        return out

'''
Unit testing class
'''
class TestResNet(unittest.TestCase):
    def test_forward(self):
        model = ResNet(in_channels=16,out_channels = 16,useMaxPool=False)
        input_tensor = torch.randn(1, 16, 64, 64)  # Example input with shape (batch_size, channels, height, width)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape, (1, 16, 64, 64))  # Adjust the expected shape based on your model architecture
        
        
if __name__ == '__main__':
    unittest.main()