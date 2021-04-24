import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



class EmbeddingLayers_pooling(nn.Module):
    def __init__(self):
        super(EmbeddingLayers_pooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=1,
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=2,
                               padding=(4, 4), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=4,
                               padding=(8, 8), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=8,
                               padding=(16, 16), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

