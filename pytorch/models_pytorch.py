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
      