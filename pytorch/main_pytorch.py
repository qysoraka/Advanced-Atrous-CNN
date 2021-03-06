
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       plot_confusion_matrix, print_accuracy, 
                       write_leaderboard_submission, write_evaluation_submission)
from models_pytorch import move_data_to_gpu, CnnPooling_Max, CnnPooling_Avg, CnnPooling_Attention
import config
from torch.autograd import Variable

Model = CnnPooling_Attention
batch_size = 16


def evaluate(model, generator, data_type, devices, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type, 
                                                devices=devices, 
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]

    loss = F.nll_loss(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy()

    loss = float(loss)
    
    confusion_matrix = calculate_confusion_matrix(
        targets, predictions, classes_num)
    
    accuracy = calculate_accuracy(targets, predictions, classes_num, 
                                  average='macro')

    return accuracy, loss

def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x, batch_y, batch_audio_names) = data
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        batch_output = model(batch_x)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        
        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names
    
    if return_target: