# Advanced-Atrous-CNN

Pytorch implementation of the research paper 'Attention-based Atrous Convolutional Neural Networks: Visualisation and Understanding Perspectives of Acoustic Scenes'.

# Data

DCASE 2018 Task 1 - Acoustic Scene Classification, containing:

subtask A: data from device A

subtask B: data from device A, B, and C

# Preparation

channels:
  - pytorch
dependencies:
  - matplotlib=2.2.2
  - numpy=1.14.5
  - h5py=2.8.0
  - pytorch=0.4.0
  - pip:
    - audioread==2.1.6
    - librosa==0.6.1
    - scikit-learn==0.19.1
    - soundfile==0.10.2

# Run 

sh runme.sh

In runme.sh, please run the following files for different tasks:
1. feature extraction: utils/features.py
2. training a model, 