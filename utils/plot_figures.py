import argparse
import matplotlib.pyplot as plt
import os

from features import LogMelExtractor, calculate_logmel
import config


def plot_logmel(args):
    """Plot log Mel feature of one audio per class. 
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    labels = config.labels
    
    # Paths
    audio_names = os.listdir(audios_dir)
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                