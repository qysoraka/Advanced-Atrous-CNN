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
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    feature_list = []
    
    # Select one audio per class and extract feature
    for label in labels:
        
        for audio_name in audio_names:
        
            if label in audio_name:
                
                audio_path = os.path.join(audios_dir, audio_name)
                
                feature = calculate_logmel(audio_path=audio_path, 
                                        sample_rate=sampl