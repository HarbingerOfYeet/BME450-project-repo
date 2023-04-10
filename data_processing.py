# imports
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import pandas as pd

# define MFCC transformation
n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256
sample_rate = 6000

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

# custom audio dataset
class AudioFileDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transform=mfcc_transform, target_transform=None):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, idx):
        # get path to audio file
        filename = self.audio_labels.iloc[idx, 0][:-3] + "wav"
        audio_path = os.path.join(self.audio_dir, filename)
        
        # load audio file
        speech_waveform, sample_rate = torchaudio.load(audio_path)
        
        # get associated label for audio_path
        label = self.audio_labels.iloc[idx, 1]
        
        # transform mfcc if there is a transform
        if self.transform:
            mfcc = self.transform(speech_waveform)
        if self.target_transform:
            label = self.target_transform(label)
        return mfcc[:,:,:150], label                # mfcc has shape [1, n_mfcc, time]

# define training data and dataloader
training_data = AudioFileDataset("filename_age.csv", "wav_training_data")
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))