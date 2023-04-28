import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms as T

# define MFCC transformation
n_fft = 2048
# win_length = None
hop_length = 256
n_mels = 128
n_mfcc = 128
sample_rate = 6000
# current_array=np.zeros(10)

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
class AudioDataset(Dataset):
    def __init__(self, root, annotations_file, audio_dir, transform=mfcc_transform, target_transform=None):
        self.root = root
        self.audio_labels = pd.read_csv(root + annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, idx):
        # get path to audio file
        filename = self.audio_labels.iloc[idx, 0][:-3] + "wav"
        audio_path = os.path.join(os.path.abspath(self.root + "/" + self.audio_dir), filename)
        
        # load audio file
        speech_waveform, sample_rate = torchaudio.load(audio_path)
        
        # get associated label for audio_path
        label = self.audio_labels.iloc[idx, 1]
        
        # transform mfcc if there is a transform
        if self.transform:
            mfcc = self.transform(speech_waveform)
        if self.target_transform:
            label = self.target_transform(label)
        return mfcc[:,:,:128], label                # mfcc has shape [1, n_mfcc, time]