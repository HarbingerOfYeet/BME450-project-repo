import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlsxwriter

# define MFCC transformation
n_fft = 2048
win_length = None
hop_length = 256
n_mels = 128
n_mfcc = 128
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
        return mfcc[:,:,:128], label                # mfcc has shape [1, n_mfcc, time]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128*128, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 9)       # 9 categories for age
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# train loop function
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    counter=0 #counter to count observed batch numbers
    current_array=np.zeros(size)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        current_array[counter]=loss
        counter= counter+1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            #print(type(current))
            #current_array[counter]=loss
            #counter= counter+1
            #print(current_array[counter])
            #print(counter)
    #return np.mean(current_array)
    return np.sum(current_array)/np.count_nonzero(current_array)

model = NeuralNetwork()

# model parameters
learning_rate = 1e-2 #Original 1e-3 
batch_size = 32 #Original 64
epochs = 100 #Would like to be 100

# initialize loss function
loss_fn = nn.CrossEntropyLoss()

# initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# define training data and dataloader
training_data = AudioFileDataset("train_files.csv", "wav_training_data")
train_dataloader = DataLoader(training_data, batch_size, shuffle=True)

# train loop
#for t in range(epochs):
#    print(f"Epoch {t+1}\n-------------------------")
#    c_array= train_loop(train_dataloader, model, loss_fn, optimizer)
    #batch_no = np.arange(np.size(c_array)-1)
    #c_keeparray([t])=

    #plt.figure()
    #plt.scatter(batch_no, c_array)
    #plt.xlabel('Batch Number')
    #plt.ylabel('Loss Value')
    #if t == 0:
    #    plt.title('Loss Function Plot for Epoch 1')
    #elif t == 1:
    #   plt.title('Loss Function Plot for Epoch 2')
    #elif t == 2:
    #    plt.title('Loss Function Plot for Epoch 3')
    #elif t == 3:
    #    plt.title('Loss Function Plot for Epoch 4')
    #elif t == 4:
    #    plt.title('Loss Function Plot for Epoch 5')
    #plt.show()
#print("Done!")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct), test_loss

# define test data and dataloader
test_data = AudioFileDataset("test_files.csv", "wav_training_data")
test_dataloader = DataLoader(training_data, batch_size, shuffle=True)

#Test script call function
acc_array = []
loss_array = []
epoch_array = []
train_avg_loss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------")
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    #batch_no = np.arange(np.size(c_array))
    acc, avg_loss = test_loop(test_dataloader, model, loss_fn)
    acc_array.append(acc)
    loss_array.append(avg_loss)
    epoch_array.append(t+1)
    train_avg_loss.append(train_loss)

#plt.figure(1)
#plt.scatter(epoch_array,acc_array)
#plt.xlabel('Epoch Iterations')
#plt.ylabel('Accuracy (%)')
#plt.title('Accuracy against Epoch iterations Plot')
#plt.show()

#plt.figure(2)
#plt.scatter(epoch_array,loss_array)
#plt.xlabel('Epoch Iterations')
#plt.ylabel('Loss Value')
#plt.title('Loss Value against Epoch iterations Plot')
#plt.show()

fig, ax = plt.subplots()
l1a=ax.plot(epoch_array, loss_array)
l1b=ax.plot(epoch_array, train_avg_loss)
ax.set_xlabel("Epoch Iterations", fontsize = 14)
ax.set_ylabel("Loss Value", fontsize = 14)
plt.legend(["Test Average Loss","Train Average Loss"])
plt.title('Loss Progression against Epoch')
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(epoch_array, acc_array)
ax2.set_xlabel("Epoch Iterations", fontsize = 14)
ax2.set_ylabel("Accuracy (%)", fontsize = 14)
plt.title('Accuracy Progression against Epoch')
plt.show()

fig.savefig('Loss Progression against Epoch.png', format = 'png', dpi = 100, bbox_inches='tight')
fig2.savefig('Accuracy Progression against Epoch.png', format = 'png',  dpi = 100, bbox_inches='tight')
 
 ##Saving Data to an Excel Workbook
workbook = xlsxwriter.Workbook('nn Run Data.xlsx')
worksheet = workbook.add_worksheet()
 
# Start from the first cell.
# Rows and columns are zero indexed.
row = 0
column = 0
 
# iterating through content list
for item in epoch_array :
 
    # write operation perform
    worksheet.write(row, column, item)
 
    row += 1

row=0
for item in train_avg_loss :
 
    # write operation perform
    worksheet.write(row, column+1, item)
 
    row += 1

row=0

for item in loss_array :
 
    # write operation perform
    worksheet.write(row, column+2, item)
 
    row += 1

row=0

for item in acc_array :
 
    # write operation perform
    worksheet.write(row, column+3, item)
 
    row += 1

workbook.close()

##Torch.save nn run
#PATH="100_epoch_model.pth"
#torch.save(model.state_dict(), PATH)

##Load torch.save
#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()-z