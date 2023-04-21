# imports
import torch
from torch import nn
from torch.utils.data import DataLoader
import AudioFileDataset
import NeuralNetwork

train_loss_arr = []
test_loss_arr = []

# train loop function
def train_loop(dataloader, model, loss_fn, optimizer):
    total_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            total_loss += loss

    # track the average train loss for each epoch
    train_loss_arr.append(total_loss / num_batches)

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
    test_loss_arr.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model = NeuralNetwork()

# model parameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# initialize loss function
loss_fn = nn.CrossEntropyLoss()

# initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# define training data and dataloader
training_data = AudioFileDataset("train_files.csv", "wav_training_data")
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# define test data and dataloader
test_data = AudioFileDataset("test_files.csv", "wav_training_data")
test_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

#Test script call function
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")