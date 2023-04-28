# imports
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import AudioDataset as ad
import NeuralNetwork as mlp
import numpy as np
import matplotlib.pyplot as plt

# lists to track losses and accuracy per epoch
train_loss_arr = []
test_loss_arr = []
accuracy_arr = []

# train_loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = 0
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = train_loss / num_batches
    train_loss_arr.append(avg_loss)
    print(f"Avg Train Loss: {avg_loss:>7f}")
    

# test_loop
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
    accuracy_arr.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# main
# main function to run training, testing, and plotting
def main():
    data_dir = os.path.abspath("./")
    model = mlp.NeuralNetwork()

    train_data = ad.AudioDataset(data_dir, "/train_files.csv", "wav_training_data")
    test_data = ad.AudioDataset(data_dir, "/test_files.csv", "test_data")


    # MODEL PARAMETERS: change after hyperparameter tuning
    learning_rate = 1e-3
    batch_size = 64

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, 
        shuffle=True,          # shuffling train data
        num_workers=2
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Done!")
    
    # convert lists to arrays
    xdata = np.arange(0, epochs, 1)
    train_data = np.array(train_loss_arr)
    test_data = np.array(test_loss_arr)
    accuracy_data = np.array(accuracy_arr)

    # plot results
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. Epochs for Testing and Training")
    ax1.plot(xdata, train_data, color="r")
    ax1.plot(xdata, test_data, color="b")
    ax1.legend(["Train", "Test"])

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs. Epochs")
    ax2.plot(xdata, accuracy_data, color="g")
    plt.show()

if __name__ == "__main__":
    main()