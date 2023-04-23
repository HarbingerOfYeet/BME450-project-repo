# imports
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import AudioFileDataset as afd
import NeuralNetwork as mlp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# model parameters
# learning_rate = 1e-3
# batch_size = 64
# epochs = 5

# train_loss_arr = []
# test_loss_arr = []
# accuracy_arr = []

# load_data
# data_dir - combine both datasets in a common directory
# wrap train and test data in a function
def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = afd.AudioFileDataset("train_files.csv", "wav_training_data")
    test_data = afd.AudioFileDataset("test_files.csv", "wav_training_data")

    return train_data, test_data


# train_func
# config - hyperparameters to train
# checkpoint_dir - restore checkpoints
# data_dir - directory to load and store data
def train_func(config, checkpoint_dir=None, data_dir=None):
    
    model = mlp.NeuralNetwork(config["l1"], config["l2"])

    # multi GPU support with data parallel training
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(model)
        model.to(device)

    # initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # set checkpoints if checkpoint_dir is true
    if checkpoint_dir:
        model_state, optimizer_state =torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_data = load_data(data_dir)         # load train data

    # define dataloaders
    train_loader = DataLoader(
        train_data, 
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8
        )

    for epoch in range(10):         # loop over dataset
        running_loss = 0.0
        epoch_steps = 0
        num_batches = len(train_loader)
        for batch, data in enumerate(train_loader):
            # data is list of [inputs, labels]
            # send data to GPU device
            audio, labels = data
            audio, labels = audio.to(device), labels.to(device)
            
            # Compute prediction and loss
            pred = model(audio)
            loss = loss_fn(pred, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if batch % 10 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, batch + 1, running_loss / epoch_steps))
                running_loss = 0.0

        # track the average train loss for each epoch
        # train_loss_arr.append(total_loss / num_batches)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict(), path))

        print("Finished Training")
        


#Test script call function
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")

def test_accuracy(net, device="cpu"):
    train_data, test_data = load_data()

    # test dataloader
    test_loader = DataLoader(
        test_data,
        batch_size=4, 
        shuffle=False,
        num_workers=2
    )
    
    # test each data in the test dataloader
    correct = 0
    size = len(test_loader.dataset)
    with torch.no_grad():
        for data in test_loader:
            audio, labels = data
            audio, labels = audio.to(device), labels.to(device)
            pred = net(audio)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return correct / size



# main
# main function to find optimal hyperparameters
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(5, 10)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(5, 10)),
        "l3": tune.sample_from(lambda _: 2**np.random.randint(5, 10)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "l3", "lr", "batch_size"]
        metric_columns=["loss", "accuracy", "training_iteration"]
    )
    result = tune.run(
        partial(train_func, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

# convert lists to arrays
# xdata = np.arange(0, 10, 1)
# train_data = np.array(train_loss_arr)
# test_data = np.array(test_loss_arr)
# accuracy_data = np.array(accuracy_arr)

# # plot results
# fig1, ax1 = plt.subplots()
# ax1.set_xlabel("Epochs")
# ax1.set_ylabel("Loss")
# ax1.set_title("Loss vs. Epochs for Testing and Training")
# ax1.plot(xdata, train_data, color="r")
# ax1.plot(xdata, test_data, color="b")
# plt.show()

# fig2, ax2 = plt.subplots()
# ax2.set_xlabel("Epochs")
# ax2.set_ylabel("Accuracy")
# ax2.set_title("Accuracy vs. Epochs")
# ax2.plot(xdata, train_data, color="g")
# plt.show()