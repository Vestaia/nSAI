import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
import numpy as np
from time import time
from IPython import display
import matplotlib.pyplot as plt

feature_maps = 16
LOSS_FUNCTION = nn.MSELoss()
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv1d(2, feature_maps, 7)
        self.conv1 = nn.Conv1d(feature_maps, feature_maps, 3)
        self.bn2 = nn.BatchNorm1d(feature_maps)
        self.dense3 = nn.Linear(1760,1)
    
    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv0(x), 2))
        x = F.relu(F.max_pool1d(self.bn2(self.conv1(x)), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dense3(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        loss = torch.mean(torch.cosh(torch.log(torch.abs(y_t - y_prime_t)+1)))
        return loss

def learn(net, optim, data, target, batch_size=10):
    device = torch.device("cuda:0" if (next(net.parameters()).is_cuda) else "cpu")
    losses = []
    calc = LOSS_FUNCTION.to(device)
    net.train()
    plt.ion()
    for batch in range(np.int(np.floor(len(data)/batch_size))):
        X = data[batch*batch_size:(batch+1)*batch_size].to(device)
        y = target[batch*batch_size:(batch+1)*batch_size].to(device)

        net.zero_grad()
        output = net(X)
        loss = calc(output, y)
        loss.backward()
        
        optim.step()

        losses.append(loss.item())
        
        # if(batch%20==0):
        #     plot_train(losses)

    return np.mean(losses)

def test(net, data, target, batch_size=10):
    device = torch.device("cuda:0" if (next(net.parameters()).is_cuda) else "cpu")
    losses = []
    calc = LOSS_FUNCTION.to(device)
    net.eval()

    for batch in range(np.int(np.floor(len(data)/batch_size))):
        X = data[batch*batch_size:(batch+1)*batch_size].to(device)
        y = target[batch*batch_size:(batch+1)*batch_size].to(device)

        output = net(X)
        loss = calc(output, y).detach()

        losses.append(loss.item())

    return np.mean(losses)

def plot_train(trainloss, testloss=None):
    plt.clf()
    plt.plot(np.linspace(2,len(trainloss),len(trainloss)-1), trainloss[1:], label="train")
    if testloss is not None:
        plt.plot(np.linspace(2,len(testloss),len(testloss)-1), testloss[1:], label="test")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss') 
    display.display(plt.gcf())
    display.clear_output(wait=True)

def train(net, optim, data, target, batch_size=10, validation_size=None, epochs=1, plot=True):

    if validation_size is None:
        validation_size = data.size(0) // 10

    #Scrambling data for training
    ind = np.random.permutation(data.size(0))

    #Splitting data to train and validation
    validationX = data[ind[:validation_size]]
    validationy = target[ind[:validation_size]]
    trainX = data[ind[validation_size:]]
    trainy = target[ind[validation_size:]]

    #Training
    print("Starting Training:")
    train_loss = []
    validation_loss = []
    plt.ion()
    start = time()
    epoch = 0

    for i in range(epochs):
        loss = learn(net, optim, trainX, trainy, batch_size=batch_size)
        train_loss.append(loss)

        loss = test(net, validationX, validationy, batch_size=batch_size)
        validation_loss.append(loss)

        elapsed = time() - start

        print("Epoch:", i+1)
        print("Train loss:", train_loss[i])
        print("validation loss:", validation_loss[i])
        print("Time:", elapsed)
        if(plot):
            plot_train(train_loss, validation_loss)
        epoch += 1

