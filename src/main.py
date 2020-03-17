from analysis import dnn
from generate import scatter1d
import torch
import matplotlib.pyplot as plt
from torchsummary import summary

BATCH_SIZE = 100

data, target = scatter1d.gen_data(5000)
data = torch.log10(data)
target = torch.log10(target)

model = dnn.Net().cuda()
optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
summary(model, input_size=(2, 450), batch_size=BATCH_SIZE)

dnn.train(model, optim, data, target, batch_size=BATCH_SIZE, validation_size=300, epochs=100)