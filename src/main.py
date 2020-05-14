from analysis import cnn
from generate import scatter1d
import torch
import matplotlib.pyplot as plt
from torchsummary import summary

BATCH_SIZE = 50

data, target = scatter1d.gen_data(2000)
data = torch.log10(data)
ind = torch.randperm(data.size(2))
data2 = data[:,:,ind]

# plt.scatter(data[0,0], data[0,1], s=2)
# plt.show()
target = torch.log10(target)

model = cnn.Net().cuda()
optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
summary(model, input_size=(2, 450), batch_size=BATCH_SIZE)

cnn.train(model, optim, data, target, batch_size=BATCH_SIZE, validation_size=200, epochs=50)
