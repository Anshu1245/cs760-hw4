import torch
import pandas as pd
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader




class feedforward(nn.Module):
    def __init__(self, d, d1, k):
        super(feedforward, self).__init__()
        self.layer1 = nn.Linear(d, d1)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(d1, k)
        self.softmax = nn.Softmax()

    def forward(self, x):
        pred = self.layer1(x)
        pred = self.sigmoid(pred)
        pred = self.layer2(pred)
        pred = self.softmax(pred)
        return pred



if __name__ == '__main__':
    net = feedforward(784, 200, 10)
    lossfn = nn.CrossEntropyLoss(reduction='sum')
    optimiser = torch.optim.SGD(net.parameters(), lr=0.001)


    dataset = np.array(pd.read_csv('./mnist/train.csv'))
    X, Y = dataset[:, 1:], dataset[:, 0]
    X_eval, Y_eval = torch.Tensor(X[:1000]), torch.Tensor(Y[:1000])
    X_train, Y_train = torch.Tensor(X[1000:]), torch.Tensor(Y[1000:])

    tensor_dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(tensor_dataset)

    train_loss = []
    for epoch in range(50):
        for i,(x_train, y_train) in enumerate(dataloader):
            y_pred = net(x_train)
            
            loss  = lossfn(y_pred, y_train.reshape(-1,1))
            train_loss.append(loss)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if epoch%1 == 0:
                print(loss)
                


