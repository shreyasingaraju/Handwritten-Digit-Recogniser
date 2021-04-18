from PyQt5.QtCore import pyqtSignal

from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F

from math import trunc

class ProjModel:
    def __init__(self):
        global batch_size
        batch_size = 64
        global num_epochs
        num_epochs = 3
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.net = Net()
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.5)

    def downloadTrainSet(self):
        # self.mnist_trainset = datasets.MNIST(root='mnist_data/',
        #                        train=True,
        #                        transform=transforms.ToTensor(),
        #                        download=True)

        self.mnist_trainset = datasets.MNIST(root='mnist_data_train/', train=True, transform=transforms.ToTensor(), download=True)

        self.train_loader = data.DataLoader(dataset=self.mnist_trainset,
                                           batch_size=batch_size,
                                           shuffle=True)

    def downloadTestSet(self):
        # self.mnist_testset = datasets.MNIST(root='mnist_data/',
        #                       train=False,
        #                       transform=transforms.ToTensor())

        self.mnist_testset = datasets.MNIST(root='mnist_data_test/', train=False, transform=transforms.ToTensor(), download=True)

        self.test_loader = data.DataLoader(dataset=self.mnist_testset,
                                          batch_size=batch_size,
                                          shuffle=False)
      

    def trainEpoch(self):
        print("trainEpoch called")       
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        print("trainEpoch() finished")

    def testModel(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.net(data)
            # sum up batch loss
            test_loss += self.criterion(output, target).item()
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
    
        correct = 100 * correct / len(self.test_loader.dataset)
        return test_loss, trunc(correct.item())
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
        
        
