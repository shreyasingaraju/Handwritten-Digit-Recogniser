from PyQt5.QtCore import pyqtSignal

from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn
import torch.nn.functional as F
import numpy as np

from math import trunc

class ProjModel:
    def __init__(self):
        global batch_size
        batch_size = 64
        self.setCancelFlag(False)
    
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.net = Net()
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.5)

    def setCancelFlag(self, flag):
        self.cancel_flag = flag

    def downloadTrainSet(self):
        # self.mnist_trainset = datasets.MNIST(root='mnist_data/',
        #                        train=True,
        #                        transform=transforms.ToTensor(),
        #                        download=True)
        try:
            self.mnist_trainset = datasets.MNIST(root='mnist_data_train/', train=True, transform=transforms.ToTensor(), download=True)
        except:
            print("Couldn't download trainset, try again")

        try:
            self.train_loader = data.DataLoader(dataset=self.mnist_trainset,
                                                batch_size=batch_size,
                                                shuffle=True)
        except:
            print("Couldn't download testset, try again")


    def downloadTestSet(self):
        # self.mnist_testset = datasets.MNIST(root='mnist_data/',
        #                       train=False,
        #                       transform=transforms.ToTensor())

        self.mnist_testset = datasets.MNIST(root='mnist_data_test/', train=False, transform=transforms.ToTensor(), download=True)

        self.test_loader = data.DataLoader(dataset=self.mnist_testset,
                                          batch_size=batch_size,
                                          shuffle=False)
      

    def trainEpoch(self):      
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cancel_flag == True:
                break

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

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
        
    def predictDigit(self, image):
        imArray = np.array(image).astype("float32")

        # Normalise colour values to be between 0 and 1
        imArray /= 255

        # Convert to tensor and then add two more dimensions to have the right format to be batch-loaded
        imTensor = torch.from_numpy(imArray)
        imTensor = torch.unsqueeze(imTensor,0)
        imTensor = torch.unsqueeze(imTensor,0)

        # Feed manipulated image into the model then softmax to normalise the net outputs to between 0 and 1 
        probArray = F.softmax(self.net(imTensor), 1).detach().numpy()

        # Convert to percentage
        probArray *= 100 

        # Remove the extra dimensions added earlier
        probArray  = np.squeeze(probArray, 0)

        digit = np.argmax(probArray)
        return digit, probArray

    def loadNet(self, path):
        try:
            self.net.load_state_dict(torch.load(path))
            print("Loaded model")
        except FileNotFoundError:
            print("Selected model does not exist")

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

# The same as the default but with two dropout layers added to reduce overfitting
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)


# Add dropout
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.l4(x))
        return self.l5(x)
        
        
