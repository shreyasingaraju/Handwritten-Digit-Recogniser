

from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F

class ProjModel:
    def __init__(self):
        self.attr = "test"

    def downloadTrainSet(self):
        self.mnist_trainset = datasets.MNIST(root='mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

        self.train_loader = data.DataLoader(dataset=self.mnist_trainset,
                                           batch_size=64,
                                           shuffle=True)

    def downloadTestSet(self):
        self.mnist_testset = datasets.MNIST(root='mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

        self.test_loader = data.DataLoader(dataset=self.mnist_testset,
                                          batch_size=64,
                                          shuffle=False)

    def TrainModel(self):
        print("TrainModel() called")
        self.net = Net()
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.5)

        for epoch in range(1,3):
            print("Epoch")
            self.net.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        print("TrainModel() finished")

    def TestModel(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        device = 'cuda' if cuda.is_available() else 'cpu'
        for data, target in self.test_loader:
            data, target = data.to(device), target.to(device)
            output = self.net(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} '
            f'({100. * correct / len(self.test_loader.dataset):.0f}%)')
        

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
        
        
