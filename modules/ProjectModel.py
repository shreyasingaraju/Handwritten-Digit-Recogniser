import torch
import torchvision
import torchvision.datasets as datasets

class ProjModel:
    def __init__(self):
        self.attr = "test"

    def downloadTrainSet(self):
        self.mnist_trainset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)

    def downloadTestSet(self):
        self.mnist_testset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=None) 

    def TrainModel(self):
        self.attr2 = "test2"
        
        
        
