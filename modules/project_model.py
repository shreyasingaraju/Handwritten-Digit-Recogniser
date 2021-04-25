from PyQt5.QtCore import pyqtSignal

from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn
import torch.nn.functional as F

import numpy as np

from math import trunc

from PIL import Image, ImageOps, ImageFilter


# ModelWrapper provides attributes to contain the datasets, parameters and the models themselves. 
class ModelWrapper:
    def __init__(self):
        self.batch_size = 64
        self.setCancelFlag(False)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model = DefaultModel()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

    def setCancelFlag(self, flag):
        self.cancel_flag = flag

    def downloadTrainSet(self):

        try:
            self.mnist_trainset = datasets.MNIST(root='mnist_data_train/', 
                                                train=True, 
                                                transform=transforms.ToTensor(), 
                                                download=True)
        except:
            print("Couldn't download train set, try again")

        self.train_loader = data.DataLoader(dataset=self.mnist_trainset,
                                            batch_size=self.batch_size,
                                            shuffle=True)

    def downloadTestSet(self):
        try:
            self.mnist_testset = datasets.MNIST(root='mnist_data_test/', 
                                                train=False, 
                                                transform=transforms.ToTensor(),
                                                download=True)
        except:
            print("Couldn't download test set, try again")

        self.test_loader = data.DataLoader(dataset=self.mnist_testset,
                                          batch_size=self.batch_size,
                                          shuffle=False)

    def trainEpoch(self):      
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cancel_flag == True:
                break

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def testModel(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            # sum up batch loss
            test_loss += self.criterion(output, target).item()
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
    
        correct = 100 * correct / len(self.test_loader.dataset)
        print(correct)
        return test_loss, trunc(correct.item())
        
    def predictDigit(self):
        # Load the image which has already been processed by either processDrawnImage() or processRandImage()
        im = Image.open('images\processedimage.png').convert("L")
        imArray = np.array(im).astype("float32")

        # Normalise colour values to be between 0 and 1
        imArray /= 255

        # Convert to tensor and then add two more dimensions to have the right format to be batch-loaded
        imTensor = torch.from_numpy(imArray)
        imTensor = torch.unsqueeze(imTensor,0)
        imTensor = torch.unsqueeze(imTensor,0)

        # Feed manipulated image into the model then softmax to normalise the model outputs to between 0 and 1 
        probArray = F.softmax(self.model(imTensor), 1).detach().numpy()

        # Convert to percentage
        probArray *= 100 

        # Remove the extra dimensions added earlier
        probArray  = np.squeeze(probArray, 0)

        digit = np.argmax(probArray)

        return digit, probArray

    def loadModel(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            print("Loaded model")
        except FileNotFoundError:
            print("Selected model does not exist")

    def processDrawnImage(self, canvas_side_length):
        # Open the image as an Image object
        im = Image.open('images\loadedimage.png').convert("L")
        arr = np.array(im)
        proImage = arr

        # find the columns/rows where the leftmost/rightmost/highest/lowest pixels reside
        found_top = False
        found_bottom = False
        highest_y = 0
        lowest_y = canvas_side_length - 1
        for row in range(canvas_side_length):
            rowHasPixel = False
            for col in range(canvas_side_length):
                if proImage[row, col] < 128:
                    if found_top == False:
                        highest_y = row
                        found_top = True
                    rowHasPixel = True
                    break
            if rowHasPixel == False and found_top == True and found_bottom == False:
                lowest_y = row

                found_bottom = True
        
        found_left = False
        found_right = False
        leftmost_x = 0
        rightmost_x = canvas_side_length - 1
        for col in range(canvas_side_length):
            colHasPixel = False
            for row in range(canvas_side_length):
                if proImage[row, col] < 128:
                    if found_left == False:
                        leftmost_x = col
                        found_left = True
                    colHasPixel = True
            if colHasPixel == False and found_left == True and found_right == False:
                rightmost_x = row
                found_right = True

        # find the rightmost x value again since for some reason it doesn't work in the above code
        found_right  = False
        for col in range(canvas_side_length):
            for row in range(canvas_side_length):
                if proImage[row, canvas_side_length - 1 - col] < 128:
                    if found_right == False:
                        rightmost_x = canvas_side_length - col - 1
                        found_right = True

        slicedWidth = rightmost_x - leftmost_x
        slicedHeight = - highest_y + lowest_y
        sliceArray = np.zeros((slicedHeight + 1, slicedWidth + 1))

        # Cut off the empty borders of the image
        for row in range(slicedHeight):
            for col in range(slicedWidth):
                if proImage[row + highest_y, col + leftmost_x] == 255:
                    sliceArray[row,col] = 255

        # The next problem is that resize() will squash whatever our newly sliced aspect ratio is into a 1:1 image, so 1 is never detected
        # need to add padding to
        size = max(slicedWidth, slicedHeight)
        paddedArray = np.ones((size, size))
        paddedArray *= 255
        if slicedWidth < slicedHeight:
            for row in range(slicedHeight):
                for col in range(slicedWidth):
                    if sliceArray[row,col] == 0:
                        paddedArray[row, int(col + (slicedHeight - slicedWidth) / 2)] = 0

        if slicedWidth > slicedHeight:
            for row in range(slicedHeight):
                for col in range(slicedWidth):
                    if sliceArray[row,col] == 0:
                        paddedArray[int(row + (slicedWidth - slicedHeight) / 2), col] = 0

                    
        
        # make a "thumbnail" version of the image to get it to the right size for MNIST
        sliceIm = Image.fromarray(paddedArray).convert("L")
        size = (20,20)
        sliceIm = sliceIm.resize(size, Image.NEAREST).filter(ImageFilter.SHARPEN)

        thumbnailArray = np.array(sliceIm)

        # make a new image with the 4px borders like MNIST has
        borderedImage = np.zeros((28,28))
        # make borders white
        for row in range(4):
            for col in range(28):
                borderedImage[row,col] = 255
                borderedImage[row + 24, col] = 255
        for col in range(4):
            for row in range(28):
                borderedImage[row,col] = 255
                borderedImage[row, col + 24] = 255

        # map the thumbnailed image into the centre of the new bordered image
        for row in range(20):
            for col in range(20):
                if thumbnailArray[row, col] == 255:
                    borderedImage[row+4, col+4] = 255

        borderedImage = Image.fromarray(borderedImage).convert("RGB")
        borderedImage = ImageOps.invert(borderedImage)
        borderedImage.save('images\processedimage.png')

    def processRandImage(self):
        image = Image.open('images\loadedimage.png').convert("L")
        image_invert = ImageOps.invert(image) # Inverts the image
        image_invert = image_invert.resize((28, 28)) # Resizes the image to match MNIST Dataset
        image_invert.save('images\processedimage.png')


# The model given to us in the lab
class DefaultModel(nn.Module):
    def __init__(self):
        super(DefaultModel, self).__init__()
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
class WithDropOutModel(nn.Module):
    def __init__(self):
        super(WithDropOutModel, self).__init__()
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
        
        
