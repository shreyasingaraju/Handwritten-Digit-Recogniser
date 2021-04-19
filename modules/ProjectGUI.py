import sys
from PyQt5.QtWidgets import QApplication, QSizePolicy, QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QProgressBar, QGridLayout, QLabel, QFileDialog, QMainWindow, QAction, qApp, QTextBrowser, QComboBox
from PyQt5.QtCore import QBasicTimer, QPoint, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QColor, QIcon
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from modules.ProjectModel import ProjModel

# The ProjectGUI class is the main window of the application, and contains the drawing and recognising interface, 
# as well as a menubar which lets the user open the other dialog boxes for training and viewing
class ProjectGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Make a model instance which contains the datasets and training data
        global model
        model = ProjModel()
        

        grid = QGridLayout()
        window = QWidget(self)
        window.setLayout(grid)
        self.setCentralWidget(window)

        # Add the file menubar
        exitAction = QAction(QIcon('D:\workspace\COMPSYS302_PyQt5\exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        openTrainModelDialog = QAction(QIcon('D:\workspace\COMPSYS302_PyQt5\exit.png'), 'Train Model', self)
        openTrainModelDialog.setShortcut('Ctrl-t')
        openTrainModelDialog.setStatusTip('Train the Model')
        openTrainModelDialog.triggered.connect(self.trainModelDialog)

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(exitAction)
        filemenu.addAction(openTrainModelDialog)


        # Add the view menubar
        openViewTrainingImagesDialog = QAction(QIcon('.\exit.png'), 'view Training Images', self)
        openViewTrainingImagesDialog.triggered.connect(self.viewTrainingImagesDialog)

        openViewTestingImagesDialog = QAction(QIcon('.\exit.png'), 'view Testing Images', self)
        openViewTestingImagesDialog.triggered.connect(self.viewTestingImagesDialog)

        viewmenu = menubar.addMenu('&View')
        viewmenu.addAction(openViewTrainingImagesDialog)
        viewmenu.addAction(openViewTestingImagesDialog)

        grid.addWidget(QLabel("Drawing Box"),0,0)
        
        self.drawing_box = QLabel()
        grid.addWidget(self.drawing_box,1,0)
        self.pen = QPen()
        self.pen.setWidth(5)
        self.canvas = QPixmap(280,280)
        self.canvas.fill(QColor(255,255,255))
        self.drawing_box.setPixmap(self.canvas)

        # This block sets up the right hand side buttons in a grid nested inside the central grid, directly to the right of the drawing box
        # TODO: improve variable naming for this section
        subgrid = QGridLayout()
        subwidget = QWidget(self)
        subwidget.setLayout(subgrid)
        clear_button = QPushButton("Clear")
        subgrid.addWidget(clear_button,0,0)
        clear_button.clicked.connect(self.clear_clicked) #connects to push button to clear method

        random_button = QPushButton("Random")
        subgrid.addWidget(random_button,1,0)
        random_button.clicked.connect(self.random_clicked) #connects to push button to random method

        self.model_button = QComboBox(self)
        models = ["Model 1", "Model 2", "Model 3", "Model 4"]
        self.model_button.setEditable(True)
        self.model_button.addItems(models)
        self.model_button.currentIndexChanged.connect(lambda: self.load(models[self.model_button.currentIndex()]))
        # Load the first model by default
        model.loadNet(models[0])
        # For text center align 
        line_edit = self.model_button.lineEdit()
        line_edit.setAlignment(Qt.AlignCenter)
        line_edit.setReadOnly(True)
        subgrid.addWidget(self.model_button,2,0)

        recognise_button = QPushButton("Recognise")
        subgrid.addWidget(recognise_button,3,0)
        grid.addWidget(subwidget,1,1)
        recognise_button.clicked.connect(self.recognise_clicked) #connects to push button to recognise method
        self.setWindowTitle('Digit Recogniser')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    # In order to draw numbers such as 4 we have to be able to "lift the pen" off of the canvas
    def mousePressEvent(self, e):
        self.last_point = QPoint(e.x(),e.y())

    # This method is responsible for drawing when the user clicks and drags the mouse
    def mouseMoveEvent(self, e):
        painter = QPainter(self.drawing_box.pixmap())
        painter.setPen(self.pen)
        self.this_point = QPoint(e.x(),e.y())
        painter.drawLine(self.this_point, self.last_point)
        self.last_point = QPoint(e.x(),e.y())
        painter.end()
        self.update()

    def mouseReleaseEvent(self, e):
        self.update()
        # Save the image when the user releases the mouse
        img = QPixmap(self.drawing_box.pixmap())
        img.save("drawnimage.png")
        print("Saved as drawnimage.png")
    
    # This method clears drawing on the canvas when 'clear' button is pressed
    def clear_clicked(self):
        self.drawing_box.setPixmap(self.canvas)
        # Prints a statement - can be removed later
        print("Clear button clicked")
    
    def random_clicked(self):
        trainloader = torch.utils.data.DataLoader(model.mnist_trainset, batch_size=64, shuffle=True)
        dataiter = iter(trainloader) # creating a iterator
        images, labels = dataiter.next() # creating images for image and lables for image number (0 to 9) 
        #print(images.shape)
        #print(labels.shape)
        plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
        plt.axis('off')
        plt.savefig("drawnimage.png")

    def model_clicked(self):
        # Connect to loading
        print("nothing happens")

    def recognise_clicked(self):
        image = Image.open('drawnimage.png').convert('RGB') # Converts handrawn digit to RGB 
        image_invert = ImageOps.invert(image) # Inverts the image
        image_invert = image_invert.resize((28, 28)) # Resizes the image to match MNIST Dataset
        image_invert.save('invertedimage.png') # Saves the new processed image

        model.predictDigit(image_invert)
        
    # trainModelDialog() creates a dialog box when the user clicks File>Train Model
    # When open, the user can press buttons to download MNIST, train the dataset and close the window.
    def trainModelDialog(self, s):
        dialog  = TrainDialog()
        dialog.parent = self
        dialog.exec_()

    # This method is called when 'View Training Images' is pressed
    def viewTrainingImagesDialog(self):
        try:
            imgDialog  = ImagesDialog()
            imgDialog.setMode('train')
            imgDialog.exec_()
        except AttributeError:
            print("Dataset not downloaded. Go to file>Train Model")

    # This method is called when 'View Testing Images' is pressed
    def viewTestingImagesDialog(self):
        try:
            imgDialog  = ImagesDialog()
            imgDialog.setMode('test')
            imgDialog.exec_()
        except AttributeError:
            print("Dataset not downloaded. Go to file>Train Model")

    def load(self, path):
        model.loadNet(path)

# TrainDialog is opened when the user selects Train Model from the file menubar.
# It allows the user to download MNIST, select and load or train the model (NOTE: implement switching models)
class TrainDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # modelPath is the path to the saved state_dict of the model to be passed into load()
        # TODO: implement switching this model using a combo box

        self.setModal(True)
        self.setWindowTitle("Dialog")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setGeometry(300, 300,300, 300)

        # Added text box in dialog window
        self.text = "Welcome"
        self.textbox = QTextBrowser(self)
        self.layout.addWidget(self.textbox)
        self.textbox.setText(self.text)

        # Added progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setMaximum(100)
        self.pbar.setMinimum(0)
        self.layout.addWidget(self.pbar)

        # This block provides buttons for downloading the dataset, training and closing the window and arranges them into a horizontal grid
        button_grid = QHBoxLayout()
        button_widg = QWidget(self)
        button_widg.setLayout(button_grid)
        self.dl_mnist_button =  QPushButton("Download MNIST")
        button_grid.addWidget(self.dl_mnist_button)
        self.dl_mnist_button.clicked.connect(self.downloadMnist)


        self.trn_button = QPushButton("Train")
        button_grid.addWidget(self.trn_button)
        self.trn_button.clicked.connect(self.train)

        # Cancel button, stops the training if it's in progress and clears the textbox
        self.cncl_button = QPushButton("Cancel")
        button_grid.addWidget(self.cncl_button)
        self.cncl_button.clicked.connect(self.cancel)
        self.cncl_button.clicked.connect(self.textbox.clear)

        # Add the buttons to the overall layout of the dialog
        self.layout.addWidget(button_widg)
        

    # This method downloads the MNIST dataset when button is pressed
    def downloadMnist(self, s):

        print("Downloading") # Can be removed later
        self.textbox.append("Downloading train dataset...")
        model.downloadTrainSet()
        self.textbox.append("Downloading test dataset...")
        model.downloadTestSet()
        self.textbox.append("Done!")

    # This method trains the DNN Model using the dataset
    def train(self, s):
        model.setCancelFlag(False)
        try:
            # Prints text when training begins NOTE: currently the textbox.append gets run after TrainModel() finishes somehow?
            self.textbox.append("Training...")
            self.thread = QThread()
            self.worker = TrainingWorker()
            self.worker.moveToThread(self.thread)

            # Connect signals and slots here
            self.thread.started.connect(self.worker.workerTrain)
            self.worker.progress.connect(self.reportProgress)
            self.thread.finished.connect(self.cancel)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            # Disable the buttons that aren't meant to be used
            self.dl_mnist_button.setEnabled(False)
            self.trn_button.setEnabled(False)

            # Start the thread
            self.thread.start()
            # Accuracy:") # Need to implement accuracy %
        except AttributeError:
            print("Please download MNIST first")

    # This method is called after each epoch, and returns the epoch/loss/accuracy and updates the progress bar
    def reportProgress(self, result_tuple):
        self.textbox.append("Epoch: " + str(result_tuple[0] + 1))
        self.textbox.append("Loss: " + str(result_tuple[1]))
        self.textbox.append("Accuracy: " + str(result_tuple[2]))
        self.pbar.setValue((result_tuple[0] + 1)* 100 / num_epochs)

    # This method cancels the training/testing at any time 
    def cancel(self):

        try:
            model.setCancelFlag(True)
            self.dl_mnist_button.setEnabled(True)
            self.trn_button.setEnabled(True)
        except AttributeError:
            print("Cancel clicked before model downloaded, not really an issue")

# This class is moved into a thread and then runs the training so that the user can keep interacting with the GUI during training
# Adapted from https://realpython.com/python-pyqt-qthread/
class TrainingWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    update = pyqtSignal
    

    def workerTrain(self):
        global num_epochs
        num_epochs = 2

        for epoch in range(num_epochs):
            # Train the network for one epoch
            model.trainEpoch()
            # Test the net and put the results in a tuple which is broadcast as a signal back to the TrainDialog textbox
            if model.cancel_flag == True:
                self.finished.emit()
                print(model.cancel_flag)
                print("Break in workerTrain")
                break
            test_loss, correct = model.testModel()
            result_tuple = (epoch, test_loss, correct)
            self.progress.emit(result_tuple)
        self.finished.emit()
        print("Finished")
        torch.save(model.net.state_dict(), 'Model 1')


# This class shows the training images or the testing images. mode is passed into initUI() and represents whether we want to display the training or testing images
# The dialog shows 100 images at a time, and can be navigated by clicking Next or Previous to view the next or last 100 images respectively
class ImagesDialog(QDialog):
    
    def __init__(self):
        super().__init__()
        
    def initUI(self):
        # self.layout is a vertical box structure, to which we add the grid of images, the next/prev buttons and the page number
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # self.images is a grid of 100 x 100 images from the dataset
        self.page = 0   # This is the page number
        self.images = QWidget()
        self.grid = QGridLayout()
        self.images.setLayout(self.grid)
        self.layout.addWidget(self.images)

        # next_prev_grid contains the next and previous buttons that change the page number 
        next_prev_grid = QHBoxLayout()
        next_prev_widg = QWidget(self)
        next_prev_widg.setLayout(next_prev_grid)
        prev_button = QPushButton("Previous")
        next_button =  QPushButton("Next")
        next_prev_grid.addWidget(prev_button)
        next_prev_grid.addWidget(next_button)
        prev_button.clicked.connect(self.prevPage)
        next_button.clicked.connect(self.nextPage)
        self.layout.addWidget(next_prev_widg)

        self.resize(300, 400)
        
        self.pageNum = QLabel()
        self.pageNum.setText(str(self.page))
        self.layout.addWidget(self.pageNum)
        self.loadImages()

        self.show()

    # Called in ProjectGUI.viewTrainingImagesDialog or ProjectGUI.viewTestingImagesDialog and sets the mode of the dialog depending on which option the user selected
    def setMode(self, mode):
        self.mode = mode
        self.initUI()

    # Increments the page number if not at max, then loads the new images
    def nextPage(self):
        if (self.mode == 'train' and self.page < 599) or (self.mode == 'test' and self.page < 99):
            self.page += 1
            self.loadImages()
            self.pageNum.setText(str(self.page))

    # Decrements the page number if not at max, then loads the new images
    def prevPage(self):
        if self.page > 0:
            self.page -= 1
            self.loadImages()
            self.pageNum.setText(str(self.page))

    # Loads the range of images determined by the page number and arranges them into the grid
    def loadImages(self):
            for i in range(0,9):
                for j in range(0,9):
                    label = QLabel()
                    if self.mode == 'train':
                        imgArr = np.squeeze(model.mnist_trainset[i+j + 100 * self.page][0])
                    elif self.mode == 'test':
                        imgArr = np.squeeze(model.mnist_testset[i+j + 100 * self.page][0])
                    plot.imsave('temp_img.png', imgArr)
                    img = QPixmap('temp_img.png')
                    label.setPixmap(img)
                    self.grid.addWidget(label, i, j)