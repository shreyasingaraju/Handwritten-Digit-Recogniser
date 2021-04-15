import sys
from PyQt5.QtWidgets import QApplication, QSizePolicy, QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QProgressBar, QGridLayout, QLabel, QFileDialog, QMainWindow, QAction, qApp, QTextBrowser
from PyQt5.QtCore import QBasicTimer, QPoint
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QColor, QIcon

import torch
import torchvision
import torchvision.datasets as datasets

import matplotlib.pyplot as plot
import numpy as np

from modules.ProjectModel import ProjModel

class ProjectGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Make a model class which contains the datasets and training data
        self.model = ProjModel

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
        model_button = QPushButton("Model")
        subgrid.addWidget(model_button,2,0)
        recognise_button = QPushButton("Recognise")
        subgrid.addWidget(recognise_button,3,0)
        grid.addWidget(subwidget,1,1)
    

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

    # trainModelDialog() creates a dialog box when the user clicks File>Train Model
    # When open, the user can press buttons to download MNIST, train the dataset and close the window.
    def trainModelDialog(self, s):
        dialog  = TrainDialog()
        dialog.parent = self
        dialog.exec_()

    # This method is called when 'View Training Images' is pressed
    def viewTrainingImagesDialog(self):
        imgDialog  = ImagesDialog()
        imgDialog.setMode('train')
        imgDialog.exec_()

    # This method is called when 'View Testing Images' is pressed
    def viewTestingImagesDialog(self):
        imgDialog  = ImagesDialog()
        imgDialog.setMode('test')
        imgDialog.exec_()

class TrainDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setModal(True)
        self.setWindowTitle("Dialog")
        #self.layout.addWidget(QLabel("Status"))

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setGeometry(300, 300,300, 300)

        # Added text box in dialog window
        self.text = " "
        self.textbox = QTextBrowser(self)

        self.layout.addWidget(self.textbox)
        self.textbox.setText(self.text)

        # Added progress bar
        self.pbar = QProgressBar(self)
        self.layout.addWidget(self.pbar)

        # This block provides buttons for downloading the dataset, training and closing the window and arranges them into a horizontal grid
        dl_trn_cncl_grid = QHBoxLayout()
        dl_trn_cncl_widg = QWidget(self)
        dl_trn_cncl_widg.setLayout(dl_trn_cncl_grid)
        dl_mnist_button =  QPushButton("Download MNIST")
        dl_trn_cncl_grid.addWidget(dl_mnist_button)
        dl_mnist_button.clicked.connect(self.downloadMnist)
        trn_button = QPushButton("Train")
        dl_trn_cncl_grid.addWidget(trn_button)
        trn_button.clicked.connect(self.train)
        cncl_button = QPushButton("Cancel")
        dl_trn_cncl_grid.addWidget(cncl_button)
        cncl_button.clicked.connect(self.cancel)
        self.layout.addWidget(dl_trn_cncl_widg)

    # This method downloads the MNIST dataset when button is pressed
    def downloadMnist(self, s):

        print("Downloading") # Can be removed later
        
        self.textbox.append("Downloading train dataset...")
        model.downloadTrainSet()

        self.textbox.append("\nDownloading test dataset...")
        model.downloadTestSet()

    # This method trains the DNN Model using the dataset
    def train(self, s):
        # Prints text when training begins
        self.textbox.append("Training...\n")
        model.TrainModel()
        
        # Accuracy:") # Need to implement accuracy %
        print("Training") # Can be removed later

    # This method cancels the downloading or training at anytime 
    def cancel(self, s):
        # Clears dialog box when cancelled
        self.textbox.clear()
        print("Canceled") # Can be removed later

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProjectGUI()
    global model
    model = ProjModel()
    sys.exit(app.exec_())
