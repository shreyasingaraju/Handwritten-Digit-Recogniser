import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QWidget, QPushButton, QProgressBar, QGridLayout, QLabel, QFileDialog, QMainWindow, QAction, qApp, QTextBrowser
from PyQt5.QtCore import QBasicTimer, QPoint
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QColor, QIcon

import torch
import torchvision
import torchvision.datasets as datasets

class ProjectGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        # self.setLayout(grid)
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

        openviewTrainingImagesDialog = QAction(QIcon('D:\workspace\COMPSYS302_PyQt5\exit.png'), 'view Training Images', self)
        openviewTrainingImagesDialog.triggered.connect(self.viewTrainingImagesDialog)

        openviewTestingImagesDialog = QAction(QIcon('D:\workspace\COMPSYS302_PyQt5\exit.png'), 'view Testing Images', self)
        openviewTestingImagesDialog.triggered.connect(self.viewTestingImagesDialog)

        viewmenu = menubar.addMenu('&View')
        viewmenu.addAction(openviewTrainingImagesDialog)
        viewmenu.addAction(openviewTestingImagesDialog)

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
        # Doesn't currently work, possibly because I'm drawing on the Qlabel rather than the QPixMap or something idk
        self.canvas.save("drawnimage.png")
        print("Saved as drawnimage.png (Saving not currently working)")
    
    # This method clears drawing on the canvas when 'clear' button is pressed
    def clear_clicked(self):
        self.drawing_box.setPixmap(self.canvas)
        # Prints a statement - can be removed later
        print("Clear button clicked")

    # trainModelDialog() creates a dialog box when the user clicks File>Train Model
    # When open, the user can press buttons to download MNIST, train the dataset and close the window.
    def trainModelDialog(self, s):
        dialog  = TrainDialog()
        dialog.exec_()

    # This method is called when 'view Training Images' is pressed
    def viewTrainingImagesDialog(self, s):
        # Needs to be implemented
        dialog  = TrainingImagesDialog()

    # This method is called when 'view Testing Images' is pressed
    def viewTestingImagesDialog(self, s):
        # Needs to be implemented
        dialog  = TestingImagesDialog()

class TrainDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setModal(True)
        self.setWindowTitle("Dialog")
        #self.layout.addWidget(QLabel("Status"))

        # Added text box in dialog window
        self.textbox = QTextBrowser(self)
        self.textbox.resize(400,200)
        self.textbox.move(15,15)

        # Added progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(15, 230, 450, 15)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setGeometry(300, 300,300, 300)
        self.show()

        # This block provides buttons for downloading the dataset, training and closing the window
        dl_trn_cncl_grid = QGridLayout()
        dl_trn_cncl_widg = QWidget(self)
        dl_trn_cncl_widg.setLayout(dl_trn_cncl_grid)
        dl_mnist_button =  QPushButton("Download MNIST")
        dl_trn_cncl_grid.addWidget(dl_mnist_button, 0, 0)
        dl_mnist_button.clicked.connect(self.downloadMnist) # Connects to download button to downloadMnist method
        trn_button = QPushButton("Train")
        dl_trn_cncl_grid.addWidget(trn_button, 0, 1)
        trn_button.clicked.connect(self.train) # Connects to train button to train method
        cncl_button = QPushButton("Cancel")
        dl_trn_cncl_grid.addWidget(cncl_button, 0, 2)
        cncl_button.clicked.connect(self.cancel) # Connects to cancel button to cancel method train method
        self.layout.addWidget(dl_trn_cncl_widg)

    # This method downloads the MNIST dataset when button is pressed
    def downloadMnist(self, s):
        # Prints text when download begins
        self.textbox.setText("Downloading train dataset...\nDownloading test dataset...")
        # Unsure if this actually works (better way to implement this)
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None) 
        print("Downloading") # Can be removed later

    # This method trains the DNN Model using the dataset
    def train(self, s):
        # Prints text when training begins
        self.textbox.append("Training...\nAccuracy:") # Need to implement accuracy %
        print("Training") # Can be removed later

    # This method cancels the downloading or training at anytime 
    def cancel(self, s):
        # Clears dialog box when cancelled
        self.textbox.clear()
        print("Canceled") # Can be removed later

class TrainingImagesDialog(QWidget): # Needs to be implemented
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        self.setGeometry(200, 200, 200, 200)
        self.show()

class TestingImagesDialog(QWidget): # Needs to be implemented
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        self.setGeometry(200, 200, 200, 200)
        self.show()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProjectGUI()
    sys.exit(app.exec_())
