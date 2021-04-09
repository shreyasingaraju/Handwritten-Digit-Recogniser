import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, QGridLayout, QLabel, QFileDialog, QMainWindow, QAction, qApp
from PyQt5.QtCore import QBasicTimer, QPoint
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QColor, QIcon

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

    def trainModelDialog(self, s):
        dialog = TrainingWindow()

class TrainingWindow(QWidget):
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
