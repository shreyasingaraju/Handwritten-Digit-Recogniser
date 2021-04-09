import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, QGridLayout, QLabel, QFileDialog
from PyQt5.QtCore import QBasicTimer, QPoint
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap

class ProjectGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        grid.addWidget(QPushButton('TODO: make this a title instead of a button'),0,0)
        
        self.drawing_box = QLabel()
        grid.addWidget(self.drawing_box,1,0)
        self.pen = QPen()
        self.pen.setWidth(5)
        self.canvas = QPixmap("./testimage.png")
        self.drawing_box.setPixmap(self.canvas)

        clear_button = QPushButton("Clear")
        grid.addWidget(clear_button,0,1)

        self.setWindowTitle('QGridLayout')
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
        # Save the image when the user releases the mouse
        # Doesn't currently work, possibly because I'm drawing on the Qlabel rather than the QPixMap or something idk
        self.canvas.save("drawnimage.png")
        print("Saved as drawnimage.png (Saving not currently working)")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProjectGUI()
    sys.exit(app.exec_())
